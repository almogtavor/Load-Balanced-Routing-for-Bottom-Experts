#!/usr/bin/env python3
"""Independent four-arm serving confirmation of DeepEP HT SM budget and routing.

  python experiments/serve_sms_confirm.py --model MODEL --text-file CORPUS \
      --output CAMPAIGN --selection-source REPLAY64.json

Creates CAMPAIGN/sms-confirm: stock_s20, tuned_s20, stock_s64, tuned_s64 in four
fresh Williams-balanced sweeps (16 launches). Tuned is frozen rows4_direct,
keep=6, penalty=1. Every arm loads the SAME worker extension and requests the
same 1GiB NVL buffer. Evidence must establish all four EP ranks, exact requested/
manager/Buffer SM budget, same requested/allocated bytes, no ubatching and valid
linear one-node EP4 scope. A healthy warmed server corroborates preallocation
evidence. All measured rounds are retained, including observed JIT events.

Fixed natural-text pure prefill: 4096 input / 1 output, concurrency64,128 requests,
DP4/TP1/EP, HT, graphs NONE, batch16384, three complete warmups, three measured
rounds. No profiler. Independent launch ratios use n=4 paired sweeps, df=3 t CIs.
Replay/selection and earlier confirmation data are never pooled. --plan-only
does not load a tokenizer, query GPU metadata or start a server.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import analyze_serving as statistics_tools
import serve_focus as focus
import serve_profile as common

EXTENSION = "deepep_sms_extension.DeepEPSMSWorkerExtension"
ARMS = {"stock_s20": {"keep": None, "penalty": None, "selector": None, "num_sms": 20},
        "tuned_s20": {"keep": 6, "penalty": 1, "selector": "rows4_direct", "num_sms": 20},
        "stock_s64": {"keep": None, "penalty": None, "selector": None, "num_sms": 64},
        "tuned_s64": {"keep": 6, "penalty": 1, "selector": "rows4_direct", "num_sms": 64}}
CONTRASTS = [("deployment_sm_gain", "stock_s64", "stock_s20"),
             ("placement_at_64", "tuned_s64", "stock_s64"),
             ("placement_at_20", "tuned_s20", "stock_s20"),
             ("combined", "tuned_s64", "stock_s20")]
SHAPE = common.Shape("prefill", 4096, 1, 64, 128)
BUFFER_BYTES = 1024 * 1024 * 1024
ORIGINAL_SERVE = common.serve
ORIGINAL_ACTIVE_CHECK = focus.check_active
ACTIVE_EVIDENCE_DIRECTORY = None
WORKER_PYTHONPATH = "/work/router/experiments"


def schedule(seed):
    labels = list(ARMS)
    random.Random(seed).shuffle(labels)
    # Even-order Williams design: positions and all 12 ordered within-sweep
    # predecessor pairs occur once across the four sweeps.
    return [{"repeat": repeat, "arms": [labels[(index + repeat) % 4] for index in (0, 1, 3, 2)]}
            for repeat in range(4)]


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("deepep_sms_extension.py"),
             Path(__file__).with_name("analyze_serving.py")]
    return {**focus.source_hashes(), **{str(path.relative_to(focus.ROOT)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def configure_helpers(worker_pythonpath):
    global WORKER_PYTHONPATH
    WORKER_PYTHONPATH = worker_pythonpath
    focus.ARM_DEFINITIONS = ARMS
    focus.CONTROLLED_SERVER_ENVIRONMENT = {"VLLM_DEEPEP_BUFFER_SIZE_MB": "1024",
        "NVSHMEM_QP_DEPTH": "4096", "OMP_NUM_THREADS": "1"}
    common.ARMS = {arm: (value["keep"], value["penalty"]) for arm, value in ARMS.items()}
    common.server_command = server_command
    common.arm_environment = arm_environment
    common.serve = verified_serve
    focus.check_active = check_active


def server_command(args, trace_dir):
    if trace_dir is not None:
        raise RuntimeError("SM confirmation requires profiling off")
    return focus.server_command(args, None) + ["--compilation-config={\"cudagraph_mode\":\"NONE\"}",
        "--worker-extension-cls=" + EXTENSION]


def arm_environment(arm):
    env = focus.arm_environment(arm)
    env["PLACEMENT_DEEPEP_NUM_SMS"] = str(ARMS[arm]["num_sms"])
    if ACTIVE_EVIDENCE_DIRECTORY is None:
        raise RuntimeError("A unique absolute evidence directory is required for each launch")
    env["PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR"] = str(ACTIVE_EVIDENCE_DIRECTORY)
    paths = [WORKER_PYTHONPATH, *env.get("PYTHONPATH", "").split(os.pathsep)]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in paths if path))
    return env


def check_active(arm, log, dp):
    ORIGINAL_ACTIVE_CHECK(arm, log, dp)
    text = log.read_text(errors="replace")
    if ARMS[arm]["keep"] is None and ("placement-aware routing active" in text or "experimental selector=" in text):
        raise RuntimeError(f"Stock routing was unexpectedly modified in {arm}")
    if focus.runtime_evidence(log)["observed_cudagraph_modes"] != ["NONE"]:
        raise RuntimeError(f"Expected actual graph mode NONE in {log}")


def validate_evidence(records, arm):
    if len(records) != 4 or {record.get("ep_rank") for record in records} != set(range(4)):
        raise RuntimeError("SM evidence must contain exactly one record for every EP rank0–3")
    expected = ARMS[arm]
    signatures = []
    for record in records:
        for field in ("requested_sms", "manager_sms_ceiling", "buffer_num_sms"):
            if record.get(field) != expected["num_sms"]:
                raise RuntimeError(f"Rank{record.get('ep_rank')} {field} does not match requested SM budget")
        if (record.get("kind") != "deepep_ht_sm_startup" or record.get("ep_world_size") != 4
                or record.get("internode") is not False or record.get("deep_ep") != "2.0.0"
                or record.get("rdma_allocated_bytes") != 0 or record.get("num_qps_per_rank") != 1
                or record.get("enable_dbo") is not False or record.get("use_ubatching") is not False
                or not isinstance(record.get("ubatch_size"), int) or not 0 <= record["ubatch_size"] <= 1):
            raise RuntimeError("Invalid one-node EP4/non-ubatched DeepEP HT scope evidence")
        if (record.get("nvl_requested_bytes") != BUFFER_BYTES or record.get("nvl_allocated_bytes") != BUFFER_BYTES
                or record.get("nvl_auto_expanded") is not False):
            raise RuntimeError("Every arm/rank must request and allocate the same1GiB NVL buffer")
        for field in ("nvl_dispatch_size_hint_bytes", "nvl_combine_size_hint_bytes"):
            value = record.get(field)
            if not isinstance(value, int) or not 0 < value <= BUFFER_BYTES:
                raise RuntimeError("Invalid or oversized DeepEP NVL buffer hint")
        if record.get("buffer_allocation_has_not_run_yet") is not True:
            raise RuntimeError("Expected explicit preallocation scope for SM evidence")
        if record.get("gpu_physical_sms", 0) < expected["num_sms"]:
            raise RuntimeError("Requested SM budget exceeds physical GPU capacity")
        if (not isinstance(record.get("hidden_size"), int) or record["hidden_size"] <= 0
                or not isinstance(record.get("model_dtype"), str)
                or not isinstance(record.get("padded_hidden_bytes"), int)
                or record["padded_hidden_bytes"] <= 0 or record["padded_hidden_bytes"] % 512):
            raise RuntimeError("Missing or invalid model/hidden-byte allocation signature")
        if (record.get("placement_keep") != (None if expected["keep"] is None else "6")
                or record.get("placement_selector") != expected["selector"]):
            raise RuntimeError("SM evidence disagrees with stock/tuned routing identity")
        signatures.append(tuple(record.get(key) for key in
            ("hidden_size", "model_dtype", "padded_hidden_bytes", "nvl_allocated_bytes", "ep_world_size")))
    if len(set(signatures)) != 1:
        raise RuntimeError("SM allocation/model signatures differ across EP ranks")
    return {"valid_all_ep_ranks": True, "requested_sms": expected["num_sms"],
        "nvl_requested_and_allocated_bytes": BUFFER_BYTES, "rank_records": sorted(records, key=lambda row: row["ep_rank"]),
        "scope_note": "Evidence precedes buffer construction; healthy server and completed full warmups corroborate allocation success. Every arm uses the same extension and1GiB buffer plan."}


def collect_evidence(directory, arm):
    paths = sorted(directory.glob("rank*_pid*.json"))
    result = validate_evidence([json.loads(path.read_text()) for path in paths], arm)
    result["files"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    return result


@contextlib.contextmanager
def verified_serve(args, arm, directory, trace_dir):
    if source_hashes() != args.frozen_source_sha256:
        raise RuntimeError("Frozen benchmark source changed before launch; use a new campaign")
    with ORIGINAL_SERVE(args, arm, directory, trace_dir) as process:
        collect_evidence(ACTIVE_EVIDENCE_DIRECTORY, arm)  # Healthy startup, before any measured workload.
        path = directory / "server.json"
        data = json.loads(path.read_text())
        data.update(worker_extension=EXTENSION, controlled_environment={key: arm_environment(arm).get(key)
            for key in ("PLACEMENT_DEEPEP_NUM_SMS", "PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR", "PYTHONPATH",
                        "VLLM_DEEPEP_BUFFER_SIZE_MB", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH")})
        common.atomic_json(path, data)
        yield process


def launch(args, arm, repeat, prompts):
    global ACTIVE_EVIDENCE_DIRECTORY
    directory = args.output / f"r{repeat:02d}_{arm}"
    path = directory / "result.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        data = json.loads(path.read_text())
        validate_evidence(data["sms_evidence"]["rank_records"], arm)
        print(f"[resume] {directory.name}", flush=True)
        return
    ACTIVE_EVIDENCE_DIRECTORY = directory / f"sms-evidence-{time.time_ns()}"
    ACTIVE_EVIDENCE_DIRECTORY.mkdir(parents=True)
    try:
        focus.measure_launch(args, arm, repeat, SHAPE, prompts)
        evidence = collect_evidence(ACTIVE_EVIDENCE_DIRECTORY, arm)
        data = json.loads(path.read_text())
        data["sms_evidence_directory"] = str(ACTIVE_EVIDENCE_DIRECTORY)
        data["sms_evidence"] = evidence
        data["benchmark_source_sha256"] = args.frozen_source_sha256
        common.atomic_json(path, data)
    except BaseException as error:
        if path.exists():
            data = json.loads(path.read_text())
            data.update(complete=False, sms_evidence_error=f"{type(error).__name__}: {error}",
                        sms_evidence_directory=str(ACTIVE_EVIDENCE_DIRECTORY))
            common.atomic_json(path, data)
        raise


def analyze(output):
    manifest = json.loads((output / "manifest.json").read_text())
    identity = manifest["identity"]
    plan = {"arms": list(ARMS), "shapes": [SHAPE.name], "rounds_per_shape": identity["config"]["rounds"],
            "workloads": {SHAPE.name: dataclasses.asdict(SHAPE)}}
    expected = {(row["repeat"], arm) for row in identity["schedule"] for arm in row["arms"]}
    launches, indexed, issues = [], {}, []
    for path in sorted(output.glob("r*_*/result.json")):
        row = statistics_tools.read_launch(path, plan)
        key = row["repeat"], row["arm"]
        try:
            source = json.loads(path.read_text())
            row["sms_evidence"] = validate_evidence(source.get("sms_evidence", {}).get("rank_records", []), row["arm"])
            if source.get("benchmark_source_sha256") != identity["source_sha256"]:
                raise RuntimeError("Launch source identity differs from frozen manifest")
            warmups = source.get("shapes", {}).get(SHAPE.name, {}).get("warmups", [])
            if len(warmups) != identity["config"]["warmup_rounds"] or any(
                statistics_tools.round_errors(value, dataclasses.asdict(SHAPE), index)
                for index, value in enumerate(warmups)):
                raise RuntimeError("Missing or invalid predetermined full-size warmup workload")
        except (KeyError, RuntimeError) as error:
            row["issues"].append(str(error))
            row["eligible_for_paired_analysis"] = False
        if key in indexed or key not in expected:
            row["issues"].append("Duplicate or unexpected launch identity")
            row["eligible_for_paired_analysis"] = False
            if key in indexed:
                indexed[key]["eligible_for_paired_analysis"] = False
        indexed[key] = row
        launches.append(row)
        for message in row["issues"] + [message for shape in row["shapes"].values() for message in shape["issues"]]:
            issues.append(f"{path.parent.name}: {message}")
    missing = expected - indexed.keys()
    issues += [f"Missing planned launch r{repeat:02d}_{arm}" for repeat, arm in sorted(missing)]
    comparisons = []
    for name, arm, baseline in CONTRASTS:
        for item in statistics_tools.supplemental_contrast(indexed, expected, plan, arm, baseline):
            item["contrast"] = name
            comparisons.append(item)
    complete = not issues and all(item["complete_pairs"] for item in comparisons if item["metric"] in statistics_tools.required_metrics(dataclasses.asdict(SHAPE)))
    valid = [row for row in launches if row.get("eligible_for_paired_analysis")]
    jit_launches = [f"r{row['repeat']:02d}_{row['arm']}" for row in valid
                    if row.get("jit_clean_measurements_observed") is False]
    return {"complete": complete, "scope": identity, "issues": issues, "launches": launches, "comparisons": comparisons,
        "launches_with_observed_measured_jit": jit_launches,
        "counts": {"expected_launches": 16, "valid_launches": len(valid),
            "measured_rounds": sum(shape["measured_rounds"] for row in valid for shape in row["shapes"].values()),
            "measured_requests": sum(shape["measured_requests"] for row in valid for shape in row["shapes"].values())},
        "limits": ["Selection/replay and earlier confirmation data are never pooled.",
            "Pure-prefill4096/1 has undefined TPOT/ITL and does not establish generation/decode gains.",
            "Each ratio uses independent server launches paired by sweep; n=4/df=3, not requests or rounds. Pointwise95% t intervals assume independent approximately normal log-ratios and are not adjusted for multiple endpoints.",
            "Williams balance controls launch position and within-sweep first-order predecessor; shared-node noise, time drift and carryover remain possible.",
            "Stock20 is the default SM budget under the same extension/1GiB allocation plan. The combined contrast includes both budget and routing changes; use the factorial contrasts to attribute them.",
            "Every measured round is retained. Observed JIT events remain in raw round/log evidence and limit steady-state interpretation."]}


def write_analysis(output):
    report = analyze(output)
    common.atomic_json(output / "sms_analysis.json", report)
    lines = ["DeepEP SM-budget and placement serving confirmation", "",
             f"Status: {'COMPLETE' if report['complete'] else 'INCOMPLETE'}; {report['counts']}",
             f"Launches with observed measured JIT (retained): {report['launches_with_observed_measured_jit']}", "",
             "Fixed4096in/1out,c64,n128,batch16384,DP4/TP1/EPHT,graphsNONE. All four arms use the same extension and1GiB NVL buffer; tuned means rows4_direct keep6/penalty1.", "",
             "| Contrast | Arm / baseline | Independent pairs | Throughput change |95%tCI | TTFTp50 change |95%tCI |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, arm, baseline in CONTRASTS:
        values = {item["metric"]: item for item in report["comparisons"] if item["contrast"] == name}
        cells = []
        for metric in ("output_tokens_per_second", "ttft_ms_p50"):
            item = values[metric]
            change, ci = item["geometric_mean_change_pct"], item["geometric_change_pct_ci95"]
            cells += [(f"{change:+.2f}%" if change is not None else "—"),
                      (f"[{ci[0]:+.2f}%,{ci[1]:+.2f}%]" if ci else "unavailable")]
        lines.append(f"|{name}|{arm}/{baseline}|{values['output_tokens_per_second']['n_paired_launches']}|" + "|".join(cells) + "|")
    if report["issues"]:
        lines += ["", "Completion issues:", ""] + ["- " + value for value in report["issues"]]
    lines += ["", "Interpretation limits:", ""] + ["- " + value for value in report["limits"]]
    (output / "sms_analysis.md").write_text("\n".join(lines) + "\n")
    with (output / "sms_comparisons.csv").open("w", newline="") as target:
        fields = ["contrast", "arm", "baseline", "metric", "n_paired_launches", "geometric_mean_change_pct", "ci95_low_pct", "ci95_high_pct"]
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        for item in report["comparisons"]:
            ci = item["geometric_change_pct_ci95"] or [None, None]
            writer.writerow({**{key: item[key] for key in fields if key in item}, "ci95_low_pct": ci[0], "ci95_high_pct": ci[1]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-source", type=Path)
    parser.add_argument("--selection-note", default="Synthetic SM replay selected64; independent serving confirmation")
    parser.add_argument("--worker-pythonpath", default="/work/router/experiments")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--gpu-util", type=float, default=.88)
    parser.add_argument("--boot-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.warmup_rounds) < 1 or not Path(args.worker_pythonpath).is_absolute():
        parser.error("Positive round counts and an absolute worker Python path are required")
    for name in ("text_file", "selection_source"):
        value = getattr(args, name)
        if value:
            value = value.resolve()
            if not value.is_file():
                parser.error(f"Missing {name}: {value}")
            setattr(args, name, value)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".focus-runner.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another serving runner already uses this campaign root") from None
    args.output = root / "sms-confirm"
    args.output.mkdir(parents=True, exist_ok=True)
    args.phase, args.backend, args.dp, args.batch_tokens = "sms-confirm", "deepep_high_throughput", 4, 16384
    args.max_model_len, args.max_num_seqs, args.jit_verbose = 4608, 128, True
    args.server_arg, args.url = [], f"http://127.0.0.1:{args.port}"
    configure_helpers(args.worker_pythonpath)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "plan_only"}
    identity = {"config": config, "schedule": schedule(args.seed), "workloads": [dataclasses.asdict(SHAPE)],
        "arm_definitions": ARMS, "worker_extension": EXTENSION, "source_sha256": source_hashes(),
        "controlled_server_environment": focus.CONTROLLED_SERVER_ENVIRONMENT,
        "corpus_sha256": hashlib.sha256(args.text_file.read_bytes()).hexdigest(),
        "selection_source_sha256": hashlib.sha256(args.selection_source.read_bytes()).hexdigest() if args.selection_source else None}
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise RuntimeError("A different SMS experiment already occupies this output; use a new root")
    else:
        common.atomic_json(path, {"identity": identity,
            "metadata": {"plan_only": True} if args.plan_only else common.metadata(),
            "protocol": "Four-arm Williams-balanced fresh launches; same extension/1GiB buffer in every arm; full fixed warmups; all rounds retained; independent factorial launch contrasts, selection never pooled."})
    args.frozen_source_sha256 = identity["source_sha256"]
    print(f"[plan]16fresh launches,4Williams-balanced sweeps; schedule={identity['schedule']}; output={args.output}", flush=True)
    if args.plan_only:
        return
    prompt_path = args.output / "prompts.json"
    if prompt_path.exists():
        prompts = json.loads(prompt_path.read_text())[SHAPE.name]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
        value = common.make_prompts(tokenizer, [SHAPE], args.seed, args.text_file.read_text())
        common.atomic_json(prompt_path, value)
        prompts = value[SHAPE.name]
        common.atomic_json(args.output / "prompts_metadata.json", {"tokenizer_name_or_path": tokenizer.name_or_path,
            "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "prompts_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()})
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal{signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for sweep in identity["schedule"]:
            for arm in sweep["arms"]:
                launch(args, arm, sweep["repeat"], prompts)
                write_analysis(args.output)
    finally:
        report = write_analysis(args.output)
        print(f"[analysis]complete={report['complete']}; {report['counts']}", flush=True)


if __name__ == "__main__":
    main()
