#!/usr/bin/env python3
"""Independent stock DeepEP HT-versus-LL confirmation, with no placement routing.

  python experiments/serve_backend_confirm.py --model MODEL --text-file CORPUS \
      --output OUT --selection-source OUT/screen/matrix_summary.json

Writes a NEW OUT/backend-confirm stage: four counterbalanced fresh HT/LL pairs
(eight server launches), each running both fixed natural-text decode shapes:
256 input / 128 output tokens, concurrency 4 / 32, requests 16 / 64 respectively.
Both use DP=4 TP=1 EP, batch=1024, max_seqs=64, linear expert placement, and
NVSHMEM_QP_DEPTH=4096. Every router/probe selector variable is cleared. Profiles
are off. Three full-size warmups and two measured rounds are predetermined;
all measured rounds are retained. Neither screen nor placement-confirmation
results are pooled with this experiment. LL explicitly requests FULL_DECODE_ONLY
with capture sizes [1,2,4,8,16,32,64], matching the passed QP4096 smoke. HT requests
NONE. LL must show the exact graph mode/sizes AND
completed graph capture from EVERY DP worker before measurements are accepted.

--plan-only performs no tokenizer/model/GPU work. Conservative planning at 60s
startup + 30s/20s per c4/c32 workload round gives ~41 minutes; LL should be faster
than this HT-based estimate, while cold compilation can add time. A different
--rounds count must be chosen before the run and becomes part of its manifest.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
from pathlib import Path
import re
import signal
import time
from types import SimpleNamespace
import traceback

import analyze_serving as analysis_tools
import serve_focus as focus
import serve_profile as common

ARMS = {"stock": "deepep_high_throughput", "stock_ll": "deepep_low_latency"}
CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 64]
COMPILATION_CONFIGS = {"deepep_high_throughput": {"cudagraph_mode": "NONE"},
    "deepep_low_latency": {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CAPTURE_SIZES}}
SHAPES = [common.Shape("decode_c4", 256, 128, 4, 16),
          common.Shape("decode_c32", 256, 128, 32, 64)]


def configure_helpers():
    focus.ARM_DEFINITIONS = {arm: {"keep": None, "penalty": None, "selector": None,
                                   "backend": backend, "compilation_config": COMPILATION_CONFIGS[backend]}
                             for arm, backend in ARMS.items()}
    focus.CONTROLLED_SERVER_ENVIRONMENT = {"NVSHMEM_QP_DEPTH": "4096"}
    common.ARMS = {arm: (None, None) for arm in ARMS}
    common.server_command = server_command
    common.arm_environment = focus.arm_environment
    common.check_active = check_stock


def server_command(args, trace_dir):
    if trace_dir is not None:
        raise RuntimeError("Backend confirmation requires profiling to be off")
    command = focus.server_command(args, None)
    command.append("--compilation-config=" + json.dumps(COMPILATION_CONFIGS[args.backend], separators=(",", ":")))
    return command


def check_stock(arm, log, dp):
    text = log.read_text(errors="replace")
    if "placement-aware routing active" in text or "experimental selector=" in text or "[moet]" in text:
        raise RuntimeError(f"A router selector/probe unexpectedly activated in stock backend arm {arm}")


def graph_evidence(log):
    evidence = focus.runtime_evidence(log)
    text = log.read_bytes().decode(errors="replace")
    finished = [line for line in text.split("\n")
                if "Graph capturing finished" in line and "Encoder graph" not in line]
    ranks = set()
    for line in finished:
        match = re.search(r"(?:Worker|EngineCore)_DP(\d+)", line)
        if match:
            ranks.add(int(match.group(1)))
    evidence.update(graph_capture_completed_log_lines=finished,
                    graph_capture_completed_dp_ranks=sorted(ranks))
    sizes = [json.loads(value) for value in re.findall(
        r"['\"]cudagraph_capture_sizes['\"]:\s*(\[[0-9, ]+\])", text)]
    evidence["observed_cudagraph_capture_sizes"] = [list(value) for value in sorted(set(tuple(value) for value in sizes))]
    return evidence


def require_graph_evidence(arm, log, dp):
    evidence = graph_evidence(log)
    modes = set(evidence["observed_cudagraph_modes"])
    if arm == "stock_ll":
        if (modes != {"FULL_DECODE_ONLY"} or CAPTURE_SIZES not in evidence["observed_cudagraph_capture_sizes"]
                or set(evidence["graph_capture_completed_dp_ranks"]) != set(range(dp))):
            raise RuntimeError("LL requires actual FULL_DECODE_ONLY mode, requested capture sizes and completed graph-capture logs "
                               f"for every DP worker; observed modes={sorted(modes)}, "
                               f"capture ranks={evidence['graph_capture_completed_dp_ranks']}; see {log}")
    elif modes != {"NONE"}:
        raise RuntimeError(f"Expected HT graph mode NONE in this vLLM image, observed {sorted(modes)}; see {log}")
    return evidence


def launch(args, arm, repeat, prompts):
    directory = args.output / f"r{repeat:02d}_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        print(f"[resume] {directory.name}", flush=True)
        return
    if path.exists():
        path.rename(directory / f"result.previous.{time.time_ns()}.json")
    server_args = SimpleNamespace(**vars(args))
    server_args.backend = ARMS[arm]
    result = {"arm": arm, "repeat": repeat, "complete": False,
        "measurement_type": "uninstrumented_streaming_http", "experiment_phase": "backend-confirm",
        "arm_definition": focus.ARM_DEFINITIONS[arm], "shapes": {}}
    common.atomic_json(path, result)
    try:
        with common.serve(server_args, arm, directory, None):
            focus.annotate_server(directory, arm, "backend-confirm")
            # Warm BOTH fixed shapes fully before measuring either, covering the
            # same decode/prefill configurations for both backend arms.
            for shape in SHAPES:
                item = {"configuration": dataclasses.asdict(shape), "warmups": [], "rounds": []}
                result["shapes"][shape.name] = item
                for index in range(args.warmup_rounds):
                    row = focus.workload(server_args, shape, prompts[shape.name], directory)
                    item["warmups"].append(row)
                    print(f"[warmup] {directory.name}/{shape.name} {index + 1}/{args.warmup_rounds}: "
                          f"{row['output_tokens_per_second']:.2f} out-tok/s", flush=True)
                    common.atomic_json(path, result)
            result["runtime_evidence_after_warmups"] = require_graph_evidence(arm, directory / "server.log", args.dp)
            for shape in SHAPES:
                for index in range(args.rounds):
                    row = focus.workload(server_args, shape, prompts[shape.name], directory)
                    result["shapes"][shape.name]["rounds"].append(row)
                    print(f"[measure] {directory.name}/{shape.name} {index + 1}/{args.rounds}: "
                          f"{row['output_tokens_per_second']:.2f} out-tok/s; "
                          f"TTFT p50={row['ttft_ms']['p50']:.2f}ms; "
                          f"JIT events={len(row['jit_events_observed_in_log_slice'])}", flush=True)
                    common.atomic_json(path, result)
            check_stock(arm, directory / "server.log", args.dp)
        result["runtime_evidence"] = require_graph_evidence(arm, directory / "server.log", args.dp)
        result["jit_clean_measurements_observed"] = all(not row["jit_events_observed_in_log_slice"]
            for item in result["shapes"].values() for row in item["rounds"])
        result["complete"] = True
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, result)


def write_analysis(args):
    report = analysis_tools.analyze(args.output)
    report["method"]["repeat_unit"] = "fresh stock HT and stock LL server launches paired within the same counterbalanced sweep"
    report["method"]["contrast_scope"] = "DeepEP backend + its compatible graph configuration/capture, with all placement routing disabled. This estimates a deployment gain, not a placement-specific gain."
    analysis_tools.write_reports(report, args.output, SimpleNamespace(
        output=None, markdown=None, csv=None, comparisons_csv=None))
    lines = ["Independent stock HT versus stock LL confirmation", "",
             "Both arms use stock routing. LL uses its compatible CUDA-graph configuration with completed capture verified; this is an overall serving configuration comparison, not a placement-router improvement.", "",
             f"Status: {'COMPLETE' if report['completion']['complete'] else 'INCOMPLETE'}; "
             f"{report['completion']['valid_complete_launches']}/{report['completion']['expected_launches']} fresh server launches; "
             f"{report['completion']['measured_rounds_in_valid_launches']} measured shape-rounds; "
             f"{report['completion']['measured_requests_in_valid_launches']} measured requests.", "",
             "Stock baseline (`stock`) = DeepEP HT/NONE; comparison (`stock_ll`) = DeepEP LL/FULL_DECODE_ONLY with capture sizes[1,2,4,8,16,32,64]. Both: DP4/TP1/EP, batch1024, linear expert placement, NVSHMEM_QP_DEPTH4096, profiler off. Each launch executes both fixed 256-input/128-output shapes, c4 with16 requests and c32 with64 requests.", "",
             "| Workload | Metric | Independent pairs | LL / HT change | 95% paired-launch t interval |",
             "|---|---|---:|---:|---:|"]
    for item in report["comparisons"]:
        if item["metric"] not in ("output_tokens_per_second", "ttft_ms_p50", "tpot_ms_p50", "itl_ms_p50"):
            continue
        ci, change = item["geometric_change_pct_ci95"], item["geometric_mean_change_pct"]
        interval = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "unavailable"
        lines.append(f"| {item['shape']} | {item['metric']} | {item['n_paired_launches']} | "
                     + (f"{change:+.2f}%" if change is not None else "—") + f" | {interval} |")
    lines += ["", "Positive throughput change is faster; negative latency change is faster. Each launch contributes the median of every measured round. Requests/rounds are not independent repeats.", "",
              "Actual graph evidence (LL must pass completed capture on all DP workers):", "",
              "| Sweep | Backend arm | Observed graph modes | Completed capture DP ranks | Measured JIT clean (observed) |",
              "|---:|---|---|---|---|"]
    for row in report["launches"]:
        evidence = row.get("runtime_evidence", {})
        lines.append(f"| {row['repeat']} | {row['arm']} | {evidence.get('observed_cudagraph_modes', [])} | "
                     f"{evidence.get('graph_capture_completed_dp_ranks', [])} | {row.get('jit_clean_measurements_observed', 'unknown')} |")
    lines += ["", "Selection/smoke data are stored separately and never pooled. This comparison combines backend and compatible graph configuration; it does not isolate their individual causal contributions. Capture logs verify that graphs were built, not that every measured step replayed a graph. Small n, shared-node interference, carryover, model-specific scope and multiple pointwise endpoints limit generalization. Raw rounds, percentile metrics, commands and graph-capture log evidence are retained."]
    (args.output / "backend_confirmation.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Campaign root; this creates a new backend-confirm subdirectory")
    parser.add_argument("--selection-source", type=Path, help="Prior smoke/screen artifact, recorded only as selection provenance")
    parser.add_argument("--selection-note", default="")
    parser.add_argument("--pairs", type=int, choices=(4, 6, 8), default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--gpu-util", type=float, default=.88)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--boot-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--estimated-boot-seconds", type=float, default=60)
    parser.add_argument("--estimated-shape-round-seconds", nargs=2, type=float, default=[30, 20], metavar=("C4", "C32"))
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.warmup_rounds) < 1:
        parser.error("rounds and warmup-rounds must be positive")
    for name in ("text_file", "selection_source"):
        value = getattr(args, name)
        if value:
            value = value.resolve()
            if not value.is_file():
                parser.error(f"{name} does not exist: {value}")
            setattr(args, name, value)
    locked = ("--all2all-backend", "--data-parallel-size", "--tensor-parallel-size", "--enable-expert-parallel",
              "--max-num-batched-tokens", "--max-num-seqs", "--expert-placement-strategy", "--profiler-config",
              "--no-enable-chunked-prefill", "--enforce-eager", "--compilation-config")
    if any(extra.split("=", 1)[0] in locked for extra in args.server_arg):
        parser.error("Server arguments may not override the matched backend, expert mapping, graph or profiling controls")
    if args.max_model_len < max(shape.prompt_tokens + shape.output_tokens for shape in SHAPES):
        parser.error("max-model-len is smaller than the fixed workload")
    campaign_root = args.output.resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    lock = (campaign_root / ".focus-runner.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another focused/backend runner is already using this campaign root") from None
    args.output = campaign_root / "backend-confirm"
    args.output.mkdir(parents=True, exist_ok=True)
    args.phase, args.backend, args.dp, args.batch_tokens = "backend-confirm", "per_arm", 4, 1024
    args.max_num_seqs, args.jit_verbose, args.profile_timeout = 64, True, 600
    args.arms, args.shapes, args.repeats = list(ARMS), [shape.name for shape in SHAPES], args.pairs
    args.url = f"http://127.0.0.1:{args.port}"
    configure_helpers()
    schedule = focus.counterbalanced_schedule(args.arms, args.pairs, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "plan_only"}
    identity = {"config": config, "schedule": schedule, "workloads": [dataclasses.asdict(shape) for shape in SHAPES],
        "arm_definitions": focus.ARM_DEFINITIONS, "controlled_server_environment": focus.CONTROLLED_SERVER_ENVIRONMENT,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {**focus.source_hashes(), "experiments/serve_backend_confirm.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "corpus_sha256": hashlib.sha256(args.text_file.read_bytes()).hexdigest(),
        "selection_source_sha256": hashlib.sha256(args.selection_source.read_bytes()).hexdigest() if args.selection_source else None}
    path = args.output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text())["identity"] != identity:
            raise RuntimeError(f"Different confirmation identity at {path}; choose a fresh output")
    else:
        common.atomic_json(path, {"identity": identity, "metadata": common.metadata(),
            "protocol": "Independent backend confirmation; stock HT vs stock LL; four counterbalanced fresh pairs; both fixed decode shapes fully warmed before measurement; no routing/profiling; actual graph-capture evidence required; every measured round retained; selection data never pooled."})
    estimate = len(ARMS) * args.pairs * (args.estimated_boot_seconds
        + (args.warmup_rounds + args.rounds) * sum(args.estimated_shape_round_seconds))
    print(f"[plan] {len(ARMS) * args.pairs} fresh launches; {args.pairs} counterbalanced pairs; "
          f"conservative runtime estimate={estimate / 60:.1f}min; schedule={schedule}; output={args.output}", flush=True)
    if args.plan_only:
        return
    prompt_path = args.output / "prompts.json"
    if prompt_path.exists():
        prompts = json.loads(prompt_path.read_text())
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
        prompts = common.make_prompts(tokenizer, SHAPES, args.seed, args.text_file.read_text())
        common.atomic_json(prompt_path, prompts)
        common.atomic_json(args.output / "prompts_metadata.json", {
            "tokenizer_name_or_path": tokenizer.name_or_path, "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": type(tokenizer).__name__, "prompts_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()})
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for sweep in schedule:
            for arm in sweep["arms"]:
                launch(args, arm, sweep["repeat"], prompts)
                write_analysis(args)
    finally:
        report = write_analysis(args)
        print(f"[analysis] {report['completion']}", flush=True)


if __name__ == "__main__":
    main()
