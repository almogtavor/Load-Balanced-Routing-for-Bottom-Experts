#!/usr/bin/env python3
"""Focused placement-router selection, controls, confirmation, and held-out serving.

The original serve_profile.py is unchanged. This script reuses its HTTP, process,
prompt, and trace helpers with explicit per-process extension points. All warmups
run the complete measured workload. Every measured round is retained, including
slow rounds and rounds with observed JIT events. Profiles are separate launches.

Suggested staged use (MODEL and CORPUS are local cached paths in the GPU pod):
  python experiments/serve_focus.py --model MODEL --text-file CORPUS --output OUT \
    --phase screen --preset prefill-concurrency --selector warp1
  python experiments/serve_focus.py --model MODEL --text-file CORPUS --output OUT \
    --phase controls --preset prefill-concurrency --cell ht_b16384_c64 --selector warp1
  python experiments/serve_focus.py --model MODEL --text-file CORPUS --output OUT \
    --phase confirm --preset prefill-concurrency --cell ht_b16384_c64 --selector warp1 \
    --selection-source OUT/screen/ht_b16384_c64/serving_analysis.json
  python experiments/serve_focus.py --model MODEL --text-file CORPUS --output OUT \
    --phase validate --preset prefill-concurrency --cell ht_b16384_c64 --selector warp1
  python experiments/serve_focus.py --model MODEL --text-file CORPUS --output OUT \
    --phase profile --preset prefill-concurrency --cell ht_b16384_c64 --selector warp1

Screen is four exploratory fresh launches: stock/tuned in two matched cells.
Controls are four launches: stock, tuned no-op keep=8, original balanced, tuned
balanced. Confirmation is four fresh stock/tuned pairs (eight launches), with
each arm occupying both positions equally. Validation independently repeats the
selected cell with 32 output tokens instead of the one-token pure-prefill screen.
Selection, confirmation, held-out validation, and profiles never pool results.

Other presets vary exactly one knob within their stock/tuned matched cells:
prefill-budget changes 8192 vs 16384 batch tokens; decode-backend changes DeepEP
HT vs LL. Backend gains for stock are deployment gains, not placement gains.
LL uses a small 1024-token budget, identical to its HT comparison, and explicit
linear expert placement preserves this router's contiguous id-to-rank mapping.
LL availability/compatibility must be checked in the actual image; failures stay
visible and never silently fall back to another backend. --plan-only needs no GPU
or tokenizer download. Runtime budgets are estimates, not result-based stopping.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import time
from types import SimpleNamespace
import traceback

import analyze_serving as analysis_tools
import serve_profile as common

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COMMAND = common.server_command
ORIGINAL_ENVIRONMENT = common.arm_environment
ORIGINAL_ACTIVE_CHECK = common.check_active
ARM_DEFINITIONS = {}
CONTROLLED_SERVER_ENVIRONMENT = {}


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    backend: str
    batch_tokens: int
    shape: common.Shape


def cells_for(preset, phase, decode_budget=1024):
    generation = 32 if phase == "validate" else 1
    if preset == "prefill-concurrency":
        return [Cell(f"ht_b16384_c{concurrency}", "deepep_high_throughput", 16384,
                     common.Shape("prefill", 4096, generation, concurrency, 128))
                for concurrency in (32, 64)]
    if preset == "prefill-budget":
        return [Cell(f"ht_b{budget}_c64", "deepep_high_throughput", budget,
                     common.Shape("prefill", 4096, generation, 64, 128))
                for budget in (8192, 16384)]
    if preset == "decode-backend":
        return [Cell(f"{short}_b{decode_budget}_c32", backend, decode_budget,
                     common.Shape("decode", 256, 128, 32, 64))
                for short, backend in (("ht", "deepep_high_throughput"),
                                       ("ll", "deepep_low_latency"))]
    raise ValueError(preset)


def counterbalanced_schedule(arms, sweeps, seed, position_offset=0):
    order = list(arms)
    random.Random(seed).shuffle(order)
    return [{"repeat": repeat, "arms": order[(repeat + position_offset) % len(order):]
             + order[:(repeat + position_offset) % len(order)]}
            for repeat in range(sweeps)]


def configure_helpers(selector, nvshmem_qp_depth=4096):
    global ARM_DEFINITIONS, CONTROLLED_SERVER_ENVIRONMENT
    CONTROLLED_SERVER_ENVIRONMENT = {"NVSHMEM_QP_DEPTH": str(nvshmem_qp_depth)}
    ARM_DEFINITIONS = {
        "stock": {"keep": None, "penalty": None, "selector": None},
        "noop8": {"keep": 8, "penalty": 0, "selector": selector},
        "balanced6p1": {"keep": 6, "penalty": 1, "selector": "baseline"},
        "tuned6p1": {"keep": 6, "penalty": 1, "selector": selector},
    }
    common.ARMS = {name: (arm["keep"], arm["penalty"]) for name, arm in ARM_DEFINITIONS.items()}
    # These bindings affect this fresh harness process only, never another runner.
    common.server_command = server_command
    common.arm_environment = arm_environment
    common.check_active = check_active


def server_command(args, trace_dir):
    command = ORIGINAL_COMMAND(args, trace_dir)
    command[command.index("--all2all-backend") + 1] = args.backend
    command += ["--expert-placement-strategy", "linear", "--enable-chunked-prefill"]
    if args.jit_verbose:
        command.append("--jit-monitor-verbose")
    return command


def arm_environment(arm):
    env = ORIGINAL_ENVIRONMENT(arm)
    for name in ("PLACEMENT_EXPERIMENTAL_SELECTOR", "PLACEMENT_NUM_WARPS"):
        env.pop(name, None)
    selector = ARM_DEFINITIONS[arm]["selector"]
    if selector is not None:
        env["PLACEMENT_EXPERIMENTAL_SELECTOR"] = selector
    env.update(CONTROLLED_SERVER_ENVIRONMENT)
    return env


def check_active(arm, log, dp):
    ORIGINAL_ACTIVE_CHECK(arm, log, dp)
    text = log.read_text(errors="replace")
    expected = ARM_DEFINITIONS[arm]
    if arm == "stock" and "placement-aware routing active" in text:
        raise RuntimeError("Stock unexpectedly activated placement routing")
    observed = set(re.findall(r"experimental selector=([A-Za-z0-9_]+)", text))
    if expected["selector"] not in (None, "baseline"):
        if observed != {expected["selector"]}:
            raise RuntimeError(f"Expected experimental selector={expected['selector']}; observed {sorted(observed)}")
    elif observed - {"baseline"}:
        raise RuntimeError(f"Unexpected experimental selector in {arm}: {sorted(observed)}")


def source_hashes():
    paths = [Path(__file__), ROOT / "experiments/serve_profile.py",
             ROOT / "src/placement_router/kernel.py", ROOT / "src/placement_router/plugin.py",
             ROOT / "src/placement_router/experimental.py",
             ROOT / "src/placement_router/vendor/fused_topk_router.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def log_events(log, start, finish):
    with log.open("rb") as source:
        source.seek(start)
        text = source.read(max(finish - start, 0)).decode(errors="replace")
    return [line for line in text.splitlines()
            if "jit_monitor.py" in line and ("compilation during inference" in line
                                             or "autotuning during inference" in line)]


def workload(args, shape, prompts, directory):
    log = directory / "server.log"
    start = log.stat().st_size
    result = common.run_workload(args, shape, prompts)
    finish = log.stat().st_size
    result["server_log_byte_offset_start"] = start
    result["server_log_byte_offset_end"] = finish
    result["jit_events_observed_in_log_slice"] = log_events(log, start, finish)
    return result


def annotate_server(directory, arm, phase):
    path = directory / "server.json"
    data = json.loads(path.read_text())
    data["placement_environment"]["PLACEMENT_EXPERIMENTAL_SELECTOR"] = ARM_DEFINITIONS[arm]["selector"]
    data["arm_definition"] = ARM_DEFINITIONS[arm]
    data["controlled_server_environment"] = CONTROLLED_SERVER_ENVIRONMENT
    data["experiment_phase"] = phase
    data["profiler_active_during_measurements"] = phase == "profile"
    common.atomic_json(path, data)


def runtime_evidence(log):
    text = log.read_text(errors="replace")
    return {"observed_cudagraph_modes": sorted(set(re.findall(
                r"['\"]cudagraph_mode['\"]:\s*<CUDAGraphMode\.([A-Z_]+):", text))),
            "cuda_graph_capture_log_lines": [line for line in text.splitlines()
                if re.search(r"Captur.*(?:CUDA graph|cudagraph)|CUDA graph.*captur|Skipping encoder and decoder CUDA graph capture", line, re.I)],
            "expert_placement_requested": "linear",
            "placement_note": "Linear logical expert blocks are required. DeepEP LL uses no global-to-physical table for linear placement; round-robin is prohibited by this harness."}


def measure_launch(args, arm, repeat, shape, prompts):
    directory = args.output / f"r{repeat:02d}_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        print(f"[resume] {directory}", flush=True)
        return
    if path.exists():
        path.rename(directory / f"result.previous.{time.time_ns()}.json")
    data = {"arm": arm, "repeat": repeat, "complete": False,
            "measurement_type": "uninstrumented_streaming_http", "experiment_phase": args.phase,
            "arm_definition": ARM_DEFINITIONS[arm], "shapes": {shape.name: {
                "configuration": dataclasses.asdict(shape), "warmups": [], "rounds": []}}}
    item = data["shapes"][shape.name]
    common.atomic_json(path, data)
    try:
        with common.serve(args, arm, directory, None):
            annotate_server(directory, arm, args.phase)
            for phase, count in (("warmups", args.warmup_rounds), ("rounds", args.rounds)):
                for index in range(count):
                    row = workload(args, shape, prompts, directory)
                    item[phase].append(row)
                    print(f"[{phase}] {directory.name} {index + 1}/{count}: "
                          f"{row['output_tokens_per_second']:.3f} out-tok/s; "
                          f"TTFT p50={row['ttft_ms']['p50']:.2f}ms; "
                          f"observed JIT events={len(row['jit_events_observed_in_log_slice'])}", flush=True)
                    common.atomic_json(path, data)
            check_active(arm, directory / "server.log", args.dp)
        data["jit_clean_measurements_observed"] = all(not row["jit_events_observed_in_log_slice"] for row in item["rounds"])
        data["runtime_evidence"] = runtime_evidence(directory / "server.log")
        data["complete"] = True
    except BaseException as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, data)


def profile_launch(args, arm, shape, prompts):
    directory = args.output / f"profile_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        print(f"[resume] {directory}", flush=True)
        return
    if path.exists():
        path.rename(directory / f"result.previous.{time.time_ns()}.json")
    trace_dir = directory / f"traces_{time.time_ns()}"
    trace_dir.mkdir()
    data = {"arm": arm, "complete": False, "measurement_type": "profile_only",
            "experiment_phase": "profile", "trace_directory": str(trace_dir), "shapes": {}, "warmups": []}
    common.atomic_json(path, data)
    try:
        with common.serve(args, arm, directory, trace_dir):
            annotate_server(directory, arm, "profile")
            for _ in range(args.warmup_rounds):
                data["warmups"].append(workload(args, shape, prompts, directory))
                common.atomic_json(path, data)
            common.post_control(args, "/start_profile")
            try:
                data["shapes"][shape.name] = workload(args, shape, prompts, directory)
            finally:
                common.post_control(args, "/stop_profile")
            check_active(arm, directory / "server.log", args.dp)
        data["trace_summary"] = common.summarize_traces(trace_dir, args.dp)
        data["runtime_evidence"] = runtime_evidence(directory / "server.log")
        if not data["trace_summary"]["valid_all_dp_ranks_cpu_and_cuda"]:
            raise RuntimeError("Profile lacks CPU+CUDA events from every DP rank")
        data["complete"] = True
    except BaseException as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, data)


def write_analysis(args):
    report = analysis_tools.analyze(args.output)
    report["method"]["supplemental_contrasts"] = "tuned6p1/noop8 uses the same experimental selector to separate routing changes from selector overhead; tuned6p1/balanced6p1 changes implementation while preserving keep=6 and load_penalty=1. Stock remains the deployment baseline."
    expected = {(row["repeat"], row["arm"]) for row in report["plan"]["expected_launches"]}
    indexed = {(row["repeat"], row["arm"]): row for row in report["launches"]}
    for baseline in ("noop8", "balanced6p1"):
        if baseline in report["plan"]["arms"] and "tuned6p1" in report["plan"]["arms"]:
            report["supplemental_comparisons"].extend(analysis_tools.supplemental_contrast(
                indexed, expected, report["plan"], "tuned6p1", baseline))
    analysis_tools.write_reports(report, args.output, SimpleNamespace(
        output=None, markdown=None, csv=None, comparisons_csv=None))
    return report


def run_cell(args, cell, arms, sweeps, position_offset=0):
    cell_args = SimpleNamespace(**vars(args))
    cell_args.output = args.output / args.phase / cell.name
    cell_args.backend, cell_args.batch_tokens = cell.backend, cell.batch_tokens
    cell_args.arms, cell_args.shapes = arms, [cell.shape.name]
    cell_args.repeats = sweeps
    cell_args.url = f"http://127.0.0.1:{args.port}"
    cell_args.output.mkdir(parents=True, exist_ok=True)
    schedule = counterbalanced_schedule(arms, sweeps, args.seed, position_offset)
    cell_args.arm_order_position_offset = position_offset
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(cell_args).items()
              if key not in ("plan_only",)}
    identity = {"config": config, "schedule": schedule, "workloads": [dataclasses.asdict(cell.shape)],
                "arm_definitions": {arm: ARM_DEFINITIONS[arm] for arm in arms}, "source_sha256": source_hashes(),
                "controlled_server_environment": CONTROLLED_SERVER_ENVIRONMENT,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "corpus_sha256": hashlib.sha256(args.text_file.read_bytes()).hexdigest() if args.text_file else None,
                "selection_source_sha256": hashlib.sha256(args.selection_source.read_bytes()).hexdigest() if args.selection_source else None}
    manifest = cell_args.output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise RuntimeError(f"Different experiment identity at {manifest}; choose a fresh output")
    else:
        common.atomic_json(manifest, {"identity": identity, "metadata": common.metadata(),
            "protocol": "Full measured-size predetermined warmups; all measured rounds retained. Fixed natural-text IDs across arms. Fresh launches paired within sweep; Latin-position rotation; no adaptive within-run warmup/filtering. Screen/control data exploratory; confirm and validate directories independent. JIT verbose logging is identical across arms. Profiles separate."})
    print(f"[plan] {cell.name}: {len(arms) * sweeps} fresh launches; schedule={schedule}; "
          f"shape={dataclasses.asdict(cell.shape)}; output={cell_args.output}", flush=True)
    if args.plan_only:
        return
    prompt_path = cell_args.output / "prompts.json"
    if prompt_path.exists():
        prompts = json.loads(prompt_path.read_text())[cell.shape.name]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  trust_remote_code=args.trust_remote_code)
        source = args.text_file.read_text() if args.text_file else None
        all_prompts = common.make_prompts(tokenizer, [cell.shape], args.seed, source)
        common.atomic_json(prompt_path, all_prompts)
        prompts = all_prompts[cell.shape.name]
        common.atomic_json(cell_args.output / "prompts_metadata.json", {
            "tokenizer_name_or_path": tokenizer.name_or_path,
            "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": type(tokenizer).__name__,
            "prompts_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()})
    if args.phase == "profile":
        for arm in arms:
            profile_launch(cell_args, arm, cell.shape, prompts)
    else:
        try:
            for sweep in schedule:
                for arm in sweep["arms"]:
                    measure_launch(cell_args, arm, sweep["repeat"], cell.shape, prompts)
        finally:
            report = write_analysis(cell_args)
            print(f"[analysis] {cell.name}: {report['completion']}", flush=True)


def write_matrix_summary(args, cells):
    if args.phase == "profile":
        return
    records, raw = [], {}
    for cell in cells:
        path = args.output / args.phase / cell.name / "serving_analysis.json"
        if not path.exists():
            records.append({"cell": cell.name, "complete": False, "report_file": str(path)})
            continue
        report = json.loads(path.read_text())
        raw[cell.name] = report
        primary = [item for item in report["comparisons"] if item["arm"] == "tuned6p1"
                   and item["metric"] == "output_tokens_per_second"][0]
        records.append({"cell": cell.name, "complete": report["completion"]["complete"],
            "report_file": str(path), "backend": cell.backend, "batch_tokens": cell.batch_tokens,
            "workload": dataclasses.asdict(cell.shape), "placement_specific_tuned_vs_stock": primary,
            "arm_absolute_summaries": report["arm_summaries"],
            "runtime_evidence": [{"arm": launch["arm"], "repeat": launch["repeat"],
                "jit_clean_measurements_observed": launch.get("jit_clean_measurements_observed"),
                **launch.get("runtime_evidence", {})} for launch in report["launches"]]})
    cross_cell = []
    if len(cells) == 2 and all(cell.name in raw for cell in cells):
        a, b = (raw[cell.name] for cell in cells)
        for arm in ("stock", "tuned6p1"):
            before = {launch["repeat"]: launch for launch in a["launches"]
                      if launch["arm"] == arm and launch.get("eligible_for_paired_analysis")}
            after = {launch["repeat"]: launch for launch in b["launches"]
                     if launch["arm"] == arm and launch.get("eligible_for_paired_analysis")}
            for metric in ("output_tokens_per_second", "ttft_ms_p50"):
                pairs = [{"repeat": repeat,
                    "baseline_value": before[repeat]["shapes"][cells[0].shape.name]["median_round_metrics"][metric],
                    "arm_value": after[repeat]["shapes"][cells[1].shape.name]["median_round_metrics"][metric]}
                    for repeat in sorted(before.keys() & after.keys())]
                cross_cell.append({"arm": arm, "metric": metric, "from_cell": cells[0].name,
                    "to_cell": cells[1].name, "pairs": pairs,
                    "interpretation": "A backend/scheduler/concurrency configuration effect for this arm; not the placement-specific effect. Screening data are exploratory and sequential cell order may confound this contrast.",
                    **analysis_tools.paired_statistics(pairs)})
    data = {"phase": args.phase, "preset": args.preset, "selector": args.selector,
            "complete": bool(records) and all(record["complete"] for record in records),
            "cells": records, "cross_cell_configuration_effects": cross_cell,
            "note": "Placement gains compare tuned vs stock INSIDE the same cell. Stock cross-cell gains measure backend/scheduler tuning. Screen/control selection data are not pooled with independent confirmation/held-out validation."}
    directory = args.output / args.phase
    common.atomic_json(directory / "matrix_summary.json", data)
    lines = [f"Focused serving: {args.phase}", "", data["note"], "",
             "| Cell | Status | Independent pairs | Tuned / matched stock throughput | 95% interval |",
             "|---|---|---:|---:|---:|"]
    for record in records:
        item = record.get("placement_specific_tuned_vs_stock")
        if not item:
            lines.append(f"| {record['cell']} | INCOMPLETE | 0 | — | — |")
            continue
        ci = item["geometric_change_pct_ci95"]
        interval = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "unavailable (n < 2)"
        effect = item["geometric_mean_change_pct"]
        lines.append(f"| {record['cell']} | {'complete' if record['complete'] else 'INCOMPLETE'} | "
                     f"{item['n_paired_launches']} | " + (f"{effect:+.2f}%" if effect is not None else "—") + f" | {interval} |")
    if cross_cell:
        lines += ["", "Configuration changes across cells (exploratory; distinguish these from placement changes):", "",
                  "| Arm | From / to cell | Metric | Change | 95% interval |", "|---|---|---|---:|---:|"]
        for item in cross_cell:
            ci = item["geometric_change_pct_ci95"]
            interval = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "unavailable (n < 2)"
            effect = item["geometric_mean_change_pct"]
            lines.append(f"| {item['arm']} | {item['from_cell']} / {item['to_cell']} | {item['metric']} | "
                         + (f"{effect:+.2f}%" if effect is not None else "—") + f" | {interval} |")
    (directory / "matrix_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("screen", "controls", "confirm", "validate", "profile"), required=True)
    parser.add_argument("--preset", choices=("prefill-concurrency", "prefill-budget", "decode-backend"), default="prefill-concurrency")
    parser.add_argument("--cell", help="Required for controls/confirm/validate/profile; choose the selected cell explicitly")
    parser.add_argument("--selector", default="warp1", help="Opt-in experimental selector: baseline, warp1, rows4, etc.")
    parser.add_argument("--selection-source", type=Path, help="Prior screen report; provenance only, never pooled into confirmation")
    parser.add_argument("--selection-note", default="", help="Reason for choosing the cell/candidate before confirmation")
    parser.add_argument("--sweeps", type=int, help="Defaults: 1 for screen/controls/profile; 4 for confirm/validate")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--gpu-util", type=float, default=.88)
    parser.add_argument("--decode-batch-tokens", type=int, choices=(512, 1024), default=1024,
                        help="Identical HT/LL token budget in the decode-backend screen")
    parser.add_argument("--nvshmem-qp-depth", type=int, default=4096,
                        help="Set NVSHMEM_QP_DEPTH identically for every arm/backend; LL requires at least 2*(batch_tokens+1)")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--boot-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--profile-timeout", type=int, default=600)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--jit-verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="Identical verbose JIT monitor across arms; retain every measured event/round")
    parser.add_argument("--estimated-boot-seconds", type=float, default=60)
    parser.add_argument("--estimated-round-seconds", type=float, default=12)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.dp, args.rounds, args.warmup_rounds) < 1 or args.sweeps is not None and args.sweeps < 1:
        parser.error("dp/rounds/warmup-rounds/sweeps must be positive")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", args.selector):
        parser.error("selector must be a simple candidate name")
    for name in ("text_file", "selection_source"):
        value = getattr(args, name)
        if value:
            value = value.resolve()
            if not value.is_file():
                parser.error(f"{name} does not exist: {value}")
            setattr(args, name, value)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_lock = (args.output / ".focus-runner.lock").open("a")
    try:
        fcntl.flock(experiment_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another focused runner is already using this output") from None
    cells = cells_for(args.preset, args.phase, args.decode_batch_tokens)
    if args.nvshmem_qp_depth < 1 or (args.preset == "decode-backend"
                                    and args.nvshmem_qp_depth < 2 * (args.decode_batch_tokens + 1)):
        parser.error("NVSHMEM_QP_DEPTH must be positive and at least 2*(decode batch tokens+1) for LL")
    if args.phase != "screen" and not args.cell:
        parser.error("Choose --cell explicitly after reviewing the exploratory screen")
    if args.cell:
        cells = [cell for cell in cells if cell.name == args.cell]
        if not cells:
            parser.error("Unknown --cell for this preset")
    if args.phase == "validate" and args.preset == "decode-backend":
        parser.error("Held-out 4096/32 validation is for the prefill presets")
    sweeps = args.sweeps or (4 if args.phase in ("confirm", "validate") else 1)
    arms = ["stock", "noop8", "balanced6p1", "tuned6p1"] if args.phase == "controls" else ["stock", "tuned6p1"]
    if args.phase in ("confirm", "validate") and sweeps % len(arms):
        parser.error("Confirmation/validation sweeps must fully balance arm positions (use an even number)")
    if args.phase == "profile" and sweeps != 1:
        parser.error("Profiles are a separate diagnostic pair, not independent serving repeats")
    if any(cell.shape.prompt_tokens + cell.shape.output_tokens > args.max_model_len for cell in cells):
        parser.error("max-model-len is smaller than the workload")
    locked = ("--all2all-backend", "--data-parallel-size", "--tensor-parallel-size", "--enable-expert-parallel",
              "--max-num-batched-tokens", "--max-num-seqs", "--gpu-memory-utilization", "--jit-monitor-verbose")
    locked += ("--expert-placement-strategy", "--no-enable-chunked-prefill")
    if any(extra.split("=", 1)[0] in locked for extra in args.server_arg):
        parser.error("Use the explicit matrix/config flags rather than overriding controlled server arguments")
    configure_helpers(args.selector, args.nvshmem_qp_depth)
    launches = len(cells) * len(arms) * sweeps
    measured_rounds = 1 if args.phase == "profile" else args.rounds
    estimate = launches * (args.estimated_boot_seconds + (args.warmup_rounds + measured_rounds) * args.estimated_round_seconds)
    print(f"[budget] {launches} launches; approximate {estimate / 60:.1f} minutes using "
          f"{args.estimated_boot_seconds:g}s boot and {args.estimated_round_seconds:g}s per workload round. "
          "Compilation, failure/retry and profile export can take longer; no speed-based stopping.", flush=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for index, cell in enumerate(cells):
            run_cell(args, cell, arms, sweeps, index if args.phase == "screen" else 0)
    finally:
        if not args.plan_only:
            write_matrix_summary(args, cells)


if __name__ == "__main__":
    main()
