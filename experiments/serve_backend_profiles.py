#!/usr/bin/env python3
"""Separate stock DeepEP HT/LL diagnostic profiles after the timing campaign.

  python experiments/serve_backend_profiles.py --model MODEL --text-file CORPUS \
      --output CAMPAIGN_ROOT

Creates CAMPAIGN_ROOT/backend-profiles with two fresh stock server launches.
Both use DP4/TP1/EP, batch1024, max_seqs64, linear placement, QP_DEPTH4096,
and OMP1. HT requests NONE; LL requests FULL_DECODE_ONLY and the verified
[1,2,4,8,16,32,64] capture buckets. Each launch executes three complete
256-input/128-output c4/n16 warmups before profiling only 256/16 c4/n16.
All traces, commands, raw HTTP records and graph evidence are retained.
Profiler latencies are diagnostic and never pooled with clean timing results.
--plan-only does not load a tokenizer, launch a server or query GPU hardware.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
from types import SimpleNamespace
import traceback

import serve_backend_confirm as backend
import serve_focus as focus
import serve_profile as common

ROOT = Path(__file__).resolve().parents[1]
WARMUP = common.Shape("decode_c4", 256, 128, 4, 16)
PROFILE = common.Shape("profile_decode_c4", 256, 16, 4, 16)
WARMUP_ROUNDS = 3
GRAPH_LAUNCH = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_|$)")
LOCKED_ARGS = {"--all2all-backend", "--data-parallel-size", "--tensor-parallel-size",
    "--enable-expert-parallel", "--max-num-batched-tokens", "--max-num-seqs",
    "--expert-placement-strategy", "--profiler-config", "--compilation-config",
    "--enforce-eager", "--no-enable-chunked-prefill", "--enable-chunked-prefill",
    "--enable-prefix-caching", "--no-enable-prefix-caching", "--host", "--port",
    "--seed", "--max-model-len", "--gpu-memory-utilization", "--jit-monitor-verbose"}


def configure_helpers():
    # Changes only bindings in this new runner process, never frozen source.
    backend.configure_helpers()
    common.server_command = server_command
    common.arm_environment = stock_environment


def stock_environment(arm):
    if arm not in backend.ARMS:
        raise ValueError(f"Not a stock backend arm: {arm}")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("PLACEMENT_") or key.startswith("VLLM_TORCH_PROFILER_"):
            del env[key]
    env.update(NVSHMEM_QP_DEPTH="4096", OMP_NUM_THREADS="1")
    return env


def server_command(args, trace_dir):
    if trace_dir is None:
        raise RuntimeError("Backend diagnostic launch requires its own trace directory")
    # Unlike backend.server_command, this wrapper intentionally permits a
    # profiler. focus delegates to the original verified common command.
    command = focus.server_command(args, trace_dir)
    command.append("--compilation-config=" + json.dumps(
        backend.COMPILATION_CONFIGS[args.backend], separators=(",", ":")))
    return command


def source_hashes():
    names = ("experiments/serve_backend_profiles.py", "experiments/serve_backend_confirm.py",
             "experiments/serve_focus.py", "experiments/serve_profile.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def summarize_traces(directory, expected_dp=4):
    """Count actual replay APIs as well as CPU+CUDA activity, one trace at a time."""
    rows = []
    for path in sorted(directory.rglob("*.pt.trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            trace = json.load(stream)
        rank_match = re.search(r"(?:^|_)dp(\d+)(?:_|\.)", path.name)
        kernels, api_names, cpu_events, placement_kernels = {}, {}, 0, set()
        for event in trace.get("traceEvents", []):
            if event.get("ph") != "X":
                continue
            name, category = str(event.get("name", "unknown")), str(event.get("cat", ""))
            if category in ("cpu_op", "user_annotation", "python_function"):
                cpu_events += 1
            if "kernel" in category.lower():
                value = kernels.setdefault(name, {"calls": 0, "summed_duration_us": 0.0})
                value["calls"] += 1
                value["summed_duration_us"] += event.get("dur", 0)
                if re.search(r"_par_select|placement", name, re.I):
                    placement_kernels.add(name)
            if category in ("cuda_runtime", "cuda_driver") and GRAPH_LAUNCH.search(name):
                api_names[name] = api_names.get(name, 0) + 1
        rows.append({"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dp_rank": int(rank_match.group(1)) if rank_match else None,
            "cpu_events": cpu_events, "cuda_kernel_events": sum(v["calls"] for v in kernels.values()),
            "cuda_graph_launch_api_counts": api_names,
            "cuda_graph_launch_api_events": sum(api_names.values()),
            "placement_kernel_names": sorted(placement_kernels),
            "top_kernels_by_summed_duration": sorted(
                ({"name": name, **value} for name, value in kernels.items()),
                key=lambda value: value["summed_duration_us"], reverse=True)[:40]})
        del trace
    ranks = {row["dp_rank"] for row in rows if row["cpu_events"] and row["cuda_kernel_events"]
             and row["dp_rank"] is not None}
    launches = {rank: sum(row["cuda_graph_launch_api_events"] for row in rows if row["dp_rank"] == rank)
                for rank in range(expected_dp)}
    return {"expected_dp_ranks": expected_dp, "observed_cpu_and_cuda_dp_ranks": sorted(ranks),
        "valid_all_dp_ranks_cpu_and_cuda": ranks == set(range(expected_dp)),
        "cuda_graph_launch_api_events_by_dp_rank": launches,
        "no_placement_kernels": not any(row["placement_kernel_names"] for row in rows),
        "traces": rows,
        "limits": "CPU CUDA API events prove observed graph launches, not every step's graph coverage. "
                  "Versioned runtime/driver API names are retained; do not sum both API layers as independent replays. "
                  "Kernel durations overlap and are not critical-path or clean serving timings."}


def require_trace_evidence(summary, arm):
    if not summary["valid_all_dp_ranks_cpu_and_cuda"]:
        raise RuntimeError("Profile lacks CPU+CUDA activity for every DP rank")
    if not summary["no_placement_kernels"]:
        raise RuntimeError("Stock profile contains a placement selector kernel")
    counts = list(summary["cuda_graph_launch_api_events_by_dp_rank"].values())
    if arm == "stock_ll" and not all(count > 0 for count in counts):
        raise RuntimeError("LL profile lacks observed CUDA graph launch APIs on every DP rank")
    if arm == "stock" and any(counts):
        raise RuntimeError("HT/NONE profile unexpectedly launched a CUDA graph")


def launch(args, arm, prompts):
    directory = args.output / ("profile_stock_ht" if arm == "stock" else "profile_stock_ll")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists() and json.loads(path.read_text()).get("complete"):
        print(f"[resume] {directory.name}", flush=True)
        return
    if path.exists():
        path.rename(directory / f"result.previous.{time.time_ns()}.json")
    trace_dir = directory / f"traces_{time.time_ns()}"
    trace_dir.mkdir()
    server_args = SimpleNamespace(**vars(args), backend=backend.ARMS[arm])
    result = {"arm": arm, "complete": False, "measurement_type": "profile_only",
        "experiment_phase": "backend-profiles", "trace_directory": str(trace_dir),
        "backend": backend.ARMS[arm], "compilation_config": backend.COMPILATION_CONFIGS[backend.ARMS[arm]],
        "warmup_configuration": dataclasses.asdict(WARMUP), "warmups": [], "shapes": {},
        "limits": "Separate diagnostic launch after the clean timing campaign; do not pool profiled HTTP latency with clean results."}
    common.atomic_json(path, result)
    try:
        with common.serve(server_args, arm, directory, trace_dir):
            server_path = directory / "server.json"
            server = json.loads(server_path.read_text())
            server.update(experiment_phase="backend-profiles", profiler_active_during_measurements=True,
                profiler_active_during_warmups=False,
                controlled_server_environment={"NVSHMEM_QP_DEPTH": "4096", "OMP_NUM_THREADS": "1"},
                stock_routing=True, arm_definition=focus.ARM_DEFINITIONS[arm])
            common.atomic_json(server_path, server)
            backend.check_stock(arm, directory / "server.log", args.dp)
            for index in range(WARMUP_ROUNDS):
                result["warmups"].append(focus.workload(server_args, WARMUP, prompts, directory))
                common.atomic_json(path, result)
                print(f"[warmup] {directory.name} {index + 1}/{WARMUP_ROUNDS}", flush=True)
            result["runtime_evidence_after_warmups"] = backend.require_graph_evidence(
                arm, directory / "server.log", args.dp)
            common.post_control(server_args, "/start_profile")
            result["profiler_start_returned_unix_ns"] = time.time_ns()
            try:
                result["shapes"][PROFILE.name] = focus.workload(server_args, PROFILE, prompts, directory)
            finally:
                common.post_control(server_args, "/stop_profile")
                result["profiler_stop_returned_unix_ns"] = time.time_ns()
                common.atomic_json(path, result)
            backend.check_stock(arm, directory / "server.log", args.dp)
        result["runtime_evidence"] = backend.require_graph_evidence(arm, directory / "server.log", args.dp)
        result["trace_summary"] = summarize_traces(trace_dir, args.dp)
        require_trace_evidence(result["trace_summary"], arm)
        result["complete"] = True
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Campaign root; creates backend-profiles")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--revision")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--gpu-util", type=float, default=.88)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--boot-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--profile-timeout", type=int, default=600)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if any(extra.split("=", 1)[0] in LOCKED_ARGS for extra in args.server_arg):
        parser.error("Extra server arguments cannot override backend, graph, profiler or workload controls")
    if args.max_model_len < WARMUP.prompt_tokens + WARMUP.output_tokens:
        parser.error("max-model-len is smaller than the fixed warmup workload")
    if min(args.boot_timeout, args.request_timeout, args.profile_timeout) <= 0:
        parser.error("Timeouts must be positive")
    args.text_file = args.text_file.resolve()
    if not args.text_file.is_file():
        parser.error("--text-file does not exist")
    campaign_root = args.output.resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    lock = (campaign_root / ".focus-runner.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("The timing/focused campaign is still using this output root") from None
    args.output = campaign_root / "backend-profiles"
    args.output.mkdir(parents=True, exist_ok=True)
    args.dp, args.batch_tokens, args.max_num_seqs, args.jit_verbose = 4, 1024, 64, True
    args.url = f"http://127.0.0.1:{args.port}"
    configure_helpers()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "plan_only"}
    identity = {"configuration": config, "source_sha256": source_hashes(),
        "corpus_sha256": hashlib.sha256(args.text_file.read_bytes()).hexdigest(),
        "arm_definitions": focus.ARM_DEFINITIONS, "schedule": list(backend.ARMS),
        "warmups": {"rounds": WARMUP_ROUNDS, "shape": dataclasses.asdict(WARMUP)},
        "profile_shape": dataclasses.asdict(PROFILE)}
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["identity"] != identity:
        raise RuntimeError(f"Different diagnostic identity at {manifest}; choose a new campaign root")
    if not manifest.exists():
        common.atomic_json(manifest, {"identity": identity,
            "protocol": "Two fresh stock launches, full warmups, bounded separate CPU+CUDA profiles; all DP and actual graph replay evidence required."})
    print(f"[plan] 2 fresh stock launches; 3 full warmups + 1 bounded profile each; output={args.output}", flush=True)
    if args.plan_only:
        return
    common.atomic_json(args.output / "runtime_metadata.json", common.metadata())
    prompt_path = args.output / "prompts.json"
    if prompt_path.exists():
        prompts = json.loads(prompt_path.read_text())[WARMUP.name]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=args.trust_remote_code)
        shared = common.make_prompts(tokenizer, [WARMUP], args.seed, args.text_file.read_text())
        common.atomic_json(prompt_path, shared)
        prompts = shared[WARMUP.name]
        common.atomic_json(args.output / "prompts_metadata.json", {
            "tokenizer_name_or_path": tokenizer.name_or_path,
            "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": type(tokenizer).__name__,
            "prompts_file_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest()})
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    for arm in backend.ARMS:
        launch(args, arm, prompts)


if __name__ == "__main__":
    main()
