#!/usr/bin/env python3
"""Separate original-selector diagnostic profiles for the paper reproduction.

Run once with --dp 4 and once with --dp 8, after the CUDA-event timing campaign:
  PYTHONPATH=/work/router/src:/work/router/experiments /usr/bin/python3 \
    /work/router/experiments/paper_profile.py --mode run --dp 4 \
    --model MODEL --checkpoint-proof PROOF --output NEW_PROFILE_DIRECTORY

Two fresh servers: native stock, then pristine original balanced6p1. Each gets
three full 4096-input/1-output, n256/c128 warmups with profiling inactive, then
one bounded n128/c128 workload of the same token shape under the torch profiler.
No paper_probe.py or other PLACEMENT_PROBE hook is installed. The original
public selector, including its index-dtype conversion, is deliberately retained.
HT, graphs NONE, batch16384, maxseq128, BF16, linear placement, OMP1, QP4096.
Installed DeepEP/vLLM sources must retain the original SM20 default contract.

These instrumented traces diagnose execution and shapes. They do not estimate
clean latency, critical-path speedup, or the CUDA-event paper headline. Exact-T
op counts do not identify every layer without correlating the raw trace further.
--mode plan performs no model, GPU, tokenizer or server operations.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import dataclasses
import fcntl
import gzip
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request

import paper_reproduce as paper
import serve_profile as common

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("stock", "balanced6p1")
WARMUP = common.Shape("paper_prefill", 4096, 1, 128, 256)
PROFILE = common.Shape("paper_profile_prefill", 4096, 1, 128, 128)
GRAPH_API = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_|$)")
EXPERT_KERNEL = re.compile(r"fused_moe_kernel|batched.*(?:gemm|mm)|(?:grouped|expert).*gemm", re.I)
LIMITS = ("Separate profiler-only launches; never pool profiled HTTP latencies with clean CUDA-event or serving results. "
          "Summed kernel durations can overlap and are not critical-path time or a speedup bound. "
          "HTTP request shape does not guarantee each rank's hidden-T; raw trace shapes are retained. "
          "Exact-T CPU-op counts provide positive coverage, not unique layer/step identities. "
          "SM20 evidence is the unmodified installed source/default contract, not a live worker counter.")


def source_hashes():
    names = ("experiments/paper_profile.py", "experiments/paper_reproduce.py",
             "experiments/serve_profile.py", "experiments/analyze_serving.py", *paper.ORIGINAL_SOURCE)
    return {name: paper.digest(ROOT / name) for name in names}


def configuration(args):
    return {"canonical_model_id": paper.MODEL_ID, "model": args.model, "revision": args.revision,
        "dp": args.dp, "ep": args.dp, "tp": 1, "backend": "deepep_high_throughput",
        "graphs": "NONE", "dtype": "bfloat16", "expert_placement": "linear",
        "num_sms_source_default": 20, "batch_tokens": 16384, "max_num_seqs": 128,
        "max_model_len": 4608, "gpu_util": args.gpu_util, "port": args.port,
        "vllm": args.vllm, "warmup_rounds": 3, "warmup_shape": dataclasses.asdict(WARMUP),
        "profile_shape": dataclasses.asdict(PROFILE), "arms": list(ARMS),
        "selector": "pristine_original_public_source", "prompt_seed": 0,
        "cuda_event_probe_active": False, "profile_activities_required": ["CPU", "CUDA"],
        "profile_activities_configuration": "Installed vLLM defaults; validate actual CPU+CUDA traces",
        "boot_timeout": args.boot_timeout, "request_timeout": args.request_timeout,
        "profile_timeout": args.profile_timeout}


def profile_environment(arm):
    paper.require(arm in ARMS, "Unknown profile arm")
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(("PAPER_PROBE_", "PLACEMENT_", "VLLM_TORCH_PROFILER_"))
                or key in ("VLLM_DEEPEP_BUFFER_SIZE_MB", "VLLM_PLUGINS")):
            env.pop(key, None)
    if arm == "balanced6p1":
        env.update(PLACEMENT_KEEP="6", PLACEMENT_LOAD_PENALTY="1")
    env.update(OMP_NUM_THREADS="1", NVSHMEM_QP_DEPTH="4096", VLLM_LOGGING_LEVEL="INFO",
        VLLM_PLUGINS="placement_aware_router")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "experiments"),
        *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return env


def server_command(args, trace_dir):
    paper.require(trace_dir is not None, "A unique profiler trace directory is required")
    # Reuse the frozen exact paper engine configuration, without its probe context.
    command = paper.server_command(args)
    command += ["--profiler-config", json.dumps({"profiler": "torch",
        "torch_profiler_dir": str(trace_dir), "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": True, "torch_profiler_use_gzip": True}, separators=(",", ":"))]
    return command


def sm20_source_contract(all2all_source, legacy_source):
    """Fail closed if this version no longer has the reviewed SM20 defaults."""
    tree = ast.parse(all2all_source)
    klass = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == "DeepEPAll2AllManagerBase"), None)
    constructor = next((n for n in klass.body if isinstance(n, ast.FunctionDef)
                        and n.name == "__init__"), None) if klass else None
    assignments = [n for n in ast.walk(constructor) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                and t.value.id == "self" and t.attr == "num_sms" for t in n.targets)] if constructor else []
    values = [n.value.value for n in assignments if isinstance(n.value, ast.Constant)]
    legacy = ast.parse(legacy_source)
    buffer = next((n for n in legacy.body if isinstance(n, ast.ClassDef) and n.name == "Buffer"), None)
    defaults = []
    for node in buffer.body if buffer else []:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if (targets and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == "num_sms" for t in targets)):
            defaults.append(node.value.value)
    paper.require(len(assignments) == 1 and values == [20] and defaults == [20],
                  f"Unreviewed SM20 source contract: manager={values}, Buffer={defaults}")
    return {"manager_num_sms_default": values[0], "legacy_Buffer_num_sms_default": defaults[0],
        "scope": "Static installed source defaults, not a live worker attribute observation"}


def runtime_contract():
    path = Path(importlib.metadata.distribution("vllm").locate_file(
        "vllm/distributed/device_communicators/all2all.py"))
    spec = importlib.util.find_spec("deep_ep")
    paper.require(spec is not None and spec.origin is not None, "deep_ep package unavailable")
    base = Path(spec.origin).parent
    legacy = base / "buffers/legacy.py"
    init = Path(spec.origin).read_text()
    paper.require(re.search(r"from\s+\.buffers\.legacy\s+import\s+Buffer", init) is not None,
                  "DeepEP Buffer is not the reviewed legacy API")
    proof = sm20_source_contract(path.read_text(), legacy.read_text())
    proof["source_files"] = {str(p): {"sha256": paper.digest(p), "bytes": p.stat().st_size}
                             for p in (path, Path(spec.origin), legacy)}
    return proof


def ensure_manifest(args):
    prompts = paper.make_prompts(args)
    identity = {"configuration": configuration(args), "source_sha256": source_hashes(),
        "checkpoint": paper.checkpoint_proof(args), "runtime_source_contract": runtime_contract(),
        "prompt_file_sha256": paper.digest(args.output / "prompts.json"),
        "prompt_metadata_sha256": paper.digest(args.output / "prompts_metadata.json"),
        "production_source_base_commit": paper.BASE_COMMIT,
        "production_source_expected_sha256": paper.ORIGINAL_SOURCE}
    paper.require_original_source(identity)
    path = args.output / "manifest.json"
    if path.exists():
        paper.require(paper.read(path)["identity"] == identity, "Profile identity changed; use a fresh output")
    else:
        common.atomic_json(path, {"identity": identity, "limits": LIMITS})
    return identity, prompts


@contextlib.contextmanager
def serve(args, arm, directory, trace_dir, identity):
    paper.require(source_hashes() == identity["source_sha256"], "Frozen profile source changed")
    paper.require_original_source(identity)
    paper.require(paper.digest(args.output / "prompts.json") == identity["prompt_file_sha256"]
        and paper.digest(args.output / "prompts_metadata.json") == identity["prompt_metadata_sha256"],
        "Frozen profile prompts changed")
    env, command = profile_environment(arm), server_command(args, trace_dir)
    with socket.socket() as sock:
        paper.require(sock.connect_ex(("127.0.0.1", args.port)) != 0, "Server port already occupied")
    common.atomic_json(directory / "server.json", {"command": command, "shell_command": shlex.join(command),
        "cwd": str(ROOT), "arm": arm, "selector": "pristine_original_public_source",
        "source_sha256": identity["source_sha256"], "cuda_event_probe_active": False,
        "profiler_active_during_warmups": False, "profiler_active_during_profile_workload": True,
        "controlled_environment": {key: env.get(key) for key in ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY",
            "PLACEMENT_PROBE", "PLACEMENT_EXPERIMENTAL_SELECTOR", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH",
            "VLLM_PLUGINS", "PYTHONPATH")}, "paper_probe_environment": {}, "limits": LIMITS})
    with (directory / "server.log").open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        common.atomic_json(directory / "pid.json", {"pid": process.pid, "started_unix_ns": time.time_ns()})
        try:
            deadline = time.monotonic() + args.boot_timeout
            while time.monotonic() < deadline:
                paper.require(process.poll() is None, "Server exited before healthy")
                try:
                    with urllib.request.urlopen(args.url + "/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    time.sleep(2)
            else:
                raise TimeoutError("Server startup timed out")
            yield process
        finally:
            common.kill_server(process)


def check_log(args, arm, path):
    text = path.read_text(errors="replace")
    paper.require(not re.search(r"\[paper-probe\]|\[moet\]|experimental selector=", text),
                  "Unexpected timing probe or experimental selector in profile logs")
    return paper.check_log(args, "base" if arm == "stock" else arm, path)


def workload(args, shape, prompts, directory):
    path = directory / "server.log"
    start = path.stat().st_size
    data = common.run_workload(args, shape, prompts)
    end = path.stat().st_size
    with path.open("rb") as source:
        source.seek(start)
        text = source.read(end - start).decode(errors="replace")
    data.update(configuration=dataclasses.asdict(shape), server_log_byte_offset_start=start,
        server_log_byte_offset_end=end, jit_events_observed_in_log_slice=[line for line in text.splitlines()
            if "jit_monitor.py" in line and ("compilation during inference" in line or "autotuning during inference" in line)])
    return data


def summarize_traces(directory, expected_dp, arm):
    rows = []
    for path in sorted(directory.rglob("*.pt.trace.json*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as source:
            trace = json.load(source)
        rank = re.search(r"(?:^|_)dp(\d+)(?:_|\.)", path.name)
        counts, Ts, names = collections.Counter(), collections.Counter(), collections.Counter()
        summed = collections.Counter()
        for event in trace.get("traceEvents", []):
            if event.get("ph") != "X":
                continue
            name, cat = str(event.get("name", "")), str(event.get("cat", ""))
            if cat in ("cpu_op", "user_annotation", "python_function"):
                counts["cpu_events"] += 1
            if cat == "cpu_op" and name == "vllm::moe_forward":
                counts["moe_cpu_calls"] += 1
                dims = event.get("args", {}).get("Input Dims", [])
                if dims and isinstance(dims[0], list) and len(dims[0]) == 2 and dims[0][1] == 2048:
                    Ts[str(dims[0][0])] += 1
            if cat in ("cuda_runtime", "cuda_driver") and GRAPH_API.search(name):
                counts["cuda_graph_launch_api_events"] += 1
            if "kernel" in cat.lower():
                counts["cuda_kernel_events"] += 1
                names[name] += 1
                summed[name] += event.get("dur", 0.)
                if name == "_par_select":
                    counts["original_router_kernel_calls"] += 1
                if re.search(r"topkGating|fused_topk", name, re.I):
                    counts["stock_router_kernel_calls"] += 1
                if "_par_select" in name and name != "_par_select":
                    counts["unexpected_selector_kernel_calls"] += 1
                if EXPERT_KERNEL.search(name):
                    counts["expert_kernel_calls"] += 1
        rows.append({"path": str(path), "sha256": paper.digest(path), "bytes": path.stat().st_size,
            "dp_rank": int(rank.group(1)) if rank else None, "counts": dict(counts),
            "moe_hidden_T_counts": dict(Ts), "router_kernel_names": sorted(name for name in names
                if re.search(r"_par_select|topkGating|fused_topk", name, re.I)),
            "top_kernels_by_summed_duration": [{"name": name, "calls": names[name],
                "summed_duration_us": duration} for name, duration in summed.most_common(40)]})
        del trace
    by_rank = {}
    for rank in range(expected_dp):
        counts, Ts = collections.Counter(), collections.Counter()
        for row in rows:
            if row["dp_rank"] == rank:
                counts.update(row["counts"])
                Ts.update(row["moe_hidden_T_counts"])
        by_rank[str(rank)] = {"counts": dict(counts), "moe_hidden_T_counts": dict(Ts)}
    observed = sorted({row["dp_rank"] for row in rows if row["dp_rank"] is not None})
    return {"expected_dp_ranks": expected_dp, "observed_dp_ranks": observed,
        "arm": arm, "by_dp_rank": by_rank, "traces": rows, "limits": LIMITS}


def require_trace_evidence(summary, arm):
    paper.require(summary["observed_dp_ranks"] == list(range(summary["expected_dp_ranks"])),
                  "Missing/unexpected trace ranks")
    for rank, row in summary["by_dp_rank"].items():
        counts = collections.Counter(row["counts"])
        paper.require(counts["cpu_events"] > 0 and counts["cuda_kernel_events"] > 0,
                      f"DP{rank}: missing CPU+CUDA evidence")
        paper.require(counts["moe_cpu_calls"] >= 48 and int(row["moe_hidden_T_counts"].get("16384", 0)) >= 48,
                      f"DP{rank}: insufficient positive exact-T16384 MoE-op coverage")
        paper.require(counts["expert_kernel_calls"] >= 48, f"DP{rank}: missing positive expert-kernel evidence")
        paper.require(not counts["cuda_graph_launch_api_events"], f"DP{rank}: unexpected CUDA graph launch")
        paper.require(not counts["unexpected_selector_kernel_calls"], f"DP{rank}: experimental selector kernel")
        if arm == "stock":
            paper.require(counts["stock_router_kernel_calls"] >= 48 and not counts["original_router_kernel_calls"],
                          f"DP{rank}: stock router identity not proven")
        else:
            paper.require(counts["original_router_kernel_calls"] >= 48 and not counts["stock_router_kernel_calls"],
                          f"DP{rank}: original placement router identity not proven")


def launch(args, arm, prompts, identity):
    directory = args.output / ("profile_" + arm)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists() and paper.read(path).get("complete"):
        saved = paper.read(path)
        paper.require(saved.get("identity_sha256") == paper.digest(args.output / "manifest.json"),
                      "Completed profile identity changed")
        summary = summarize_traces(Path(saved["trace_directory"]), args.dp, arm)
        require_trace_evidence(summary, arm)
        paper.require(summary == saved["trace_summary"], "Completed raw traces changed")
        print(f"[resume] {directory.name}", flush=True)
        return saved
    attempt = str(time.time_ns())
    if path.exists():
        path.rename(directory / f"result.previous.{attempt}.json")
        for name in ("server.log", "server.json", "pid.json"):
            previous = directory / name
            if previous.exists():
                previous.rename(directory / f"{name}.previous.{attempt}")
    traces = directory / ("traces_" + attempt)
    traces.mkdir()
    result = {"arm": arm, "complete": False, "measurement_type": "profile_only",
        "identity_sha256": paper.digest(args.output / "manifest.json"), "trace_directory": str(traces),
        "warmups": [], "profile_workload": None, "limits": LIMITS}
    common.atomic_json(path, result)
    try:
        with serve(args, arm, directory, traces, identity):
            check_log(args, arm, directory / "server.log")
            for index in range(3):
                result["warmups"].append(workload(args, WARMUP, prompts, directory))
                common.atomic_json(path, result)
                print(f"[warmup] {directory.name} {index + 1}/3", flush=True)
            result["runtime_evidence_after_warmups"] = check_log(args, arm, directory / "server.log")
            common.post_control(args, "/start_profile")
            result["profiler_start_returned_unix_ns"] = time.time_ns()
            try:
                result["profile_workload"] = workload(args, PROFILE, prompts[:PROFILE.requests], directory)
            finally:
                common.post_control(args, "/stop_profile")
                result["profiler_stop_returned_unix_ns"] = time.time_ns()
                common.atomic_json(path, result)
        result["runtime_evidence"] = check_log(args, arm, directory / "server.log")
        result["trace_summary"] = summarize_traces(traces, args.dp, arm)
        require_trace_evidence(result["trace_summary"], arm)
        result["complete"] = True
    except BaseException as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, result)
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("plan", "run"), default="plan")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model")
    p.add_argument("--checkpoint-proof", type=Path)
    p.add_argument("--revision")
    p.add_argument("--dp", type=int, choices=(4, 8), required=True)
    p.add_argument("--vllm", default="vllm")
    p.add_argument("--gpu-util", type=float, default=.88)
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--boot-timeout", type=float, default=2400)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--profile-timeout", type=float, default=1200)
    args = p.parse_args(argv)
    if not 0 < args.gpu_util < 1 or min(args.boot_timeout, args.request_timeout, args.profile_timeout) <= 0:
        p.error("GPU fraction must be in(0,1); timeouts must be positive")
    if args.mode == "run" and (not args.model or not args.checkpoint_proof):
        p.error("--model and --checkpoint-proof are required for run")
    args.output = args.output.resolve()
    if args.checkpoint_proof:
        args.checkpoint_proof = args.checkpoint_proof.resolve()
    args.url = f"http://127.0.0.1:{args.port}"
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == "plan":
        print(json.dumps({"configuration": configuration(args), "fresh_servers": 2,
            "production_source_expected_sha256": paper.ORIGINAL_SOURCE,
            "checkpoint_proof_required": True, "limits": LIMITS}, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".paper-profile.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity, prompts = ensure_manifest(args)
        metadata_path = args.output / "runtime_metadata.json"
        if metadata_path.exists():
            metadata_path = args.output / f"runtime_metadata_resume_{time.time_ns()}.json"
        common.atomic_json(metadata_path, common.metadata())
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"Stopped by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        for arm in ARMS:
            launch(args, arm, prompts, identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
