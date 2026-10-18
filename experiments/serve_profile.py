#!/usr/bin/env python3
"""Controlled DP/EP serving measurements, followed by separate CPU+CUDA profiles.

Example (inside the GPU pod, with this package installed):
  python experiments/serve_profile.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
      --output /results/qwen-ep4 --repeats 3

The default experiment uses fixed, seeded natural prose, stock/no-op/plain/balanced
routing, DP=4 TP=1 and DeepEP HT. Every measurement arm is a fresh server, without
PLACEMENT_PROBE or an active profiler. Profiles are fresh, independent launches.
Warmups are explicit and never selected using the measured speed. All raw request
records, commands, logs, hardware metadata and traces are retained. Re-running the
same command resumes completed arms, rejecting a different experiment manifest.

Streaming ITL is reported only when all chunks contain exactly one token; chunk
arrival gaps and TPOT are always retained. A CPU timer on the HTTP client measures
user-visible arrival times; these are not CUDA kernel times. Trace kernel durations
are sums of events, which may overlap and must not be read as critical-path times.
No external dataset or API key is required; the original prose below is a small
controlled workload, not a claim of production traffic representativeness.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import dataclasses
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"stock": (None, None), "noop8": (8, 0), "plain6": (6, 0), "balanced6p1": (6, 1)}
PROSE = [
    "The town library opened a workshop where neighbors could repair household objects. "
    "Maya brought a lamp with a loose switch, and Daniel carried a wooden chair with an uneven leg. "
    "A volunteer first checked which repairs were safe, then wrote down the tools and time needed. "
    "By the afternoon they had fixed four lamps and three chairs. The group decided to keep spare "
    "parts in labeled drawers and publish a short guide explaining each repair.",
    "A research team placed water sensors along a river to understand how rainfall affects its flow. "
    "They measured temperature and depth every fifteen minutes, checked the batteries weekly, and "
    "compared each sensor with a manual reading. One storm raised the river overnight, but a nearby "
    "station failed to send data. The analysts marked the missing observations before calculating "
    "averages, rather than treating the gap as a period of unusually low water.",
    "A small software team planned a booking system for a community garden. Each member could "
    "reserve one plot, record watering visits, and exchange unused time with a neighbor. The first "
    "version stored reservations in a database with a unique constraint on the plot and date. "
    "When two members requested the same slot, the application explained the conflict and offered "
    "the next available time. The team wrote examples covering cancellations and daylight saving changes.",
    "The train museum prepared an exhibition about the workers who maintained its oldest railway. "
    "Letters described early starts, difficult winters, and the skill needed to inspect a damaged rail. "
    "The curator arranged photographs beside maps so visitors could connect each story to a place. "
    "Students interviewed retired engineers, compared their memories with maintenance records, and "
    "created an audio guide that explained both the achievements and the limits of the surviving evidence.",
    "A bakery tested a new schedule to reduce food waste without running out of bread at lunchtime. "
    "Staff counted sales by hour, recorded the weather, and noted nearby events. Instead of changing "
    "all recipes at once, they compared two baking schedules on similar weekdays. Warm rolls sold "
    "quickly in the morning, while larger loaves were more popular after work. The owner reviewed "
    "the remaining stock each evening and adjusted the following week's plan.",
    "An engineering class designed a bridge from paper and string. Each group predicted the weight "
    "its design could carry before testing it with small metal washers. A wide deck made one bridge "
    "easy to use, but weak joints limited its strength. Another design distributed the load through "
    "several triangular supports. After the test, the students described the failure point, compared "
    "their predictions with observations, and proposed a single change for the next version.",
]
TASKS = ["Summarize the practical lessons and explain the evidence for each one.",
         "Compare the decisions described here and propose a careful follow-up plan.",
         "Explain the sequence of events, then identify assumptions worth checking.",
         "Write an accessible analysis with concrete recommendations and their tradeoffs."]


@dataclasses.dataclass(frozen=True)
class Shape:
    name: str
    prompt_tokens: int
    output_tokens: int
    concurrency: int
    requests: int


SHAPES = {"decode": Shape("decode", 256, 256, 4, 16),
          "prefill": Shape("prefill", 4096, 32, 32, 64),
          "mixed": Shape("mixed", 1024, 128, 16, 64)}


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def capture(cmd: list[str], cwd=None) -> dict:
    try:
        p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=40)
        return {"command": cmd, "returncode": p.returncode, "output": p.stdout}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": cmd, "error": str(e)}


def metadata() -> dict:
    packages = {}
    for name in ("vllm", "torch", "triton", "transformers", "placement-router"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version, "packages": packages,
            "hostname": os.uname().nodename, "invocation": shlex.join(sys.argv),
            "git_commit": capture(["git", "rev-parse", "HEAD"], ROOT),
            "git_status": capture(["git", "status", "--short"], ROOT),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "hardware": capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id,compute_mode", "--format=csv"]),
            "topology": capture(["nvidia-smi", "topo", "-m"]),
            "selected_environment": {k: os.environ[k] for k in (
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "VLLM_USE_V1",
                "VLLM_ATTENTION_BACKEND", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
                "NCCL_IB_HCA", "VLLM_ALL2ALL_BACKEND") if k in os.environ}}


def make_prompts(tokenizer, shapes: list[Shape], seed: int, source_text: str | None = None) -> dict:
    """Persist IDs as well as prose: every arm receives precisely the same tokens."""
    out = {}
    corpus_ids = tokenizer.encode(source_text, add_special_tokens=False) if source_text else None
    for shape in shapes:
        prompts = []
        for i in range(shape.requests):
            rng = random.Random(seed + i * 1009 + shape.prompt_tokens)
            task = tokenizer.encode("\n\n" + TASKS[i % len(TASKS)] + "\nAnswer:\n",
                                    add_special_tokens=False)
            needed = shape.prompt_tokens - len(task)
            if corpus_ids is not None:
                if len(corpus_ids) < needed:
                    raise ValueError("Text corpus is shorter than the requested prompt shape")
                offset = rng.randrange(len(corpus_ids) - needed + 1)
                ids = corpus_ids[offset:offset + needed] + task
            else:
                paragraphs = PROSE.copy()
                rng.shuffle(paragraphs)
                text = f"Case study {i + 1}. Read the following notes carefully.\n\n"
                text += "\n\n".join(paragraphs * max(1, math.ceil(shape.prompt_tokens / 250)))
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) < needed:
                    raise ValueError("Natural prose template is too short for the requested shape")
                ids, offset = ids[:needed] + task, None
            prompts.append({"request_id": i, "token_ids": ids, "corpus_token_offset": offset,
                            "text": tokenizer.decode(ids, skip_special_tokens=False)})
        out[shape.name] = prompts
    return out


def sse_events(response):
    """Parse SSE data fields, including comments and multiline events."""
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield "\n".join(lines)


def stream_request(url: str, model: str, prompt: dict, output_tokens: int,
                   timeout: int, index: int) -> dict:
    # vLLM's return_token_ids exposes delta token IDs without computing logprobs.
    # Verify single-token chunks instead of assuming text chunks equal tokens.
    body = {"model": model, "prompt": prompt["token_ids"], "max_tokens": output_tokens,
            "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}, "return_token_ids": True}
    request = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    start = time.perf_counter()
    chunks, usage, done, finish_reason = [], None, False, None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for payload in sse_events(response):
            now = time.perf_counter()
            if payload == "[DONE]":
                done = True
                break
            item = json.loads(payload)
            if "error" in item:
                raise RuntimeError(f"Streaming error: {item['error']}")
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                token_names = choice.get("token_ids")
                if token_names is None:
                    token_names = (choice.get("logprobs") or {}).get("tokens", [])
                if token_names or choice.get("text"):
                    chunks.append({"arrival_seconds": now - start,
                                   "tokens": len(token_names) if token_names else None,
                                   "text_chars": len(choice.get("text", ""))})
    end = time.perf_counter()
    if not done or not usage or not chunks:
        raise RuntimeError(f"Incomplete stream: done={done}, usage={usage}, chunks={len(chunks)}")
    if usage["completion_tokens"] != output_tokens or usage["prompt_tokens"] != len(prompt["token_ids"]):
        raise RuntimeError(f"Token budget mismatch: {usage}; expected {len(prompt['token_ids'])}/{output_tokens}")
    if all(c["tokens"] is not None for c in chunks):
        if sum(c["tokens"] for c in chunks) != usage["completion_tokens"]:
            raise RuntimeError("Streaming token counts disagree with final usage")
    first, last = chunks[0]["arrival_seconds"], chunks[-1]["arrival_seconds"]
    gaps = [(b["arrival_seconds"] - a["arrival_seconds"]) * 1000
            for a, b in zip(chunks, chunks[1:])]
    all_single = all(c["tokens"] == 1 for c in chunks)
    return {"request_id": index, "prompt_id": prompt["request_id"],
            "ttft_ms": first * 1000, "latency_ms": (end - start) * 1000,
            "last_token_latency_ms": last * 1000,
            "tpot_ms": (last - first) * 1000 / (output_tokens - 1) if output_tokens > 1 else None,
            "itl_ms": gaps if all_single else None, "inter_chunk_ms": gaps,
            "all_chunks_single_token": all_single, "chunks": chunks,
            "usage": usage, "finish_reason": finish_reason}


def percentiles(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    def q(p):
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": q(.5), "p95": q(.95), "p99": q(.99), "max": ordered[-1]}


def run_workload(args, shape: Shape, prompts: list[dict], *, count=None, output_tokens=None) -> dict:
    count = count or shape.requests
    generation = output_tokens or shape.output_tokens
    started_unix_ns = time.time_ns()
    started = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=min(shape.concurrency, count)) as executor:
        futures = [executor.submit(stream_request, args.url, args.model,
                                  prompts[i % len(prompts)], generation,
                                  args.request_timeout, i) for i in range(count)]
        # Propagate every failure; an incomplete workload never becomes a result.
        records = [f.result() for f in futures]
    elapsed = time.perf_counter() - started
    n_out = sum(r["usage"]["completion_tokens"] for r in records)
    n_in = sum(r["usage"]["prompt_tokens"] for r in records)
    return {"started_unix_ns": started_unix_ns, "finished_unix_ns": time.time_ns(),
            "elapsed_seconds": elapsed, "requests": count, "concurrency": min(shape.concurrency, count),
            "prompt_tokens": n_in, "completion_tokens": n_out,
            "requests_per_second": count / elapsed, "output_tokens_per_second": n_out / elapsed,
            "total_tokens_per_second": (n_in + n_out) / elapsed,
            "ttft_ms": percentiles([r["ttft_ms"] for r in records]),
            "request_latency_ms": percentiles([r["latency_ms"] for r in records]),
            "tpot_ms": percentiles([r["tpot_ms"] for r in records if r["tpot_ms"] is not None]),
            "itl_ms": percentiles([v for r in records for v in (r["itl_ms"] or [])]),
            "requests_with_single_token_chunks": sum(r["all_chunks_single_token"] for r in records),
            "inter_chunk_ms": percentiles([v for r in records for v in r["inter_chunk_ms"]]),
            "records": records}


def post_control(args, path: str) -> None:
    req = urllib.request.Request(args.url + path, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=args.profile_timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}")


def server_command(args, trace_dir: Path | None) -> list[str]:
    command = [args.vllm, "serve", args.model, "--host", "127.0.0.1", "--port", str(args.port),
               "--data-parallel-size", str(args.dp), "--tensor-parallel-size", "1",
               "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput",
               "--max-model-len", str(args.max_model_len), "--seed", str(args.seed),
               "--max-num-batched-tokens", str(args.batch_tokens),
               "--max-num-seqs", str(args.max_num_seqs),
               "--no-enable-prefix-caching", "--disable-log-stats",
               "--gpu-memory-utilization", str(args.gpu_util)]
    if args.revision:
        command += ["--revision", args.revision]
    if args.trust_remote_code:
        command += ["--trust-remote-code"]
    if trace_dir:
        config = {"profiler": "torch", "torch_profiler_dir": str(trace_dir),
                  "torch_profiler_with_stack": False, "torch_profiler_record_shapes": True,
                  "torch_profiler_use_gzip": True}
        command += ["--profiler-config", json.dumps(config)]
    return command + args.server_arg


def arm_environment(arm: str) -> dict:
    env = os.environ.copy()
    for name in ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY", "PLACEMENT_PROBE",
                 "VLLM_TORCH_PROFILER_DIR"):
        env.pop(name, None)
    keep, penalty = ARMS[arm]
    if keep is not None:
        env.update(PLACEMENT_KEEP=str(keep), PLACEMENT_LOAD_PENALTY=str(penalty))
    return env


def kill_server(process) -> None:
    # Children can outlive the parent; address the original process group even if
    # the launcher has already exited, and reap the parent in both branches.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=10)


@contextlib.contextmanager
def serve(args, arm: str, directory: Path, trace_dir: Path | None):
    command = server_command(args, trace_dir)
    env = arm_environment(arm)
    atomic_json(directory / "server.json", {"command": command, "shell_command": shlex.join(command),
                "placement_environment": {k: env.get(k) for k in
                                          ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY", "PLACEMENT_PROBE")},
                "profiler_active_during_measurements": False})
    # Never mistake an already-running server for the one this arm launched.
    import socket
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError(f"Port {args.port} is already occupied")
    log_path = directory / "server.log"
    if log_path.exists():
        log_path.rename(directory / f"server.previous.{time.time_ns()}.log")
    with log_path.open("w", buffering=1) as log:
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(directory / "pid.json", {"pid": process.pid, "process_group": process.pid})
        print(f"[launch] {directory.name}: {shlex.join(command)}", flush=True)
        try:
            deadline = time.monotonic() + args.boot_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; see {directory / 'server.log'}")
                try:
                    with urllib.request.urlopen(args.url + "/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, OSError):
                    time.sleep(2)
            else:
                raise TimeoutError(f"Server did not become healthy after {args.boot_timeout}s")
            yield process
        finally:
            kill_server(process)


def summarize_traces(trace_dir: Path, expected_dp: int) -> dict:
    files = sorted(trace_dir.rglob("*.pt.trace.json*"))
    summaries, ranks = [], set()
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as f:
            trace = json.load(f)
        events = trace.get("traceEvents", [])
        kernels = [e for e in events if "kernel" in str(e.get("cat", "")).lower()
                   and e.get("ph") == "X"]
        cpu_count = sum(e.get("ph") == "X" and str(e.get("cat", "")) in
                        ("cpu_op", "user_annotation", "python_function") for e in events)
        rank = re.search(r"(?:^|_)dp(\d+)(?:_|\.)", path.name)
        if kernels and rank:
            ranks.add(int(rank.group(1)))
        grouped = {}
        for e in kernels:
            name = e.get("name", "unknown")
            entry = grouped.setdefault(name, {"calls": 0, "summed_duration_us": 0.0})
            entry["calls"] += 1
            entry["summed_duration_us"] += e.get("dur", 0)
        categories = {}
        for category, pattern in {
            "router": r"topk|top_k|select_kernel|_par_select|placement|router",
            "dispatch": r"dispatch|intranode.*send|internode.*send",
            "gemm": r"gemm|cutlass|grouped_mm|matmul|mma",
            "combine": r"combine"}.items():
            matched = {k: v for k, v in grouped.items() if re.search(pattern, k, re.I)}
            categories[category] = {"calls": sum(v["calls"] for v in matched.values()),
                                    "summed_duration_us": sum(v["summed_duration_us"] for v in matched.values()),
                                    "matching_kernel_names": list(matched)}
        summaries.append({"path": str(path), "bytes": path.stat().st_size,
                          "dp_rank": int(rank.group(1)) if rank else None,
                          "cpu_events": cpu_count, "cuda_kernel_events": len(kernels),
                          "categories": categories,
                          "top_kernels_by_summed_duration": sorted(
                              ({"name": k, **v} for k, v in grouped.items()),
                              key=lambda v: v["summed_duration_us"], reverse=True)[:40]})
        del trace, events, kernels
    valid = ranks == set(range(expected_dp)) and all(
        any(x["dp_rank"] == rank and x["cpu_events"] and x["cuda_kernel_events"] for x in summaries)
        for rank in range(expected_dp))
    return {"valid_all_dp_ranks_cpu_and_cuda": valid, "expected_dp_ranks": expected_dp,
            "observed_cuda_dp_ranks": sorted(ranks), "traces": summaries,
            "duration_note": "Kernel durations are summed and may overlap; categories use kernel-name heuristics."}


def check_active(arm: str, log: Path, dp: int):
    text = log.read_text(errors="replace")
    expected = ARMS[arm][0]
    if expected is not None:
        message = f"placement-aware routing active, keep={expected} of top_k=8, ep_size={dp}"
        if message not in text:
            raise RuntimeError(f"Expected active placement routing was not seen in {log}: {message}")
    if "[moet]" in text:
        raise RuntimeError("Timing probe was unexpectedly active during an uninstrumented experiment")


def launch_measurement(args, arm, repeat, shapes, prompts):
    directory = args.output / f"r{repeat:02d}_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "result.json"
    if result_file.exists() and json.loads(result_file.read_text()).get("complete"):
        print(f"[resume] {directory.name}", flush=True)
        return
    if result_file.exists():
        result_file.rename(directory / f"result.previous.{time.time_ns()}.json")
    result = {"arm": arm, "repeat": repeat, "complete": False,
              "measurement_type": "uninstrumented_streaming_http", "shapes": {}}
    atomic_json(result_file, result)
    try:
        with serve(args, arm, directory, None):
            for shape in shapes:
                shape_result = {"configuration": dataclasses.asdict(shape), "warmups": [], "rounds": []}
                result["shapes"][shape.name] = shape_result
                for i in range(args.warmup_rounds):
                    print(f"[warmup] {directory.name} {shape.name} {i + 1}/{args.warmup_rounds}", flush=True)
                    # Exercise the measured concurrency and token shape, including
                    # long decode, graph capture, and DeepEP scratch allocation.
                    warmup = run_workload(args, shape, prompts[shape.name], count=max(shape.concurrency, 4))
                    shape_result["warmups"].append(warmup)
                    atomic_json(result_file, result)
                for i in range(args.rounds):
                    measured = run_workload(args, shape, prompts[shape.name])
                    shape_result["rounds"].append(measured)
                    print(f"[measure] {directory.name} {shape.name} {i + 1}/{args.rounds}: "
                          f"{measured['output_tokens_per_second']:.2f} out-tok/s; "
                          f"TTFT p50={measured['ttft_ms']['p50']:.1f}ms", flush=True)
                    atomic_json(result_file, result)
                check_active(arm, directory / "server.log", args.dp)
        result["complete"] = True
    except BaseException as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        atomic_json(result_file, result)


def launch_profile(args, arm, shapes, prompts):
    directory = args.output / f"profile_{arm}"
    directory.mkdir(parents=True, exist_ok=True)
    result_file = directory / "result.json"
    if result_file.exists() and json.loads(result_file.read_text()).get("complete"):
        print(f"[resume] {directory.name}", flush=True)
        return
    if result_file.exists():
        result_file.rename(directory / f"result.previous.{time.time_ns()}.json")
    # Keep retry traces separate; never validate stale worker traces as a new capture.
    trace_dir = directory / f"traces_{time.time_ns()}"
    trace_dir.mkdir()
    result = {"arm": arm, "complete": False, "measurement_type": "profile_only",
              "trace_directory": str(trace_dir), "shapes": {}}
    atomic_json(result_file, result)
    try:
        with serve(args, arm, directory, trace_dir):
            for shape in shapes:
                run_workload(args, shape, prompts[shape.name], count=max(shape.concurrency, args.dp), output_tokens=16)
            post_control(args, "/start_profile")
            try:
                for shape in shapes:
                    # Four waves cover every DP worker even with internal load balancing.
                    result["shapes"][shape.name] = run_workload(
                        args, shape, prompts[shape.name], count=max(args.dp * 4, shape.concurrency),
                        output_tokens=args.profile_output_tokens)
            finally:
                post_control(args, "/stop_profile")
            check_active(arm, directory / "server.log", args.dp)
        result["trace_summary"] = summarize_traces(trace_dir, args.dp)
        if not result["trace_summary"]["valid_all_dp_ranks_cpu_and_cuda"]:
            raise RuntimeError("Profile capture lacks CPU+CUDA traces for every DP rank; see result.json")
        result["complete"] = True
    except BaseException as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        atomic_json(result_file, result)


def aggregate(args):
    results = []
    for path in sorted(args.output.glob("r*_*/result.json")):
        item = json.loads(path.read_text())
        if item.get("complete"):
            for name, shape in item["shapes"].items():
                rounds = shape["rounds"]
                results.append({"arm": item["arm"], "repeat": item["repeat"], "shape": name,
                                "median_output_tokens_per_second": statistics.median(r["output_tokens_per_second"] for r in rounds),
                                "median_total_tokens_per_second": statistics.median(r["total_tokens_per_second"] for r in rounds),
                                "median_ttft_p50_ms": statistics.median(r["ttft_ms"]["p50"] for r in rounds),
                                "median_request_latency_p99_ms": statistics.median(r["request_latency_ms"]["p99"] for r in rounds)})
    atomic_json(args.output / "summary.json", {"launches": results,
        "note": "Independent server launches are the repeat unit. No throughput-based filtering. Profile workloads are excluded. Small repeat counts do not resolve small speedups reliably."})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--text-file", type=Path, help="UTF-8 natural-text corpus; sample reproducible token windows")
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--profile-arms", nargs="*", choices=ARMS, default=["stock", "balanced6p1"])
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=["decode", "prefill"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--boot-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--profile-timeout", type=int, default=600)
    parser.add_argument("--profile-output-tokens", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--gpu-util", type=float, default=.88)
    parser.add_argument("--vllm", default="vllm")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--server-arg", action="append", default=[], help="Extra server argument, e.g. --server-arg=--enforce-eager")
    parser.add_argument("--profiles-only", action="store_true")
    parser.add_argument("--profile-first", action="store_true", help="Verify short CPU+CUDA profiles before the measurement sweeps")
    parser.add_argument("--plan-only", action="store_true", help="Materialize prompts and manifest without starting any server")
    args = parser.parse_args()
    if min(args.dp, args.repeats, args.rounds, args.warmup_rounds, args.profile_output_tokens) < 1:
        parser.error("dp/repeats/rounds/warmup-rounds/profile-output-tokens must be positive")
    shapes = [SHAPES[name] for name in args.shapes]
    if any(s.prompt_tokens + max(s.output_tokens, 16, args.profile_output_tokens) > args.max_model_len for s in shapes):
        parser.error("max-model-len is smaller than a workload's prompt + generation")
    if args.text_file:
        args.text_file = args.text_file.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    # Hold the descriptor until main exits, so two resumptions cannot overwrite
    # each other's state or launch competing servers from this results directory.
    experiment_lock = (args.output / ".runner.lock").open("a")
    try:
        fcntl.flock(experiment_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another runner is already using this output directory") from None
    args.url = f"http://127.0.0.1:{args.port}"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in ("plan_only", "profiles_only", "profile_first")}
    order = args.arms.copy()
    random.Random(args.seed).shuffle(order)
    schedule = [{"repeat": i, "arms": order[i % len(order):] + order[:i % len(order)]}
                for i in range(args.repeats)]
    identity = {"config": config, "schedule": schedule,
                "workloads": [dataclasses.asdict(s) for s in shapes],
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "prose_sha256": hashlib.sha256(json.dumps([PROSE, TASKS]).encode()).hexdigest(),
                "corpus_sha256": hashlib.sha256(args.text_file.read_bytes()).hexdigest() if args.text_file else None}
    manifest = args.output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["identity"] != identity:
            raise RuntimeError("Output contains a different experiment manifest; choose a new --output")
    else:
        atomic_json(manifest, {"identity": identity, "metadata": metadata(),
                              "protocol": "Fixed closed-loop concurrency; latency starts when a worker sends its request, excluding local executor queue time. Prefix caching disabled. Each sweep rotates a seeded arm order. Warmups excluded, no adaptive rejection of measured rounds. Token budget checked against SSE final usage."})
    prompt_file = args.output / "prompts.json"
    if prompt_file.exists():
        prompts = json.loads(prompt_file.read_text())
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                                  trust_remote_code=args.trust_remote_code)
        source_text = args.text_file.read_text() if args.text_file else None
        prompts = make_prompts(tokenizer, shapes, args.seed, source_text)
        atomic_json(prompt_file, prompts)
        atomic_json(args.output / "prompts_metadata.json", {
            "tokenizer_name_or_path": tokenizer.name_or_path,
            "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": type(tokenizer).__name__,
            "corpus_sha256": identity["corpus_sha256"],
            "prompts_file_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest()})
    print(f"Experiment: {manifest}; schedule={schedule}", flush=True)
    if args.plan_only:
        return
    # A signal must unwind the server context and terminate its entire group.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if args.profile_first:
            for arm in args.profile_arms:
                launch_profile(args, arm, shapes, prompts)
        if not args.profiles_only:
            for sweep in schedule:
                for arm in sweep["arms"]:
                    launch_measurement(args, arm, sweep["repeat"], shapes, prompts)
                    aggregate(args)
        if not args.profile_first:
            for arm in args.profile_arms:
                launch_profile(args, arm, shapes, prompts)
    finally:
        aggregate(args)


if __name__ == "__main__":
    main()
