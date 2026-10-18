#!/usr/bin/env python3
"""Controlled lambda=8 prefill serving replication; no cluster operations.

The timed target has no placement probe, profiler, CUDA events, or timer hooks.
Routing contracts are verified on distinct preflight servers. Every launch, raw
response (including partial failures), round and warning is retained. Nothing is
retried or filtered by speed. Statistical observations are six fresh sweeps.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
CONFIG_SHA = "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438"
PROMPTS_SHA = "7660341b6e1419c80a71caec4a4c364c7e44339c6089c7c60d62390d1975f479"
SOURCE = {
    "src/placement_router/__init__.py": "8425741d4c8d90b5ce2ce122cff4abb8a095b8e488ce9e18ce468a0916f5d0cf",
    "src/placement_router/kernel.py": "06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854",
    "src/placement_router/plugin.py": "c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8",
    "src/placement_router/vendor/__init__.py": "623bf62f4ae70782a32c1691d3648c120c67c5e4853725d3881265b7dfd96b25",
    "src/placement_router/vendor/fused_topk_router.py": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc"}
ARMS = {"stock": (None, None), "keep8": (8, 0.0), "plain6": (6, 0.0),
        "keep6p8": (6, 8.0), "keep4p8": (4, 8.0), "stock_repeat": (None, None)}
WILLIAMS = [[(x + row) % 6 for x in [0, 1, 5, 2, 4, 3]] for row in range(6)]
ENV_KEYS = ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH",
            "VLLM_LOGGING_LEVEL", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY", "VLLM_PLUGINS")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def save(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not exclusive or not path.exists(), "Refusing to overwrite " + str(path))
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bytes_sha(value):
    return hashlib.sha256(value).hexdigest()


def schedule():
    labels = list(ARMS)
    return [{"sweep": row, "arms": [labels[i] for i in order]}
            for row, order in enumerate(WILLIAMS)]


def process_record(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "state": fields[0], "pgrp": int(fields[2]), "start_ticks": fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_members(pgid):
    values = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            value = process_record(int(entry.name))
            if value and value["pgrp"] == pgid and value["state"] != "Z":
                values.append(value)
    return values


def stop_owned(child, identity):
    require(identity and identity["pid"] == identity["pgrp"] == child.pid,
            "Cleanup requires a group created by this controller")
    current = process_record(child.pid)
    require(not current or current["start_ticks"] == identity["start_ticks"], "Owned PID reused")
    value = {"identity": identity, "signals": [], "started_unix_ns": time.time_ns()}
    for signum, duration in [(signal.SIGTERM, 40), (signal.SIGKILL, 15)]:
        if not group_members(child.pid):
            break
        try:
            os.killpg(child.pid, signum)
            value["signals"].append(signum.name)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + duration
        while group_members(child.pid) and time.monotonic() < deadline:
            child.poll()
            time.sleep(.2)
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    value.update(remaining_members=group_members(child.pid), returncode=child.poll(),
                 finished_unix_ns=time.time_ns())
    require(not value["remaining_members"], "Owned process group remains alive")
    return value


def capture(command, timeout=30):
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {"command": command, "returncode": done.returncode,
            "stdout": done.stdout, "stderr": done.stderr}


def gpu_idle(wait=30):
    deadline = time.monotonic() + wait
    while True:
        value = capture(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                         "--format=csv,noheader,nounits"])
        require(value["returncode"] == 0, "nvidia-smi compute query failed")
        if not value["stdout"].strip():
            return value
        require(time.monotonic() < deadline, "GPU already has compute processes: " + value["stdout"])
        time.sleep(1)


def source_identity(proof_path, production_root, site_root):
    proof = read(proof_path)
    require(proof.get("source_commit", proof.get("base_commit")) == COMMIT, "Wrong production commit")
    require(Path(proof["source_root"]).resolve() == Path(production_root).resolve(), "Source root/proof mismatch")
    require(Path(proof["site_root"]).resolve() == Path(site_root).resolve(), "Installed site/proof mismatch")
    result = {}
    for name, expected in SOURCE.items():
        for kind, path in [("public", Path(production_root) / name),
                           ("installed", Path(site_root) / name.removeprefix("src/"))]:
            require(sha(path) == expected, "Production source drift: " + str(path))
            result[kind + ":" + name] = expected
    return {"commit": COMMIT, "proof_sha256": sha(proof_path), "files": result}


def checkpoint_identity(proof_path, model):
    value = read(proof_path)
    require(value.get("canonical_identity_verified") is True and value.get("model_id") == MODEL_ID
            and value.get("revision") == REVISION, "Canonical checkpoint identity absent")
    require(Path(value["local_model_path"]).resolve() == Path(model).resolve(), "Model proof/path mismatch")
    require(value.get("canonical_config_sha256") == CONFIG_SHA
            and sha(Path(model) / "config.json") == CONFIG_SHA, "Canonical model config differs")
    files = value.get("safetensors", [])
    require(len(files) == 16 and len({x["filename"] for x in files}) == 16
            and value.get("weights_verified") == files, "Root must fully SHA-verify all16 canonical shards")
    signatures = {}
    for item in files:
        path = Path(model) / item["filename"]
        require(path.stat().st_size == item["bytes"], "Missing/truncated canonical model shard")
        signatures[item["filename"]] = [path.stat().st_size, path.stat().st_mtime_ns]
    return {"model_id": MODEL_ID, "revision": REVISION, "proof_sha256": sha(proof_path),
            "file_stat_signatures": signatures,
            "verification": "Root full-SHA proof for16 shards; controller freezes sizes/mtimes, no61GB reread/launch"}


def verify_prompts(path):
    require(sha(path) == PROMPTS_SHA, "Frozen64 synthetic prompt file differs")
    values = read(path)
    require(len(values) == 64 and [r["request_id"] for r in values] == list(range(64)), "Prompt IDs/size differ")
    require(all(len(r["token_ids"]) == 4096 and all(type(t) is int and 1 <= t < 100000
                for t in r["token_ids"]) for r in values), "Prompt shape or token range differs")
    return values


def sanitized_env(plan, arm, *, probe=None):
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(("PLACEMENT_", "PAPER_PROBE_", "TASK_BENCH_", "PREFILL_BENCH_", "VLLM_"))
                or key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN", "VLLM_DEEPEP_BUFFER_SIZE_MB")):
            env.pop(key, None)
    env.update(plan["environment"])
    keep, penalty = ARMS[arm]
    if keep is not None:
        env.update(PLACEMENT_KEEP=str(keep), PLACEMENT_LOAD_PENALTY=str(penalty))
    if probe is not None:
        env.update(PLACEMENT_PROBE=plan["probe"], PREFILL_BENCH_DIR=str(probe),
                   PREFILL_BENCH_ARM=arm, PREFILL_BENCH_EP="4",
                   PREFILL_BENCH_SOURCE_PROOF=plan["source_proof"],
                   PREFILL_BENCH_MAX_NUM_SEQS=str(plan["max_num_seqs"]))
    require(probe is not None or not any(k.startswith(("PAPER_PROBE_", "TASK_BENCH_", "PREFILL_BENCH_"))
            or k == "PLACEMENT_PROBE" or "PROFILER" in k for k in env), "Timed server has instrumentation environment")
    return env


def server_command(plan):
    return [plan["vllm"], "serve", plan["model"], "--served-model-name", "prefill-model",
            "--host", "127.0.0.1", "--port", str(plan["port"]),
            "--data-parallel-size", "4", "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
            "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput",
            "--dtype", "bfloat16", "--expert-placement-strategy", "linear",
            '--compilation-config={"cudagraph_mode":"NONE"}', "--max-model-len", "4105", "--seed", "0",
            "--max-num-batched-tokens", "16384", "--max-num-seqs", str(plan["max_num_seqs"]),
            "--no-enable-prefix-caching", "--disable-log-stats", "--jit-monitor-verbose",
            "--gpu-memory-utilization", str(plan["gpu_util"])]


def cpu_import_identity(plan, env):
    """Resolve actual target Python import paths; no torch/vLLM/model import."""
    script = r'''import hashlib,importlib.metadata,importlib.util,json,sys
from pathlib import Path
names=['placement_router','placement_router.plugin','placement_router.kernel','placement_router.vendor','placement_router.vendor.fused_topk_router']
paths={name:importlib.util.find_spec(name).origin for name in names}
print(json.dumps({'python':sys.executable,'module_paths':paths,'module_sha256':{name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in paths.items()},'entry_points':[(e.name,e.value) for e in importlib.metadata.entry_points(group='vllm.general_plugins') if e.name=='placement_aware_router']}))'''
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)
    require(done.returncode == 0, "CPU target import resolution failed: " + done.stderr)
    value = json.loads(done.stdout)
    for name, path in value["module_paths"].items():
        relative = name.replace(".", "/")
        relative += "/__init__.py" if name in ["placement_router", "placement_router.vendor"] else ".py"
        require(Path(path).resolve() == (Path(plan["site_root"]) / relative).resolve()
                and value["module_sha256"][name] == SOURCE["src/" + relative], "Target import resolves outside frozen site/source")
    require(value["entry_points"] == [["placement_aware_router", "placement_router.plugin:register"]],
            "Target plugin entrypoint missing/duplicated")
    return value


def make_body(row):
    return json.dumps({"model": "prefill-model", "prompt": row["token_ids"], "max_tokens": 1,
                       "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True}, "return_token_ids": True,
                       "add_special_tokens": False, "n": 1}, separators=(",", ":")).encode()


def parse_event(payload, value, arrival_ns, expected_prompt_ids=None):
    if payload == "[DONE]":
        value["done"] = True
        value["done_ns"] = arrival_ns
        return
    item = json.loads(payload)
    require("error" not in item, "Streaming server error: " + str(item.get("error")))
    if item.get("usage") is not None:
        require(value.get("usage") is None, "Duplicate final usage event")
        value["usage"] = item["usage"]
    for choice in item.get("choices", []):
        require(choice.get("index") == 0, "Unexpected completion choice index")
        if choice.get("prompt_token_ids") is not None:
            require(expected_prompt_ids is not None and choice["prompt_token_ids"] == expected_prompt_ids,
                    "Server prompt IDs differ from frozen4096 inputs")
            require(not value.get("prompt_ids_verified"), "Duplicate returned prompt IDs")
            value["prompt_ids_verified"] = True
        tokens = choice.get("token_ids")
        if tokens is None:
            require(not choice.get("text"), "Text without return_token_ids in stream choice")
            tokens = []  # API-valid empty/finish-only chunks may omit optional IDs.
        require(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens), "Bad token IDs")
        if tokens:
            value["chunks"].append({"arrival_ns": arrival_ns, "token_ids": tokens,
                                    "text": choice.get("text", "")})
        else:
            require(not choice.get("text"), "Text without actual token IDs")
        if choice.get("finish_reason") is not None:
            require(value.get("finish_reason") is None, "Duplicate finish event")
            value["finish_reason"] = choice["finish_reason"]


def stream_one(url, body, row, timeout, barrier, cohort):
    """Memory-only while timed. Caller persists every response after batch clock stops."""
    value = {"request_id": row["request_id"], "done": False, "chunks": [], "complete": False}
    raw = bytearray()
    try:
        barrier.wait(timeout=30)
        value["cohort_start_ns"] = cohort["start_ns"]
        value["request_start_ns"] = time.perf_counter_ns()
        value["request_start_unix_ns"] = time.time_ns()
        request = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
        deadline = time.monotonic() + timeout
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value["http_status"] = response.status
            value["content_type"] = response.headers.get("Content-Type")
            require(response.status == 200 and (value["content_type"] or "").startswith("text/event-stream"), "Expected HTTP200 SSE")
            fields = []
            while True:
                remaining = deadline - time.monotonic()
                require(remaining > 0, "Absolute request deadline reached before read")
                # urllib exposes the underlying HTTP socket here. Shrink each
                # read timeout to the absolute remaining budget; never reset600s.
                response.fp.raw._sock.settimeout(remaining)
                line = response.readline()
                arrival = time.perf_counter_ns()
                raw.extend(line)
                require(time.monotonic() < deadline, "Absolute request deadline reached")
                require(line, "EOF before [DONE]")
                require(len(raw) <= 2 * 1024 * 1024, "Unexpectedly large one-token response")
                decoded = line.decode("utf-8").rstrip("\r\n")
                if not decoded and fields:
                    parse_event("\n".join(fields), value, arrival, row["token_ids"])
                    fields = []
                    if value["done"]:
                        break
                elif decoded.startswith("data:"):
                    fields.append(decoded[5:].lstrip(" "))
        value["request_finished_ns"] = time.perf_counter_ns()
        usage = value.get("usage", {})
        require(usage.get("prompt_tokens") == 4096 and usage.get("completion_tokens") == 1,
                "One-request token accounting mismatch: " + str(usage))
        require(sum(len(c["token_ids"]) for c in value["chunks"]) == 1
                and value.get("finish_reason") == "length", "Expected exactly1 generated token and length finish")
        require(value.get("prompt_ids_verified") is True, "Server omitted exact input token-ID echo")
        first = value["chunks"][0]["arrival_ns"]
        value.update(complete=True, ttft_ms=(first - value["request_start_ns"]) / 1e6,
                     cohort_ttft_ms=(first - cohort["start_ns"]) / 1e6,
                     latency_ms=(value["done_ns"] - value["request_start_ns"]) / 1e6)
    except BaseException:
        value["error"] = traceback.format_exc()
        if isinstance(sys.exc_info()[1], urllib.error.HTTPError):
            error = sys.exc_info()[1]
            value["http_status"] = error.code
            try:
                raw.extend(error.read(2 * 1024 * 1024))
            except OSError:
                pass
    value["raw_bytes"] = len(raw)
    value["raw_sha256"] = bytes_sha(raw)
    return value, bytes(raw)


def run_round(plan, prompts, executor, directory, phase, number):
    bodies = [make_body(row) for row in prompts]  # serialization excluded, identical fixed inputs
    barrier = threading.Barrier(65)
    cohort = {"start_ns": time.perf_counter_ns(), "start_unix_ns": time.time_ns()}
    futures = [executor.submit(stream_one, plan["url"], body, row, plan["request_timeout"], barrier, cohort)
               for row, body in zip(prompts, bodies)]
    try:
        barrier.wait(timeout=30)
        cohort["release_ns"] = time.perf_counter_ns()
    except threading.BrokenBarrierError:
        cohort["barrier_error"] = "64 workers did not reach common barrier"
    results = [f.result() for f in futures]  # all workers retained even when one failed
    finished = time.perf_counter_ns()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    for row, body, (record, raw) in zip(prompts, bodies, results):
        prefix = f"request{row['request_id']:03d}"
        request_path, raw_path = directory / (prefix + ".request.json"), directory / (prefix + ".sse")
        request_path.write_bytes(body)
        raw_path.write_bytes(raw)
        record.update(request_file=request_path.name, request_sha256=bytes_sha(body), raw_file=raw_path.name)
        records.append(record)
    seconds = (finished - cohort["start_ns"]) / 1e9
    value = {"complete": all(r["complete"] for r in records) and "barrier_error" not in cohort,
             "phase": phase, "round": number, "clock": "time.perf_counter_ns",
             "cohort": cohort, "finished_ns": finished, "elapsed_seconds": seconds,
             "requests": 64, "concurrency": 64, "input_tokens": 64 * 4096, "output_tokens": 64,
             "input_tokens_per_second": 64 * 4096 / seconds,
             "total_tokens_per_second": 64 * 4097 / seconds,
             "records": records, "request_body_serialization_timed": False,
             "clock_scope": "Before submitting64 concurrent workers through all64 streams consumed/validated; raw disk writes excluded"}
    save(directory / "round.json", value, exclusive=True)
    require(value["complete"], "Round incomplete; all64 responses retained at " + str(directory))
    return {"round_file": str(directory / "round.json"), "round_sha256": sha(directory / "round.json"),
            "phase": phase, "round": number, "elapsed_seconds": seconds,
            "total_tokens_per_second": value["total_tokens_per_second"]}


def make_plan(args):
    output = str(args.output.resolve())
    require(not (Path(output) / "plan.json").exists(), "Plan already exists; do not overwrite")
    require(args.stop_unix > time.time() + 300, "Supply future absolute compute cutoff")
    model, source, site = (str(p.resolve()) for p in (args.model, args.production_root, args.site_root))
    directory = Path(__file__).resolve().parent
    prompts = directory / "prompts64.json"
    verify_prompts(prompts)
    packages = {}
    for name in ["vllm", "torch", "triton", "transformers", "deep_ep"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    allocation = read(args.allocation_proof)
    require(allocation.get("verified_live") is True and allocation.get("namespace") == "itay"
            and allocation.get("container") == "experiment" and allocation.get("gpu_count") == 8
            and "H100" in allocation.get("gpu_name", "")
            and all(isinstance(allocation.get(k), str) and allocation[k] for k in ["pod", "uid", "node", "image"]),
            "Allocation requires fresh exact8H100 pod identity")
    deployment_pod = read(args.source_proof).get("pod_identity", {})
    require(all(deployment_pod.get(key) == allocation.get(key) for key in ["namespace", "pod", "uid", "node", "image"]),
            "Source deployment and live allocation identify different pods")
    plan = {"schema": "prefill-e2e-v1", "created_unix_ns": time.time_ns(), "output": output,
            "model": model, "model_id": MODEL_ID, "revision": REVISION,
            "source_root": source, "site_root": site, "source_commit": COMMIT,
            "checkpoint_proof": str(args.checkpoint_proof.resolve()), "source_proof": str(args.source_proof.resolve()),
            "source_identity": source_identity(args.source_proof, source, site),
            "checkpoint_identity": checkpoint_identity(args.checkpoint_proof, model),
            "prompt_file": str(prompts), "prompt_sha256": PROMPTS_SHA, "probe": str(directory / "preflight_probe.py"),
            "vllm": str(args.vllm.resolve()), "port": args.port, "url": f"http://127.0.0.1:{args.port}",
            "max_num_seqs": args.max_num_seqs, "gpu_util": args.gpu_util,
            "warmup_rounds": 3, "measured_rounds": 6, "preflight_rounds": 2,
            "schedule": schedule(), "arms": ARMS, "stop_unix": args.stop_unix,
            "boot_timeout": args.boot_timeout, "request_timeout": args.request_timeout,
            "packages": packages, "allocation": allocation,
            "allocation_proof": str(args.allocation_proof.resolve()), "allocation_proof_sha256": sha(args.allocation_proof),
            "environment": {"PYTHONPATH": site, "CUDA_VISIBLE_DEVICES": args.devices,
                            "OMP_NUM_THREADS": "1", "NVSHMEM_QP_DEPTH": "4096", "VLLM_LOGGING_LEVEL": "INFO",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": "placement_aware_router"},
            "estimand": {"launch_summary": "arithmetic mean of exactly6 fixed measured cohort walltimes",
                         "baseline": "geometric mean of two fresh native stock launch summaries within same sweep",
                         "paired_ratio": "treatment launch mean walltime / geometricmean(stock,stock_repeat)",
                         "primary_unit": "one paired fresh six-arm sweep; n=6",
                         "throughput_ratio": "reciprocal of walltime ratio; total tokens fixed64x4097",
                         "interval": "descriptive95percent t(df5) interval on6 log ratios; exponentiate endpoints",
                       "filtering": "None. Fixed3 warmup rounds excluded by round index only; all raw preserved",
                         "rounds_requests_layers_are_replicates": False},
            "comparison_to_author": {"same": "EP4,64 concurrent independent HTTP requests,4096 seeded token IDs,1 output,lambda8,tile16,DeepEPHT,BF16,max_model_len4105",
                                     "controlled_differences": "No probe/timers; graphs explicitlyNONE; max_num_seqs explicit; fixed3+6rounds instead of6rounds/adaptive80percent-fastest filter; balancedorder; SSE returns tokenIDs and usage; exact checkpoint pinned",
                                     "not_established": "DP internal token packing/exactT16384 on every worker is not guaranteed by64 client requests"},
            "request_deadline": "Absolute budget checked between reads and socket timeout shrinks to remaining; conservative cutoff margin2xrequest_timeout+30 covers urllib connect/header phases"}
    require(len(args.devices.split(",")) == 4 and len(set(args.devices.split(","))) == 4, "Exactly4 distinct selected GPUs required")
    plan["server_command"] = server_command(plan)
    plan["files"] = {str(path): sha(path) for path in [Path(__file__).resolve(), directory / "preflight_probe.py",
                       prompts, directory / "prompts64.manifest.json", args.source_proof.resolve(), args.checkpoint_proof.resolve(),
                       args.allocation_proof.resolve(), args.vllm.resolve()]}
    save(Path(output) / "plan.json", plan, exclusive=True)
    return plan


class Campaign:
    def __init__(self, plan_path, mode):
        self.plan_path, self.plan, self.mode = Path(plan_path), read(plan_path), mode
        self.output = Path(self.plan["output"])
        self.root = self.output / mode
        self.root.mkdir(parents=True, exist_ok=False)
        self.lock = (self.output / "exclusive-gpu-controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = {"complete": False, "cleanup_complete": False, "mode": mode,
                      "plan_sha256": sha(plan_path), "controller_pid": os.getpid(),
                      "controller_process": process_record(os.getpid()), "controller_command": sys.argv,
                      "started_unix_ns": time.time_ns(), "launches": []}
        self.child = self.child_identity = None
        self.persist()

    def persist(self):
        self.state["updated_unix_ns"] = time.time_ns()
        save(self.root / "status.json", self.state)

    def validate(self):
        require(time.time() < self.plan["stop_unix"], "Absolute campaign compute cutoff reached")
        for path, expected in self.plan["files"].items():
            require(sha(path) == expected, "Frozen input changed: " + path)
        require(source_identity(self.plan["source_proof"], self.plan["source_root"], self.plan["site_root"])
                == self.plan["source_identity"], "Source identity changed")
        require(checkpoint_identity(self.plan["checkpoint_proof"], self.plan["model"])
                == self.plan["checkpoint_identity"], "Checkpoint stat/proof identity changed")
        require(self.plan["schedule"] == schedule() and self.plan["server_command"] == server_command(self.plan),
                "Frozen schedule/server command differs")
        require(str(self.plan["packages"]["vllm"]).split("+")[0] == "0.31.0", "Need pinned vLLM0.31.0 runtime")

    def wait_health(self):
        deadline = min(time.monotonic() + self.plan["boot_timeout"],
                       time.monotonic() + self.plan["stop_unix"] - time.time())
        while True:
            require(self.child.poll() is None, "Fresh server exited before health")
            require(time.monotonic() < deadline, "Fresh server startup deadline reached")
            try:
                with urllib.request.urlopen(self.plan["url"] + "/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            self.persist()
            time.sleep(2)

    def preflight_gate(self):
        path = self.output / "preflight" / "receipt.json"
        receipt = read(path)
        require(receipt.get("complete") and receipt.get("cleanup_complete")
                and receipt["plan_sha256"] == sha(self.plan_path), "Separate preflight incomplete/drifted")
        expected = set(ARMS) - {"stock_repeat"}
        require(len(receipt["launches"]) == 5 and {x["arm"] for x in receipt["launches"]} == expected,
                "Preflight requires exactly5 unique expected arms")
        paths = [x["receipt_file"] for x in receipt["launches"]]
        require(len(set(paths)) == 5, "Preflight launch receipt reused")
        for item in receipt["launches"]:
            require(item["complete"] and sha(item["receipt_file"]) == item["receipt_sha256"], "Preflight launch receipt changed")
            launch = read(item["receipt_file"])
            require(launch["complete"] and launch["arm"] == item["arm"]
                    and launch["plan_sha256"] == sha(self.plan_path), "Preflight receipt arm/plan differs")
            require(self.verify_probe(Path(launch["directory"]), item["arm"]) == launch["probe_files"],
                    "Preflight worker source/runtime proof differs")
            for probe in launch["probe_files"]:
                require(sha(probe["path"]) == probe["sha256"], "Preflight runtime evidence changed")
        self.state["preflight_receipt_sha256"] = sha(path)

    def verify_probe(self, directory, arm):
        paths = sorted((directory / "probe").glob("rank*_pid*.json"))
        require(len(paths) == 4, "Separate preflight must have exactly4 physical workers")
        values = [read(path) for path in paths]
        require({x["ep_rank"] for x in values} == set(range(4)), "Preflight rank set differs")
        for value in values:
            require(value["complete"] and value["arm"] == arm and value["all_routers_expected"]
                    and value["observed_router_objects"] == value["registered_layers"] == 48,
                    "Preflight did not validate all48 actual router objects")
            require(value["max_model_len"] == 4105 and value["max_num_seqs"] == self.plan["max_num_seqs"]
                    and value["max_num_batched_tokens"] == 16384 and value["physical_dp"] == value["ep_world_size"] == 4
                    and value["tp"] == value["pp"] == 1 and value["graphs"] == "NONE" and not value["prefix_caching"]
                    and value["backend"] == "deepep_high_throughput" and value["load_tile"] == 16
                    and value["source_commit"] == COMMIT and value["probe_sha256"] == sha(self.plan["probe"]),
                    "Preflight runtime contract differs")
            require(value["placement_keep"] == ARMS[arm][0] and value["load_penalty"] == ARMS[arm][1], "Preflight arm differs")
            require(value["deployment_proof_sha256"] == sha(self.plan["source_proof"])
                    and Path(value["model_path"]).resolve() == Path(self.plan["model"]).resolve()
                    and value["vllm_version"].split("+")[0] == "0.31.0", "Preflight model/deployment/runtime identity differs")
            require("H100" in value["gpu_name"], "Expected H100 for headline replication")
        return [{"path": str(p), "sha256": sha(p)} for p in paths]

    def launch(self, sweep, arm, position):
        self.validate()
        directory = self.root / f"sweep{sweep:02d}" / f"position{position}-{arm}"
        directory.mkdir(parents=True, exist_ok=False)
        gpu_before = gpu_idle()
        with socket.socket() as sock:
            require(sock.connect_ex(("127.0.0.1", self.plan["port"])) != 0, "Target port occupied before fresh launch")
        probe_dir = directory / "probe" if self.mode == "preflight" else None
        env = sanitized_env(self.plan, arm, probe=probe_dir)
        import_identity = cpu_import_identity(self.plan, env)
        entry = {"complete": False, "sweep": sweep, "position": position, "arm": arm,
                 "launch_id": f"{self.mode}/sweep{sweep:02d}/position{position}-{arm}",
                 "started_unix_ns": time.time_ns(), "directory": str(directory),
                 "server_command": self.plan["server_command"], "server_environment": {k: env[k] for k in ENV_KEYS if k in env},
                 "instrumentation": "Separate preflight routing-identity wrapper" if probe_dir else "None: no probe/profiler/CUDA event hooks",
                 "preflight_scope": "Activation/config/source only; may execute during startup/dummy routing; does not prove live headline shape coverage",
                 "previous_receipt_sha256": self.state["launches"][-1]["receipt_sha256"] if self.state["launches"] else None,
                 "plan_sha256": sha(self.plan_path), "gpu_before": gpu_before,
                 "cpu_target_import_identity": import_identity,
                 "source_before": source_identity(self.plan["source_proof"], self.plan["source_root"], self.plan["site_root"]),
                 "rounds": []}
        log_path = directory / "server.log"
        try:
            with log_path.open("xb") as log:
                self.child = subprocess.Popen(self.plan["server_command"], env=env, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                self.child_identity = process_record(self.child.pid)
                require(self.child_identity and self.child_identity["pgrp"] == self.child.pid, "Cannot bind owned server process")
                entry["server_process"] = self.child_identity
                self.state["active_server"] = {"directory": str(directory), "identity": self.child_identity,
                                                "arm": arm, "sweep": sweep, "position": position}
                self.persist()
                self.wait_health()
                entry["healthy_unix_ns"] = time.time_ns()
                prompts = verify_prompts(self.plan["prompt_file"])
                with cf.ThreadPoolExecutor(max_workers=64) as executor:
                    phases = [("preflight", self.plan["preflight_rounds"])] if probe_dir else [
                        ("warmup", self.plan["warmup_rounds"]), ("measured", self.plan["measured_rounds"])]
                    for phase, count in phases:
                        for number in range(count):
                            require(self.child.poll() is None, "Target server died before round")
                            require(time.time() + 2 * self.plan["request_timeout"] + 30 < self.plan["stop_unix"], "Insufficient compute-cutoff margin")
                            self.state["stage"] = f"{self.mode} sweep{sweep} {arm} {phase}{number}"
                            self.persist()
                            entry["rounds"].append(run_round(self.plan, prompts, executor, directory / f"{phase}{number:02d}", phase, number))
                if probe_dir:
                    entry["probe_files"] = self.verify_probe(directory, arm)
                else:
                    measured = [r["elapsed_seconds"] for r in entry["rounds"] if r["phase"] == "measured"]
                    require(len(measured) == 6, "Expected all6 measured rounds")
                    entry["mean_measured_seconds"] = statistics.mean(measured)
                    entry["total_tokens_per_second"] = 64 * 4097 / entry["mean_measured_seconds"]
                require(self.child.poll() is None, "Server died during target workload")
                entry["complete"] = True
        finally:
            finalization_errors = []
            if self.child is not None:
                try:
                    entry["server_cleanup"] = stop_owned(self.child, self.child_identity)
                    self.child = self.child_identity = None
                    self.state.pop("active_server", None)
                except BaseException:
                    finalization_errors.append({"step": "owned_server_cleanup", "error": traceback.format_exc()})
            entry["finished_unix_ns"] = time.time_ns()
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            warnings = [{"line": i + 1, "text": line} for i, line in enumerate(log.splitlines())
                        if any(t in line.lower() for t in ["warn", "error", "compil", "jit", "placement-router"])]
            active = "placement-router: placement-aware routing active"
            entry["activation_log_contract"] = {
                "no_fallback": "placement-router: routing left unchanged" not in log,
                "activation_matches_arm": (active in log) == (ARMS[arm][0] is not None)}
            entry["complete"] = entry["complete"] and all(entry["activation_log_contract"].values())
            save(directory / "warnings-and-compile.json", warnings, exclusive=True)
            entry["server_log_sha256"] = sha(log_path)
            entry["warnings_file_sha256"] = sha(directory / "warnings-and-compile.json")
            for label, action in [("source_after", lambda: source_identity(self.plan["source_proof"], self.plan["source_root"], self.plan["site_root"])),
                                  ("gpu_after", gpu_idle)]:
                try:
                    entry[label] = action()
                except BaseException:
                    finalization_errors.append({"step": label, "error": traceback.format_exc()})
            entry["finalization_errors"] = finalization_errors
            entry["complete"] = entry["complete"] and not finalization_errors
            receipt = directory / "launch.json"
            save(receipt, entry, exclusive=True)
            self.state["launches"].append({"complete": entry["complete"], "arm": arm, "sweep": sweep,
                                           "position": position, "receipt_file": str(receipt), "receipt_sha256": sha(receipt)})
            self.persist()
        require(entry["complete"], "Launch failed activation or workload contract; do not interpret timing")
        self.validate()

    def run(self):
        try:
            self.validate()
            self.state["hardware"] = capture(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id", "--format=csv"])
            self.state["topology"] = capture(["nvidia-smi", "topo", "-m"])
            if self.mode == "timed":
                self.preflight_gate()
                for sweep in schedule():
                    for position, arm in enumerate(sweep["arms"]):
                        self.launch(sweep["sweep"], arm, position)
            else:
                for position, arm in enumerate(a for a in ARMS if a != "stock_repeat"):
                    self.launch(0, arm, position)
            self.state.update(complete=True, stage="All predeclared launches complete")
        except BaseException:
            self.state.update(error=traceback.format_exc(), stage="Failed; preserve raw results; no silent retry")
            raise
        finally:
            cleanup_errors = []
            if self.child is not None:
                try:
                    self.state["emergency_cleanup"] = stop_owned(self.child, self.child_identity)
                    self.child = self.child_identity = None
                except BaseException:
                    cleanup_errors.append(traceback.format_exc())
            try:
                self.state["final_gpu_idle"] = gpu_idle()
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
            self.state.update(cleanup_complete=not cleanup_errors, cleanup_errors=cleanup_errors,
                              finished_unix_ns=time.time_ns())
            if cleanup_errors:
                self.state["complete"] = False
            self.persist()
            save(self.root / "receipt.json", self.state, exclusive=True)
            require(not cleanup_errors, "Owned cleanup or GPU idle not verified; inspect receipt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    p = commands.add_parser("plan")
    for name in ["output", "model", "checkpoint-proof", "source-proof", "production-root", "site-root", "vllm", "allocation-proof"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-num-seqs", type=int, choices=[128, 256], required=True)
    p.add_argument("--devices", default="0,1,2,3")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--gpu-util", type=float, default=.90)
    p.add_argument("--boot-timeout", type=int, default=2400)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--stop-unix", type=float, required=True)
    for name in ["preflight", "timed"]:
        p = commands.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        make_plan(args)
        print(args.output / "plan.json")
        return
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError("Campaign received " + signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    Campaign(args.plan, args.mode).run()


if __name__ == "__main__":
    main()
