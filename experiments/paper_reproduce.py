#!/usr/bin/env python3
"""Finite paper-style, in-engine MoE timing campaign; no cluster operations.

Use --mode plan, then --mode preflight, then --mode run with identical arguments.
The full protocol is six outer sweeps, each with eight fresh servers. Three native
stock servers bracket five treatments. CUDA-event measurements come from the new
paper_probe.py; HTTP rates are workload diagnostics, never the paper headline.
All raw observations and JIT diagnostics are retained. Only the predeclared exact
T stratum is summarized; model steps, layers and requests are not independent
replicates. Each statistical observation is one outer sweep.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import signal
import socket
import statistics
import subprocess
import time
import traceback
import urllib.error
import urllib.request

import analyze_serving as launch_statistics
import serve_profile as common

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
BASE_COMMIT = "2d8f5192ba195ec1fd3e62a259ce67c43cafa4c3"
ORIGINAL_SOURCE = {
    "src/placement_router/__init__.py": "8425741d4c8d90b5ce2ce122cff4abb8a095b8e488ce9e18ce468a0916f5d0cf",
    "src/placement_router/kernel.py": "46fe57767d6b4230f311bc40f529bf04e1608bab68d627efd5dcf08fc7a75ff2",
    "src/placement_router/plugin.py": "d4e7de6b64885c7b9329e6d2fef39678249328328e4049f64ee76430eef2380e",
    "src/placement_router/vendor/__init__.py": "623bf62f4ae70782a32c1691d3648c120c67c5e4853725d3881265b7dfd96b25",
    "src/placement_router/vendor/fused_topk_router.py": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc"}
PHASES = ("dispatch_ms", "combine_ms", "moe_ms")
ESTIMATORS = ("paper_all_post96", "fixed_warmup_measured_only")
BASES = ("base", "base2", "base3")
ARMS = {"base": (None, None), "base2": (None, None), "base3": (None, None),
        "keep8": (8, 0), "plain6": (6, 0), "balanced6p1": (6, 1),
        "plain4": (4, 0), "balanced4p1": (4, 1)}
TREATMENTS = tuple(name for name in ARMS if name not in BASES)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def schedule(seed=1729):
    """Fixed stock anchors; five Latin rotations, then reverse the first order.

    Six sweeps cannot perfectly balance five treatment positions. The first five
    cover each treatment position once; the sixth is explicitly recorded and no
    claim of full Williams/predecessor balance is made.
    """
    labels = list(TREATMENTS)
    random.Random(seed).shuffle(labels)
    orders = [labels[i:] + labels[:i] for i in range(5)] + [list(reversed(labels))]
    return [{"repeat": i, "arms": ["base", *order[:2], "base2", *order[2:], "base3"]}
            for i, order in enumerate(orders)]


def source_hashes():
    names = ("experiments/paper_reproduce.py", "experiments/paper_probe.py",
             "experiments/serve_profile.py", "experiments/analyze_serving.py", *ORIGINAL_SOURCE)
    return {name: digest(ROOT / name) for name in names}


def require_original_source(identity):
    require(all(identity["source_sha256"].get(k) == v for k, v in ORIGINAL_SOURCE.items()),
            "Deployed production source is not pristine git " + BASE_COMMIT)


def checkpoint_proof(args):
    proof = read(args.checkpoint_proof)
    require(proof.get("canonical_identity_verified") is True, "Canonical checkpoint identity is unverified")
    require(proof.get("model_id") == MODEL_ID, "Expected the canonical Qwen Instruct-2507 checkpoint")
    require(re.fullmatch(r"[0-9a-f]{40}", proof.get("revision", "")), "Checkpoint revision must be a resolved commit SHA")
    require(isinstance(proof.get("local_model_path"), str), "Checkpoint proof has no resolved local_model_path")
    require(Path(proof["local_model_path"]).resolve() == Path(args.model).resolve(),
            "Checkpoint proof does not name the actual local model path")
    if args.revision:
        require(args.revision == proof["revision"], "Requested revision differs from verified checkpoint")
    config = read(Path(args.model) / "config.json")
    require(digest(Path(args.model) / "config.json") == proof.get("canonical_config_sha256"),
            "Local config differs from canonical checkpoint")
    require(config.get("num_experts") == 128 and config.get("num_experts_per_tok") == 8,
            "Expected E128, top-k8")
    require(config.get("num_hidden_layers") == 48 and config.get("hidden_size") == 2048,
            "Expected 48 Qwen layers and hidden size2048")
    require(config.get("vocab_size", 0) >= 100000, "Original token IDs exceed this vocabulary")
    files = proof.get("safetensors", [])
    require(len(files) == 16, "Checkpoint proof must list all16 canonical weight files")
    signatures = {}
    for file in files:
        path = Path(args.model) / file["filename"]
        require(path.is_file() and path.stat().st_size == file["bytes"], "Missing/truncated checkpoint weight: " + file["filename"])
        signatures[file["filename"]] = [path.stat().st_size, path.stat().st_mtime_ns]
    cache_path = args.output / "checkpoint-local-verification.json"
    cache = read(cache_path) if cache_path.exists() else {}
    if cache.get("proof_sha256") != digest(args.checkpoint_proof) or cache.get("file_stat_signatures") != signatures:
        for file in files:
            require(digest(Path(args.model) / file["filename"]) == file["sha256"],
                    "Local weights differ from canonical checkpoint: " + file["filename"])
            print(f"[checkpoint] verified {file['filename']}", flush=True)
        common.atomic_json(cache_path, {"complete": True, "proof_sha256": digest(args.checkpoint_proof),
            "file_stat_signatures": signatures, "verified_weight_files": files,
            "verified_unix": time.time(), "scope": "All16localweightSHA256andconfigSHAverified; reuseonlywhileproofandfile size/mtime remain unchanged"})
    return {"proof": proof, "proof_sha256": digest(args.checkpoint_proof),
            "config_sha256": digest(Path(args.model) / "config.json"),
            "canonical_local_weight_files_verified": files,
            "paper_checkpoint_identity_verified": proof.get("paper_checkpoint_identity_verified") is True}


def shape(args):
    return common.Shape("paper_prefill", 4096, 1, 128, 256)


def configuration(args):
    return {"model": args.model, "canonical_model_id": MODEL_ID, "revision": args.revision,
            "tag": args.tag, "port": args.port, "vllm": args.vllm,
            "request_timeout": args.request_timeout, "boot_timeout": args.boot_timeout,
            "drain_timeout": args.drain_timeout, "dp": args.dp, "tp": 1,
            "ep": args.dp, "backend": "deepep_high_throughput", "dtype": "bfloat16",
            "graphs": "NONE", "expert_placement": "linear", "selector": "baseline",
            "sweeps": 6, "warmup_rounds": 3, "measured_rounds": 3,
            "target_tokens": 16384, "expected_layers": 48,
            "minimum_complete_steps_per_rank_per_measured_round": 4,
            "shape": dataclasses.asdict(shape(args)), "batch_tokens": 16384,
            "max_num_seqs": 128, "max_model_len": 4608, "gpu_util": args.gpu_util,
            "prompt_seed": 0, "prompt_id_range": [1, 99999],
            "schedule_seed": args.schedule_seed, "expected_gpu_name": args.expected_gpu_name,
            "timer_warmup_forwards": 96, "timer_flush_forwards": 64,
            "timer_flush_mode": "original_end_synchronize_in_span_insertion_order",
            "probe_drain": "one_final_completed_query_drain_after_all_six_HTTP_rounds"}


def manifest_identity(args):
    make_prompts(args)
    return {"config": configuration(args), "schedule": schedule(args.schedule_seed),
            "arm_definitions": {a: {"keep": k, "penalty": p, "selector": "baseline"}
                                for a, (k, p) in ARMS.items()},
            "source_sha256": source_hashes(), "checkpoint": checkpoint_proof(args),
            "overlay_source_sha256": {k: v for k, v in source_hashes().items() if k.startswith("experiments/")},
            "prompt_file_sha256": digest(args.output / "prompts.json"),
            "prompt_metadata_sha256": digest(args.output / "prompts_metadata.json"),
            "production_source_base_commit": BASE_COMMIT,
            "production_source_expected_sha256": ORIGINAL_SOURCE,
            "order_balance": "Fixed stock anchors first/middle/last; complete five-position Latin coverage in sweeps0–4, sixth reversed. Partial order balance, not a full Williams design."}


def ensure_manifest(args):
    identity = manifest_identity(args)
    path = args.output / "manifest.json"
    if path.exists():
        require(read(path)["identity"] == identity, "Experiment identity changed; choose a fresh output directory")
    else:
        common.atomic_json(path, {"identity": identity, "metadata": common.metadata(),
            "protocol": "Six predeclared outer sweeps; eight fresh servers per sweep; stock base/base2/base3 bracket five original-selector treatments. Three fixed full-size warmups plus three measured HTTP rounds. Original96-forward warmup and64-forward synchronization cadence; preserve all post96 timing spans and all JIT warnings. ExactT16384 summaries require all physical ranks and48layers. No slow-result filtering, no statistical independence assigned to layers/steps/requests."})
    return identity


def make_prompts(args):
    path = args.output / "prompts.json"
    if path.exists():
        values = read(path)
        require(len(values) == 256 and all(len(p["token_ids"]) == 4096 for p in values), "Stored prompt shape differs")
        require(read(args.output / "prompts_metadata.json").get("prompt_sha256") == digest(path),
                "Stored prompt hash differs; never change inputs within a campaign")
        return values
    import numpy as np
    ids = np.random.default_rng(0).integers(1, 100000, size=(256, 4096))
    values = [{"request_id": i, "token_ids": row.tolist()} for i, row in enumerate(ids)]
    common.atomic_json(path, values)
    common.atomic_json(args.output / "prompts_metadata.json", {
        "generator": "numpy.random.default_rng(0).integers(1,100000,size=(256,4096))",
        "numpy": np.__version__, "prompt_sha256": digest(path),
        "token_ids_little_endian_int64_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
        "text_corpus": None, "same_prompts_every_arm_and_round": True})
    return values


def server_command(args):
    return [args.vllm, "serve", args.model, "--host", "127.0.0.1", "--port", str(args.port),
            "--data-parallel-size", str(args.dp), "--tensor-parallel-size", "1",
            "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput",
            "--dtype", "bfloat16", "--expert-placement-strategy", "linear",
            "--compilation-config={\"cudagraph_mode\":\"NONE\"}",
            "--max-model-len", "4608", "--seed", "0", "--max-num-batched-tokens", "16384",
            "--max-num-seqs", "128", "--no-enable-prefix-caching", "--disable-log-stats", "--jit-monitor-verbose",
            "--gpu-memory-utilization", str(args.gpu_util)]


def arm_environment(args, arm, repeat, directory):
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith("PAPER_PROBE_") or key.startswith("PLACEMENT_")
                or key.startswith("VLLM_TORCH_PROFILER_") or key == "VLLM_DEEPEP_BUFFER_SIZE_MB"):
            env.pop(key, None)
    keep, penalty = ARMS[arm]
    if keep is not None:
        env.update(PLACEMENT_KEEP=str(keep), PLACEMENT_LOAD_PENALTY=str(penalty))
    env.update(PLACEMENT_PROBE=str(ROOT / "experiments/paper_probe.py"),
        PAPER_PROBE_DIR=str(directory / "probe"), PAPER_PROBE_CONTROL=str(directory / "probe-control.json"),
        PAPER_PROBE_ARM=arm, PAPER_PROBE_REPEAT=str(repeat), PAPER_PROBE_TAG=args.tag,
        PAPER_PROBE_EXPECTED_EP=str(args.dp), PAPER_PROBE_EXPECTED_LAYERS="48", PAPER_PROBE_TOKENS="16384",
        PAPER_PROBE_WARMUP_FORWARDS="96", PAPER_PROBE_FLUSH_FORWARDS="64",
        OMP_NUM_THREADS="1", NVSHMEM_QP_DEPTH="4096", VLLM_LOGGING_LEVEL="INFO")
    return env


def control(directory, phase, epoch, round_number):
    value = {"phase": phase, "epoch": epoch, "round": round_number}
    common.atomic_json(directory / "probe-control.json", value)
    return value


def rank_of(snapshot):
    meta = snapshot.get("metadata", {})
    return meta.get("ep_rank", meta.get("rank"))


def probe_files(directory):
    return sorted((directory / "probe").glob("rank*_pid*.json"))


def snapshots(directory):
    return [read(path) for path in probe_files(directory)]


def wait_ack(args, directory, wanted, *, drain=False, process=None):
    deadline = time.monotonic() + args.drain_timeout
    last = "No rank snapshots"
    while time.monotonic() < deadline:
        if process is not None:
            require(process.poll() is None, "Server died while awaiting probe acknowledgement")
        try:
            rows = snapshots(directory)
            require(len(rows) == args.dp and {rank_of(r) for r in rows} == set(range(args.dp)), "Missing/duplicate probe rank")
            require(all(not row.get("errors") for row in rows), "Probe reports an error")
            require(all(row.get("ready") is True for row in rows), "Probe runtime is not ready on every rank")
            require(all(all(row.get("control_ack", {}).get(k) == v for k, v in wanted.items()) for row in rows), "Workers have not all acknowledged control epoch")
            if drain:
                require(all(row.get("drained") is True and row.get("pending_spans") == 0
                            and row.get("open_forwards") == 0 and row.get("flush_in_progress") is False
                            and row.get("records_included") is True
                            for row in rows), "Probe has pending or open spans")
            return rows
        except (RuntimeError, OSError, ValueError) as error:
            last = str(error)
            time.sleep(0.1)
    raise TimeoutError(f"Probe acknowledgement failed: {last}")


def check_log(args, arm, path):
    text = path.read_text(errors="replace")
    modes = sorted(set(re.findall(r"['\"]cudagraph_mode['\"]:\s*<CUDAGraphMode\.([A-Z_]+):", text)))
    require(modes == ["NONE"], "Actual graph mode is not verified NONE")
    require("routing left unchanged -" not in text, "Placement fell back to native routing")
    keep, _ = ARMS[arm]
    if keep is None:
        require("placement-aware routing active" not in text, "Stock arm unexpectedly changed routing")
    else:
        require(text.count(f"placement-aware routing active, keep={keep} of top_k=8, ep_size={args.dp}") >= args.dp,
                "Missing all-rank placement activation evidence")
    selectors = set(re.findall(r"experimental selector=([A-Za-z0-9_]+)", text))
    require(selectors <= {"baseline"}, "An experimental selector was used")
    return {"observed_cudagraph_modes": modes, "observed_selectors": sorted(selectors), "fallback_observed": False}


@contextlib.contextmanager
def serve(args, arm, repeat, directory, identity):
    require(source_hashes() == identity["source_sha256"], "Frozen benchmark source changed")
    require_original_source(identity)
    require(digest(args.output / "prompts.json") == identity["prompt_file_sha256"]
            and digest(args.output / "prompts_metadata.json") == identity["prompt_metadata_sha256"],
            "Frozen prompt identity changed")
    env = arm_environment(args, arm, repeat, directory)
    command = server_command(args)
    with socket.socket() as sock:
        require(sock.connect_ex(("127.0.0.1", args.port)) != 0, "Server port already occupied")
    common.atomic_json(directory / "server.json", {"command": command, "shell_command": shlex.join(command),
        "arm": arm, "repeat": repeat, "placement_environment": {k: env.get(k) for k in
            ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY", "PLACEMENT_EXPERIMENTAL_SELECTOR", "PLACEMENT_PROBE")},
        "probe_environment": {k: v for k, v in env.items() if k.startswith("PAPER_PROBE_")},
        "controlled_environment": {k: env.get(k) for k in ("OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH", "VLLM_LOGGING_LEVEL")},
        "profiler_active": False, "cuda_event_probe_active": True})
    with (directory / "server.log").open("w", buffering=1) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        common.atomic_json(directory / "pid.json", {"pid": process.pid})
        try:
            deadline = time.monotonic() + args.boot_timeout
            while time.monotonic() < deadline:
                require(process.poll() is None, "Server exited before healthy")
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


def jit_workload(args, prompts, directory):
    path = directory / "server.log"
    start = path.stat().st_size
    data = common.run_workload(args, shape(args), prompts)
    end = path.stat().st_size
    with path.open("rb") as source:
        source.seek(start)
        text = source.read(end - start).decode(errors="replace")
    data.update(server_log_byte_offset_start=start, server_log_byte_offset_end=end,
        jit_events_observed_in_log_slice=[line for line in text.splitlines()
            if "jit_monitor.py" in line and ("compilation during inference" in line or "autotuning during inference" in line)])
    return data


def validate_snapshot(rows, config, *, expected_arm=None, expected_repeat=None, expected_sources=None):
    dp, layers, target = config["dp"], config["expected_layers"], config["target_tokens"]
    require(len(rows) == dp and {rank_of(r) for r in rows} == set(range(dp)), "Expected one probe file per actual EP rank")
    require(len({r.get("metadata", {}).get("pid") for r in rows}) == dp, "Worker PIDs are not distinct")
    coverage, timing = [], {e: {} for e in ESTIMATORS}
    for row in rows:
        rank, meta = rank_of(row), row.get("metadata", {})
        require(not row.get("errors"), f"Probe errors on rank{rank}: {row.get('errors')}")
        require(row.get("ready") is True and row.get("drained") is True
                and row.get("pending_spans") == row.get("open_forwards") == 0
                and row.get("flush_in_progress") is False,
                "Probe not fully drained")
        require(row.get("records_included") is True, "Final probe snapshot omits raw forward records")
        require(row.get("control_ack") == {"phase": "drain", "epoch": 7, "round": 3},
                "Final probe control acknowledgement differs")
        require(meta.get("ep_world_size", meta.get("world_size")) == dp, "Probe actual EP world differs")
        require(meta.get("dp_rank") == rank, "DP/EP rank mapping differs")
        require(config["expected_gpu_name"] in str(meta.get("device_name", meta.get("gpu", ""))), "Unexpected GPU hardware")
        require(meta.get("backend") == "deepep_high_throughput" and meta.get("cudagraph_mode") == "NONE",
                "Probe backend/graph contract differs")
        require(meta.get("activation_dtype") == "torch.bfloat16" and meta.get("hidden_size") == 2048,
                "Probe model dtype/hidden size differs")
        require(meta.get("selector") == "baseline" and meta.get("experimental_impl_is_none") is True,
                "Probe observed a different selector")
        require(meta.get("kernel_source_sha256") == ORIGINAL_SOURCE["src/placement_router/kernel.py"]
                and meta.get("pristine_source_commit") == BASE_COMMIT, "Actual imported kernel source differs")
        if expected_sources is not None:
            require(meta.get("probe_sha256") == expected_sources["experiments/paper_probe.py"], "Actual imported probe source differs")
        require(len(meta.get("registered_MoE_layer_names", [])) == layers, "Missing registered layer coverage")
        semantics = meta.get("timing_semantics", {})
        require(semantics.get("warmup_global_forwards") == 96 and semantics.get("flush_whole_forwards") == 64,
                "Original timer warmup/cadence differs")
        if expected_arm is not None:
            require(meta.get("arm") == expected_arm, "Probe arm identity differs")
            keep, penalty = ARMS[expected_arm]
            require(meta.get("placement_keep") == (str(keep) if keep is not None else None)
                    and meta.get("load_penalty") == (str(penalty) if penalty is not None else None),
                    "Actual placement keep/penalty differs")
        if expected_repeat is not None:
            require(meta.get("repeat") == str(expected_repeat), "Probe outer-sweep identity differs")
        require(meta.get("model") == config["model"], "Actual loaded model differs from verified checkpoint")
        records = row.get("forward_records", [])
        require(records and all(r.get("valid") is True and not r.get("errors") for r in records), "Missing/invalid timing records")
        require(len({r["forward_id"] for r in records}) == len(records), "Duplicate forward IDs")
        for record in records:
            require(record["phase"] in ("off", "warmup", "measure"), "Unexpected recorded control phase")
            require(record.get("seen_forward", 0) > 96 and record.get("flush_origin") in ("chunk64", "drain_tail"),
                    "Unexpected original warmup/flush metadata")
            require(record.get("dispatch_input_T") == record["T"], "Dispatch input T differs from timed MoE T")
            require(all(record.get("span_calls", {}).get(p) == record.get("span_ends", {}).get(p) == 1
                        for p in ("moe", "dispatch", "combine")), "Missing or repeated timing receiver span")
            require(0 <= record["layer_index"] < layers and isinstance(record["T"], int), "Invalid layer/token metadata")
            require(all(isinstance(record.get(ph), (int, float)) and math.isfinite(record[ph]) and record[ph] > 0 for ph in PHASES),
                    "Missing/nonpositive dispatch, combine or MoE timing")
        exact = [r for r in records if r["T"] == target]
        for round_number in range(1, config["measured_rounds"] + 1):
            selected = [r for r in exact if r["phase"] == "measure" and r["round"] == round_number]
            by_step = collections.defaultdict(list)
            for record in selected:
                by_step[(record["epoch"], record["step_id"])].append(record)
            require(by_step, f"No exactT measured coverage rank{rank}/round{round_number}")
            for step in by_step.values():
                require(len(step) == layers and {r["layer_index"] for r in step} == set(range(layers)),
                        f"Partial/duplicate exactT model step on rank{rank}/round{round_number}")
            require(len(by_step) >= config["minimum_complete_steps_per_rank_per_measured_round"],
                    f"Insufficient complete exactT steps rank{rank}/round{round_number}")
            coverage.append({"rank": rank, "round": round_number, "complete_exact_steps": len(by_step),
                             "exact_forwards": len(selected), "layers": layers})
        for estimator in ESTIMATORS:
            selected = exact if estimator == "paper_all_post96" else [r for r in exact if r["phase"] == "measure"]
            require(selected and {r["layer_index"] for r in selected} == set(range(layers)), "Missing all-layer exactT coverage")
            timing[estimator][rank] = {"forwards": len(selected), **{ph: statistics.median(r[ph] for r in selected) for ph in PHASES}}
    return {"coverage": coverage, "rank_medians": {e: {str(r): v for r, v in by_rank.items()}
                                                    for e, by_rank in timing.items()},
            "runtime_metadata": [r["metadata"] for r in sorted(rows, key=rank_of)],
            "mean_rank_medians": {e: {ph: statistics.mean(timing[e][r][ph] for r in range(dp)) for ph in PHASES}
                                   for e in ESTIMATORS}}


def validate_launch(path, identity):
    data = read(path)
    require(data.get("complete") is True, "Incomplete launch")
    require(data.get("source_sha256") == identity["source_sha256"], "Frozen source identity differs")
    for entry in data["probe_files"]:
        file = path.parent / entry["path"]
        require(file.stat().st_size == entry["bytes"] and digest(file) == entry["sha256"], "Probe snapshot hash differs")
    timing = validate_snapshot(snapshots(path.parent), identity["config"], expected_arm=data["arm"],
                               expected_repeat=data["repeat"],
                               expected_sources=identity["source_sha256"])
    require(timing == data["timing_summary"], "Stored timing summary differs from raw records")
    validate_http_rounds(data, identity["config"])
    return data, timing


def validate_http_rounds(data, config):
    require(len(data["warmups"]) == len(data["rounds"]) == 3, "Missing fixed HTTP workloads")
    for rows in (data["warmups"], data["rounds"]):
        require(all(not launch_statistics.round_errors(row, config["shape"], i)
                    for i, row in enumerate(rows)), "HTTP workload/token budget incomplete")


def measure_launch(args, arm, repeat, directory, identity, prompts):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists():
        old, _ = validate_launch(path, identity)
        return old
    data = {"schema_version": 1, "arm": arm, "repeat": repeat, "complete": False,
            "measurement_type": "in_engine_cuda_events_original_cadence",
            "source_sha256": identity["source_sha256"], "warmups": [], "rounds": []}
    common.atomic_json(path, data)
    try:
        control(directory, "off", 0, 0)
        with serve(args, arm, repeat, directory, identity) as process:
            epoch = 0
            for phase, output in (("warmup", data["warmups"]), ("measure", data["rounds"])):
                for round_number in range(1, 4):
                    epoch += 1
                    wanted = control(directory, phase, epoch, round_number)
                    # Engine startup dummy forwards initialize every real worker.
                    # Refuse an unverified probe before issuing any HTTP workload.
                    wait_ack(args, directory, wanted, process=process)
                    value = jit_workload(args, prompts, directory)
                    output.append({"probe_control": wanted, **value})
                    common.atomic_json(path, data)
                    wait_ack(args, directory, wanted, process=process)
                    print(f"[{directory.name}] {phase} {round_number}/3 complete; no result filtering", flush=True)
            wanted = control(directory, "drain", epoch + 1, 3)
            wait_ack(args, directory, wanted, drain=True, process=process)
            data["runtime_evidence"] = check_log(args, arm, directory / "server.log")
        # The server is stopped before final snapshot hashes are recorded.
        rows = snapshots(directory)
        data["timing_summary"] = validate_snapshot(rows, identity["config"], expected_arm=arm,
                                                    expected_repeat=repeat,
                                                    expected_sources=identity["source_sha256"])
        data["probe_files"] = [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size,
                                "sha256": digest(p)} for p in probe_files(directory)]
        validate_http_rounds(data, identity["config"])
        data["complete"] = True
    except BaseException as error:
        data.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        common.atomic_json(path, data)
    return data


def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = d if abs(d) > 1e-300 else 1e-300
            c = 1 + aa / c
            c = c if abs(c) > 1e-300 else 1e-300
            d = 1 / d
            delta = d * c
            h *= delta
        if abs(delta - 1) < 3e-14:
            return h
    raise RuntimeError("Student-t incomplete-beta evaluation did not converge")


def two_sided_t_pvalue(t_value, df):
    if math.isinf(t_value):
        return 0.0
    if t_value == 0:
        return 1.0
    x, a, b = df / (df + t_value * t_value), df / 2, 0.5
    if x >= 1:
        return 1.0
    if x <= 0:
        return 0.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    result = factor * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - factor * _betacf(b, a, 1 - x) / b
    return min(1.0, max(0.0, result))


def sample_summary(values):
    ci = launch_statistics.mean_ci(values)
    n, mean = len(values), ci["mean"]
    if not n:
        return {**ci, "se": None, "two_sided_t_pvalue": None,
                "t_statistic": None, "zero_se_nonzero_mean": False}
    se = ci["sd"] / math.sqrt(n) if n > 1 else None
    t_value = mean / se if se else (0.0 if mean == 0 else math.copysign(math.inf, mean))
    return {**ci, "se": se, "two_sided_t_pvalue": two_sided_t_pvalue(t_value, n - 1) if n > 1 else None,
            "t_statistic": t_value if math.isfinite(t_value) else None,
            "zero_se_nonzero_mean": se == 0 and mean != 0}


def effect_summary(ratios):
    gains = [100 * (r - 1) for r in ratios]
    logs = launch_statistics.mean_ci([math.log(r) for r in ratios])
    return {"paper_arithmetic_speedup_pct": sample_summary(gains),
            "geometric_speedup_pct": 100 * (math.exp(logs["mean"]) - 1) if ratios else None,
            "geometric_speedup_ci95_pct": [100 * (math.exp(x) - 1) for x in logs["ci95"]] if logs["ci95"] else None,
            "ratios": ratios, "independent_unit": "outer sweep; not rank/layer/step/request"}


def analyze(output):
    identity = read(output / "manifest.json")["identity"]
    require_original_source(identity)
    expected = {(row["repeat"], arm) for row in identity["schedule"] for arm in row["arms"]}
    indexed, issues, launches = {}, [], []
    for path in sorted(output.glob("r*_*/result.json")):
        try:
            data, timing = validate_launch(path, identity)
            key = data["repeat"], data["arm"]
            require(key in expected and key not in indexed, "Unexpected/duplicate launch")
            jit = [{"phase": phase, "round": i + 1, "warnings": row["jit_events_observed_in_log_slice"]}
                   for phase, rows in (("warmup", data["warmups"]), ("measure", data["rounds"]))
                   for i, row in enumerate(rows) if row["jit_events_observed_in_log_slice"]]
            row = {"arm": key[1], "repeat": key[0], "directory": str(path.parent),
                   "timing_summary": timing, "jit_events_retained": jit,
                   "warmup_HTTP_rounds": len(data["warmups"]), "measured_HTTP_rounds": len(data["rounds"]),
                   "warmup_requests": sum(r["requests"] for r in data["warmups"]),
                   "measured_requests": sum(r["requests"] for r in data["rounds"])}
            indexed[key] = row
            launches.append(row)
        except (RuntimeError, KeyError, OSError, ValueError) as error:
            issues.append(f"{path.parent.name}: {error}")
    issues += [f"Missing valid planned launch r{r:02d}_{a}" for r, a in sorted(expected - indexed.keys())]
    contracts = [meta["timed_methods"] for row in launches for meta in row["timing_summary"]["runtime_metadata"]]
    if contracts and any(value != contracts[0] for value in contracts):
        issues.append("Actual timed vLLM method sources/signatures differ between ranks or launches")
    comparisons, load_term, floors = [], [], {}
    for estimator in ESTIMATORS:
        floors[estimator] = {}
        for phase in PHASES:
            baselines, spreads = {}, []
            for repeat in range(6):
                if all((repeat, b) in indexed for b in BASES):
                    times = [indexed[(repeat, b)]["timing_summary"]["mean_rank_medians"][estimator][phase] for b in BASES]
                    baselines[repeat] = statistics.mean(times)
                    spreads.append({"repeat": repeat, "base_ms": dict(zip(BASES, times)),
                                    "spread_pct": 100 * (max(times) - min(times)) / min(times)})
            floor = max((r["spread_pct"] for r in spreads), default=math.nan) / 2
            floors[estimator][phase] = {"half_widest_base_spread_pct": floor if spreads else None,
                "baseline_spreads": spreads, "baseline_mean_ms_by_sweep": baselines,
                "baseline_mean_ms_across_sweeps": statistics.mean(baselines.values()) if baselines else None}
            for arm in TREATMENTS:
                pairs = [{"repeat": repeat, "baseline_mean_ms": base,
                          "arm_ms": indexed[(repeat, arm)]["timing_summary"]["mean_rank_medians"][estimator][phase]}
                         for repeat, base in baselines.items() if (repeat, arm) in indexed]
                ratios = [row["baseline_mean_ms"] / row["arm_ms"] for row in pairs]
                comparisons.append({"estimator": estimator, "phase": phase, "arm": arm,
                    "baseline": "mean(base,base2,base3) within same outer sweep", "pairs": pairs,
                    "complete_pairs": len(pairs) == 6, "floor_pct": floor if spreads else None,
                    **effect_summary(ratios)})
            for plain, balanced in (("plain6", "balanced6p1"), ("plain4", "balanced4p1")):
                pairs = [{"repeat": repeat,
                          "plain_ms": indexed[(repeat, plain)]["timing_summary"]["mean_rank_medians"][estimator][phase],
                          "balanced_ms": indexed[(repeat, balanced)]["timing_summary"]["mean_rank_medians"][estimator][phase]}
                         for repeat in range(6) if (repeat, plain) in indexed and (repeat, balanced) in indexed]
                load_term.append({"estimator": estimator, "phase": phase, "plain": plain, "balanced": balanced,
                                  "pairs": pairs, "complete_pairs": len(pairs) == 6,
                                  **effect_summary([p["plain_ms"] / p["balanced_ms"] for p in pairs])})
    return {"complete": not issues and len(indexed) == 48, "issues": issues,
            "counts": {"expected_fresh_servers": 48, "valid_fresh_servers": len(indexed),
                       "outer_sweeps": 6, "expected_warmup_HTTP_rounds": 144,
                       "expected_measured_HTTP_rounds": 144,
                       "actual_warmup_HTTP_rounds": sum(r["warmup_HTTP_rounds"] for r in launches),
                       "actual_measured_HTTP_rounds": sum(r["measured_HTTP_rounds"] for r in launches),
                       "warmup_requests": sum(r["warmup_requests"] for r in launches),
                       "measured_requests": sum(r["measured_requests"] for r in launches)},
            "scope": identity, "launches": launches, "noise_floors": floors,
            "comparisons": comparisons, "load_term_comparisons": load_term,
            "method": {"primary": "paper_all_post96: all exactT16384 spans after the original96-forward timer warmup, including startup/off and explicitly labeled HTTPwarmup rounds; final completed tail retained",
                "secondary": "fixed_warmup_measured_only: exactT16384 spans from the three predeclared measured rounds after three full HTTPwarmups",
                "rank_estimator": "Median of exactT forward events per physical rank, then arithmetic mean across every rank",
                "paper_effect": "Within outer sweep, mean of three stock-server estimates divided by treatment estimate minusone; arithmetic mean/SD across six outer sweeps",
                "floor": "Half of the widest within-sweep stock spread: max_sweep((max(base)-min(base))/min(base)*100)/2",
                "load_term": "plain/balanced -1 at same keep within outer sweep; mean, sampleSE and two-sided one-sample t-test across sweeps",
                "additional_intervals": "95% Student-t intervals for arithmetic gains and log-ratios at the six-sweep unit; nominal pointwise, not multiplicity-adjusted",
                "retention": "All raw T strata and all JIT/slow rounds retained. ExactT is a predeclared measurement stratum, not a speed-dependent rejection."}}


def write_analysis(output):
    data = analyze(output)
    common.atomic_json(output / "paper_analysis.json", data)
    proof = data["scope"].get("checkpoint", {}).get("proof", {})
    lines = ["Paper-style MoE timing reproduction", "", f"Complete: {data['complete']}; {data['counts']}", "",
        f"Canonical checkpoint: {proof.get('model_id', 'unspecified')} at {proof.get('revision', 'unspecified')}. Historical paper checkpoint identity verified: {proof.get('paper_checkpoint_identity_verified') is True}.", "",
        "Primary uses original96-forward warmup/64-forward CUDA-event reduction cadence. ExactT=16384 only; median/rank then mean/ranks; six independent outer sweeps.", "",
        "| Estimator | Phase | Arm | Sweeps | Mean speedup ± SD | 95% CI for mean speedup | Own noise floor |",
        "|---|---|---|---:|---:|---:|---:|"]
    for c in data["comparisons"]:
        s = c["paper_arithmetic_speedup_pct"]
        mean = f"{s['mean']:+.3f}% ± {s['sd']:.3f}%" if s["sd"] is not None else "unavailable"
        ci = f"[{s['ci95'][0]:+.3f}%, {s['ci95'][1]:+.3f}%]" if s["ci95"] else "unavailable"
        floor = f"±{c['floor_pct']:.3f}%" if c["floor_pct"] is not None else "unavailable"
        lines.append(f"|{c['estimator']}|{c['phase']}|{c['arm']}|{s['n']}|{mean}|{ci}|{floor}|")
    lines += ["", "Load term at fixed keep, paired within each outer sweep:", "",
              "| Estimator | Phase | Balanced / plain | Sweeps | Mean speedup ± SE | Two-sided t-test p |",
              "|---|---|---|---:|---:|---:|"]
    for c in data["load_term_comparisons"]:
        s = c["paper_arithmetic_speedup_pct"]
        value = f"{s['mean']:+.3f}% ± {s['se']:.3f}%" if s["se"] is not None else "unavailable"
        pvalue = f"{s['two_sided_t_pvalue']:.6g}" if s["two_sided_t_pvalue"] is not None else "unavailable"
        lines.append(f"|{c['estimator']}|{c['phase']}|{c['balanced']}/{c['plain']}|{s['n']}|{value}|{pvalue}|")
    lines += ["", "Partial treatment-order balance is recorded; fixed stock anchors are not randomized. No claim of exact historical checkpoint identity is made without its proof flag.",
              "CUDA event spans include the original timer's synchronization cadence. They are MoE/collective timings, not clean HTTP performance or critical-rank latency."]
    if data["issues"]:
        lines += ["", "Completion issues:"] + ["- " + item for item in data["issues"]]
    (output / "paper_analysis.md").write_text("\n".join(lines) + "\n")
    with (output / "paper_comparisons.csv").open("w", newline="") as target:
        fields = ["estimator", "phase", "arm", "n", "mean_speedup_pct", "sd_pct", "se_pct", "ci95_low_pct", "ci95_high_pct", "floor_pct"]
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for c in data["comparisons"]:
            s = c["paper_arithmetic_speedup_pct"]
            writer.writerow({"estimator": c["estimator"], "phase": c["phase"], "arm": c["arm"],
                "n": s["n"], "mean_speedup_pct": s["mean"], "sd_pct": s["sd"], "se_pct": s["se"],
                "ci95_low_pct": s["ci95"][0] if s["ci95"] else None,
                "ci95_high_pct": s["ci95"][1] if s["ci95"] else None, "floor_pct": c["floor_pct"]})
    return data


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("plan", "preflight", "run", "analyze"), default="plan")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model")
    p.add_argument("--checkpoint-proof", type=Path)
    p.add_argument("--revision")
    p.add_argument("--dp", type=int, choices=(4, 8), default=4)
    p.add_argument("--tag", default="qwen-instruct2507")
    p.add_argument("--schedule-seed", type=int, default=1729)
    p.add_argument("--expected-gpu-name", default="H100")
    p.add_argument("--gpu-util", type=float, default=0.88)
    p.add_argument("--vllm", default="vllm")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--boot-timeout", type=float, default=2400)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--drain-timeout", type=float, default=120)
    args = p.parse_args(argv)
    args.output = args.output.resolve()
    args.url = f"http://127.0.0.1:{args.port}"
    if args.mode in ("preflight", "run"):
        p.error("--model and --checkpoint-proof are required") if not args.model or not args.checkpoint_proof else None
        args.checkpoint_proof = args.checkpoint_proof.resolve()
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == "analyze":
        report = write_analysis(args.output)
        return 0 if report["complete"] else 2
    if args.mode == "plan":
        print(json.dumps({"configuration": configuration(args), "schedule": schedule(args.schedule_seed),
                          "planned_fresh_servers": 48, "preflight_required": True,
                          "production_source_base_commit": BASE_COMMIT,
                          "production_source_expected_sha256": ORIGINAL_SOURCE,
                          "checkpoint_proof_required_before_preflight": True}, indent=2))
        return 0
    identity = ensure_manifest(args)
    require_original_source(identity)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Stopped by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    prompts = make_prompts(args)
    preflight = args.output / "preflight" / "base"
    if args.mode == "preflight":
        data = measure_launch(args, "base", -1, preflight, identity, prompts)
        common.atomic_json(args.output / "preflight-readiness.json", {"complete": True,
            "hostname": socket.gethostname(),
            "manifest_identity_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "preflight_result_sha256": digest(preflight / "result.json"), "timing_summary": data["timing_summary"]})
        print("Preflight passed exactT/all-rank/all-layer coverage. Full campaign remains unstarted.", flush=True)
        return 0
    proof = read(args.output / "preflight-readiness.json")
    require(proof.get("complete") and proof["manifest_identity_sha256"] == hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            and proof["preflight_result_sha256"] == digest(preflight / "result.json")
            and proof.get("hostname") == socket.gethostname(), "Matching same-host preflight readiness is required before any full campaign launch")
    validate_launch(preflight / "result.json", identity)
    for sweep in identity["schedule"]:
        for arm in sweep["arms"]:
            measure_launch(args, arm, sweep["repeat"], args.output / f"r{sweep['repeat']:02d}_{arm}", identity, prompts)
    report = write_analysis(args.output)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
