#!/usr/bin/env python3
"""Bounded fresh causal campaign: six training cells, freeze, three held-out cells.

CPU freeze is separate from run. Nothing resumes or combines the old partial
campaign. Every failed attempt and cleanup result is retained without retry.
"""
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common_v2 import file_sha, read, require, write_new
from fresh_owner import fresh_owner_gate, selected_gate

TOOLS = Path("/work/hybrid-tools/causal-v3")
OUTPUT = Path("/work/results/causal-v3")
OUTER = Path("/work/results/causal-outer-v3")
OWNER = Path("/work/results/provenance/causal-fresh-owner-v1.json")
NATIVE = Path("/work/results/provenance/native-runtime19.json")
QUALITY = Path("/work/results/provenance/causal-prior/prior-quality9.json")
SOURCE = Path("/work/results/provenance/task-source-deployment.json")
ALLOCATION = Path("/work/results/allocation-proof.json")
CHECKPOINT = Path("/work/results/checkpoint-proof.json")
PREFLIGHT = Path("/work/results/quality-prepared-v1/preflight.json")
FULL = Path("/work/results/quality-prepared-v1/requests.json")
LIFECYCLE = Path("/work/followup/performance/prefill_e2e.py")
LIFECYCLE_SHA = "2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3"
FROZEN_OUTER_SHA = "bf480d27b7d06eecbcf0f1c90bbe9463db081e49e5dbe58de3b2e213ecf15146"
MODEL = "/work/hf/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"


def load(path, name, expected):
    require(file_sha(path) == expected, "Frozen lifetime helper source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def commands(stop):
    helper = [sys.executable, "-u", str(TOOLS / "causal_driver_v3.py")]
    plan = helper + ["plan", "--output", str(OUTPUT), "--model", MODEL,
        "--production-root", "/work/task-router", "--site-root", "/work/task-site",
        "--source-proof", str(SOURCE), "--checkpoint-proof", str(CHECKPOINT),
        "--allocation-proof", str(ALLOCATION), "--lifecycle-helper", str(LIFECYCLE),
        "--vllm", "/usr/local/bin/vllm", "--preflight-inputs", "/work/results/quality-prepared-v1/preflight.json",
        "--full-inputs", "/work/results/quality-prepared-v1/requests.json", "--quality-receipt", str(QUALITY),
        "--heldout-helper", str(TOOLS / "heldout_causal_v2.py"),
        "--heldout-review", str(TOOLS / "heldout-causal-v2-independent-review-v1.json"),
        "--owner-proof", str(OWNER), "--native-manifest", str(NATIVE), "--stop-unix", str(stop)]
    return [("plan", plan),
            ("training", helper + ["run", "--plan", str(OUTPUT / "plan.json"), "--mode", "training"]),
            ("freeze-heldout", helper + ["freeze-heldout", "--plan", str(OUTPUT / "plan.json")]),
            ("heldout", helper + ["run", "--plan", str(OUTPUT / "plan.json"), "--mode", "heldout"])]


def frozen_inputs(native):
    names = {"helper-freeze-v1.json", "helper-freeze-v2.json", "causal_driver_v3.py", "fresh_owner.py",
             "causal_outer_v3.py", "frozen_causal_outer_v1.py"}
    for manifest in ("helper-freeze-v1.json", "helper-freeze-v2.json"):
        value = read(TOOLS / manifest)
        for name, info in value["files"].items():
            require(file_sha(TOOLS / name) == info["sha256"], "Frozen causal helper changed")
            names.add(name)
    files = {str(TOOLS / name): file_sha(TOOLS / name) for name in sorted(names)}
    extra = [OWNER, ALLOCATION, SOURCE, NATIVE, QUALITY, CHECKPOINT, LIFECYCLE,
             PREFLIGHT, FULL]
    extra += [Path(p) for p in read(OWNER)["prior_inputs"].values()]
    files.update({str(p): file_sha(p) for p in extra})
    files.update(native)
    require(files[str(LIFECYCLE)] == LIFECYCLE_SHA
            and files[str(TOOLS / "frozen_causal_outer_v1.py")] == FROZEN_OUTER_SHA,
            "Wrong frozen cleanup dependencies")
    return files


def freeze(path, stop, pod_created_unix):
    require(time.time() + 300 < stop <= pod_created_unix + 5 * 3600,
            "Compute cutoff must leave an hour of the six-hour pod lease for preservation")
    require(read(OWNER)["pod_created_unix"] == pod_created_unix
            and read(OWNER)["pod_active_deadline_seconds"] == 21600,
            "Actual fresh pod lease/creation binding differs")
    native = fresh_owner_gate(OWNER, ALLOCATION, SOURCE, NATIVE, QUALITY)
    files = frozen_inputs(native)
    write_new(path, {"schema": "fresh-four-h100-causal-outer-v3", "complete": True,
        "driver_sha256": file_sha(__file__), "stop_unix": stop, "pod_created_unix": pod_created_unix,
        "pod_active_deadline_seconds": 21600, "child_cleanup_grace_seconds": 720,
        "commands": commands(stop), "files": files, "output": str(OUTPUT), "outer_output": str(OUTER),
        "pod_identity": read(OWNER)["pod_identity"], "old_partial_results_reused": False,
        "created_unix_ns": time.time_ns()})


def validate(path, expected):
    require(file_sha(path) == expected, "Outer request changed")
    r = read(path)
    require(r["schema"] == "fresh-four-h100-causal-outer-v3" and r["complete"] is True
            and r["driver_sha256"] == file_sha(__file__) and r["old_partial_results_reused"] is False
            and r["output"] == str(OUTPUT) and r["outer_output"] == str(OUTER)
            and r["pod_active_deadline_seconds"] == 21600 and r["child_cleanup_grace_seconds"] == 720
            and time.time() < r["stop_unix"] <= r["pod_created_unix"] + 5 * 3600
            and r["commands"] == [[k, v] for k, v in commands(r["stop_unix"])], "Outer scope/deadline/commands differ")
    require(r["pod_identity"] == read(OWNER)["pod_identity"], "Fresh outer owner differs")
    require(r["pod_created_unix"] == read(OWNER)["pod_created_unix"]
            and read(OWNER)["pod_active_deadline_seconds"] == 21600, "Fresh lease binding differs")
    native = fresh_owner_gate(OWNER, ALLOCATION, SOURCE, NATIVE, QUALITY)
    require(r["files"] == frozen_inputs(native), "Frozen outer input inventory/bytes changed")
    return r


def stage_receipt(stage, plan_sha, frozen, stop):
    p = OUTPUT / "plan.json"
    value = read(p)
    require(value["schema"] == "fresh-four-h100-causal-v3" and value["output"] == str(OUTPUT)
            and value["stop_unix"] == stop and value["model"] == MODEL
            and value["owner_proof"] == str(OWNER) and value["native_manifest"] == str(NATIVE)
            and value["allocation"]["gpu_count"] == 4, "Fresh operational plan differs")
    if stage == "plan":
        return {"path": str(p), "sha256": file_sha(p)}
    require(file_sha(p) == plan_sha, "Fresh plan changed after creation")
    if stage in ("training", "heldout"):
        return frozen.mode_receipt(OUTPUT / stage / "receipt.json", stage, plan_sha, 6 if stage == "training" else 3)
    selected_gate(OUTPUT / "heldout-prepared")
    admin = read(OUTPUT / "freeze-heldout/receipt.json")
    f = OUTPUT / "heldout-freeze-receipt.json"
    r = read(f)
    require(admin.get("complete") is True and admin.get("cleanup_complete") is True
            and admin["plan_sha256"] == plan_sha and r.get("complete") is True and r["plan_sha256"] == plan_sha
            and r["training_receipt_sha256"] == file_sha(OUTPUT / "training/receipt.json")
            and r["protocol_sha256"] == file_sha(OUTPUT / "heldout-prepared/protocol.json")
            and r["selected_requests_sha256"] == file_sha(OUTPUT / "heldout-prepared/selected-requests.json")
            and r["review_sha256"] == file_sha(TOOLS / "heldout-causal-v2-independent-review-v1.json")
            and r["no_heldout_model_queries_before_freeze"] is True, "Post-training subset freeze incomplete")
    return {"path": str(f), "sha256": file_sha(f)}


def run(path, expected):
    r = validate(path, expected)
    lib = load(LIFECYCLE, "frozen_causal_lifecycle", LIFECYCLE_SHA)
    frozen = load(TOOLS / "frozen_causal_outer_v1.py", "frozen_causal_lifetime", FROZEN_OUTER_SHA)
    OUTER.mkdir(exist_ok=False)
    lock = (OUTER / "outer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {"complete": False, "cleanup_complete": False, "request_sha256": expected,
             "controller_process": lib.process_record(os.getpid()), "started_unix_ns": time.time_ns(),
             "stages": [], "cleanup_errors": []}
    child = identity = None
    def save():
        state["updated_unix_ns"] = time.time_ns()
        lib.save(OUTER / "status.json", state)
    def cancelled(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError("Fresh causal outer interrupted; retain partial evidence")
    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    save()
    try:
        lib.gpu_idle()
        plan_sha = None
        for stage, command in r["commands"]:
            validate(path, expected)
            item = {"stage": stage, "command": command, "complete": False, "started_unix_ns": time.time_ns()}
            state["stage"] = stage
            state["stages"].append(item)
            save()
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            for key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                env.pop(key, None)
            with (OUTER / (stage + ".log")).open("xb") as log:
                child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                identity = lib.process_record(child.pid)
                require(identity and identity["pid"] == identity["pgrp"] == child.pid, "Cannot bind owned causal child")
                item["process"] = identity
                state["active_child"] = identity
                save()
                while child.poll() is None:
                    require(time.time() < r["stop_unix"], "Fresh causal compute cutoff reached")
                    save()
                    time.sleep(1)
                item["returncode"] = child.returncode
                require(child.returncode == 0, "Fresh causal stage failed: " + stage)
            item["cleanup"] = frozen.cleanup_owned(child, identity, lib, grace=720)
            child = identity = None
            state.pop("active_child", None)
            validate(path, expected)
            item["receipt"] = stage_receipt(stage, plan_sha, frozen, r["stop_unix"])
            if stage == "plan":
                plan_sha = item["receipt"]["sha256"]
                state["plan_sha256"] = plan_sha
            item.update(complete=True, finished_unix_ns=time.time_ns())
            save()
        state["complete"] = True
    except BaseException:
        state["error"] = traceback.format_exc()
        raise
    finally:
        if child is not None:
            try:
                state["emergency_child_cleanup"] = frozen.cleanup_owned(child, identity, lib, grace=720)
                state.pop("active_child", None)
            except BaseException:
                state["cleanup_errors"].append(traceback.format_exc())
        try:
            state["final_gpu_idle"] = lib.gpu_idle(wait=60)
        except BaseException:
            state["cleanup_errors"].append(traceback.format_exc())
        state["cleanup_complete"] = not state["cleanup_errors"]
        state["complete"] = state["complete"] and state["cleanup_complete"]
        state["finished_unix_ns"] = time.time_ns()
        save()
        write_new(OUTER / "receipt.json", state)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--stop-unix", type=float, required=True)
    p.add_argument("--pod-created-unix", type=float, required=True)
    p = sub.add_parser("run")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.request, args.stop_unix, args.pod_created_unix)
    else:
        run(args.request, args.request_sha256)
