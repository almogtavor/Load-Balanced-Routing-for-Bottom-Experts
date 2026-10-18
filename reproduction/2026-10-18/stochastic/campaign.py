#!/usr/bin/env python3
"""Finite observer-v4 training proof and unchanged 27-server quality campaign.

No cluster operations. Plan construction is CPU-only and freezes every cell
before any evaluation. Public source hashes establish lineage; same-server
activation is checked solely through the new private task adapter contract.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import urllib.request

DRIVER_SHA = "fd7abde187164028d14ecebd460b3348433fa4b66634a3b9aec07410de63b81b"
LIFECYCLE_SHA = "2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3"
TASK_SHA = "6e7905cff18904677c705f233ea502b21c3bf2a746e013c483086ba74fb58cec"
BUILDER_SHA = "57810d0eef0ad27cc634afb21eefd75519afc757c9419fffe6253ac600634b67"
ADAPTER_SHA = "55471264a02d1fc031034f26661c4a81463fe16941d5297bedfe7427a7b89c6f"
SELECTION_FREEZE_SHA = "7c9d6db6448657c61b5e9281b9b6d4de3352f909d1896769d6d754ecb6aa7a09"
ADAPTER_FREEZE_SHA = SELECTION_FREEZE_SHA
PRIVATE_FILES = ("__init__.py", "reference.py", "kernel.py", "plugin.py", "native_hooks.py", "worker_extension.py")
CLEANUP_MARGIN = 300
SCHEMA = "stochastic-observer-v4-campaign-v1"
LEDGER_SHA = "82cf281121ef80a3b53c09de5efe42d1f8b0e4cfdf02486591325af106fad475"
COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
ARMS = ("stock", "par6", "par4", "plain6", "plain4", "uniform6", "uniform4", "weighted6", "weighted4")
ORDERS = [list(ARMS[n:] + ARMS[:n]) for n in (0, 3, 6)]
PARAMETERS = {"stock": ("stock", None, None), "par6": ("par", 6, 8.0), "par4": ("par", 4, 8.0),
    "plain6": ("par", 6, 0.0), "plain4": ("par", 4, 0.0), "uniform6": ("uniform_hot", 6, None),
    "uniform4": ("uniform_hot", 4, None), "weighted6": ("weighted_hot", 6, None), "weighted4": ("weighted_hot", 4, None)}
PROMPTS = {"requests_sha256": "2f669baa51550f4a2d7cb2404259c24d81f96a5d4698a579fcbe4898445a024d",
           "preflight_sha256": "f9011f0f55ef8cfd92c2b65973febe3b527ed8fa554aa9afb301f62d71c517e9"}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def module(path, expected, name):
    need(sha(path) == expected, "Frozen dependency changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(result)
    finally:
        sys.dont_write_bytecode = previous
    return result


def safe_path(value):
    path = Path(value)
    need(path.is_absolute() and ".." not in path.parts and str(path) == value
         and not any(p.is_symlink() for p in [path, *path.parents]), "Unsafe tool/output path: " + value)
    return path


def server_command(plan):
    return [plan["vllm"], "serve", plan["model"], "--host", "127.0.0.1", "--port", "8077",
        "--served-model-name", "task-model", "--data-parallel-size", "4", "--tensor-parallel-size", "1",
        "--pipeline-parallel-size", "1", "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput",
        "--dtype", "bfloat16", "--expert-placement-strategy", "linear", '--compilation-config={"cudagraph_mode":"NONE"}',
        "--max-model-len", "8192", "--seed", "0", "--max-num-batched-tokens", "16384", "--max-num-seqs", "256",
        "--no-enable-prefix-caching", "--logprobs-mode", "raw_logprobs", "--disable-log-stats", "--jit-monitor-verbose",
        "--gpu-memory-utilization", "0.88", "--worker-extension-cls", "stochastic_tail.worker_extension.CheckpointExtension"]


def environment(plan, cell=None):
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(("PLACEMENT_", "STOCHASTIC_", "VLLM_", "TASK_BENCH_", "PAR_PROFILE_", "PAPER_PROBE_", "PREFILL_BENCH_", "HYBRID_", "CAUSAL_"))
                or "PROFILER" in key or key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
            env.pop(key, None)
    env.update(OMP_NUM_THREADS="1", NVSHMEM_QP_DEPTH="4096", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", NO_PROXY="localhost,127.0.0.1",
               CUDA_VISIBLE_DEVICES="0,1,2,3" if cell else "")
    if cell:
        env.update(VLLM_SERVER_DEV_MODE="1", VLLM_PLUGINS="stochastic_tail_experiment", PYTHONPATH=plan["private_site"] + ":" + plan["public_site"],
                   STOCHASTIC_EXPERIMENT_PLAN=cell["plan"], STOCHASTIC_EXPERIMENT_PLAN_SHA256=cell["plan_sha256"])
        mode, keep, penalty = PARAMETERS[cell["arm"]]
        if mode == "par":
            env.update(PLACEMENT_KEEP=str(keep), PLACEMENT_LOAD_PENALTY=str(penalty))
    else:
        # Canonical CPU preparation imports no model and needs only pristine
        # source lineage. No vLLM private/general plugin is enabled here.
        env["PYTHONPATH"] = plan["public_site"]
    return env


def assignment_gate(plan, allocation):
    seeds = plan.get("seeds", [0, 1, 2])
    need(isinstance(seeds, list) and seeds and all(type(s) is int and s in (0, 1, 2) for s in seeds)
         and seeds == sorted(set(seeds)), "Explicit nonempty unique sorted seed subset required")
    assignment = read(plan["global_assignment"])
    need(assignment["schema"] == "stochastic-seed-host-assignment-v1"
         and assignment["global_required_cells"] == 27 and assignment["orders"] == ORDERS
         and assignment["arms"] == list(ARMS), "Global predeclared27-cell assignment differs")
    rows = assignment["assignments"]
    need(len(rows) == 3 and {r["seed"] for r in rows} == set(range(3)), "Allthree seed/host assignments required")
    local = {r["seed"] for r in rows if r["pod_uid"] == allocation["uid"]}
    need(local == set(seeds), "Local explicit seed subset differs from global assigned pod UID")
    for row in rows:
        need(type(row["seed"]) is int and type(row["stop_unix"]) in (int, float)
             and type(row["pod_created_unix"]) in (int, float) and math.isfinite(row["stop_unix"])
             and math.isfinite(row["pod_created_unix"]) and row["stop_unix"] == row["pod_created_unix"] + 36000,
             "Assignment deadline is not pod creation+10h")
        if row["seed"] in seeds:
            need(row["node"] == allocation["node"] and row["pod_uid"] == allocation["uid"]
                 and row["stop_unix"] == plan["stop_unix"] and row["pod_created_unix"] == plan["pod_created_unix"],
                 "Local seed host/deadline differs from global assignment")
    return seeds, assignment


def training_proof_gate(plan, files):
    """Replay three closed proof receipts and root's explicit remote PID checks.

    Proof files may be copies collected from other pods. Their immutable content
    hashes bind the original campaign/launch/audit; local PID inspection cannot
    establish remote exit, so the root records that observation explicitly.
    """
    rows = plan.get("training_proofs", [])
    if plan["campaign_kind"] == "proof":
        need(rows == [], "Training proof cannot consume or skip a previous proof")
        return
    need(plan["campaign_kind"] == "quality" and len(rows) == 3
         and {r["seed"] for r in rows} == {0, 1, 2}, "All three closed training proofs required before quality planning")
    assignment = {r["seed"]: r for r in read(plan["global_assignment"])["assignments"]}
    package = {"stochastic_tail/" + n: files[str(Path(plan["private_site"]) / "stochastic_tail" / n)] for n in PRIVATE_FILES}
    for row in rows:
        values = {}
        for key in ("plan", "receipt", "launch", "audit", "exit"):
            ref = row[key]
            need(files.get(ref["path"]) == ref["sha256"] == sha(ref["path"]), "Unpinned/changed collected proof: " + key)
            values[key] = read(ref["path"])
        proof, receipt, launch, audit, exited = (values[k] for k in ("plan", "receipt", "launch", "audit", "exit"))
        seed = row["seed"]
        need(proof["schema"] == SCHEMA and proof["campaign_kind"] == "proof" and proof["observer_revision"] == 4
             and proof["seeds"] == [seed] and proof["allocation"]["uid"] == assignment[seed]["pod_uid"]
             and proof["files"][proof["private_freeze"]] == SELECTION_FREEZE_SHA
             and proof["files"][proof["task_adapter"]] == ADAPTER_SHA
             and {"stochastic_tail/" + n: proof["files"][str(Path(proof["private_site"]) / "stochastic_tail" / n)] for n in PRIVATE_FILES} == package,
             "Collected proof is another observer, seed, host, or installed package")
        need(receipt.get("complete") is True and receipt.get("cleanup_complete") is True
             and receipt.get("closed_ledgers_audited") is True and receipt["campaign_kind"] == "proof"
             and receipt["plan_sha256"] == row["plan"]["sha256"] and len(receipt["launches"]) == 1,
             "Collected training campaign did not close and audit")
        cell = receipt["launches"][0]
        need(cell.get("complete") is True and cell["receipt_sha256"] == row["launch"]["sha256"]
             and cell["raw_ledger_audit_sha256"] == row["audit"]["sha256"] and cell["arm"] == "weighted6"
             and cell["seed"] == seed and cell["position"] == 0,
             "Collected proof cell/audit acceptance differs")
        need(launch.get("complete") is True and launch.get("workload_complete") is True
             and launch["campaign_plan_sha256"] == row["plan"]["sha256"] and not launch["finalization_errors"]
             and launch["server_cleanup"]["remaining_members"] == [], "Collected proof server not closed")
        need(audit.get("complete") is True and audit["schema"] == "closed-stochastic-native-seal-audit-v2"
             and audit["purpose"] == "proof" and audit["seed"] == seed and audit["arm"] == "weighted6"
             and audit["campaign_sha256"] == row["plan"]["sha256"] and audit["launch_sha256"] == row["launch"]["sha256"]
             and audit["adapter_sha256"] == ADAPTER_SHA and len(audit["worker_pids"]) == 4
             and audit["helper_sha256"] == files[plan["ledger_helper"]], "Strict native-seal proof audit differs")
        identity = receipt["controller_process"]
        need(exited.get("complete") is True and exited.get("exited") is True
             and exited["pod_uid"] == assignment[seed]["pod_uid"] and exited["controller_process"] == identity
             and exited["observed_process"] is None and exited["verified_unix"] >= receipt["finished_unix"],
             "Root has not verified this original remote controller PID/start_ticks exited")


def adapter_command(plan, cell, mode):
    # -c/runpy prevents the source helper directory from shadowing the installed
    # private package that every worker and read_plan require by absolute path.
    adapter, private, public = plan["task_adapter"], plan["private_site"], plan["public_site"]
    code = ("import os,runpy,sys;"
            "sys.path[:]=[" + repr(private) + "," + repr(public) + "]+[p for p in sys.path if os.path.realpath(p or os.getcwd())!=" + repr(str(Path(adapter).parent)) + "];"
            "sys.argv=[" + repr(adapter) + "]+sys.argv[1:];runpy.run_path(" + repr(adapter) + ",run_name='__main__')")
    command = ["python3", "-c", code, mode, "--task-harness", plan["task_harness"],
        "--plan", cell["plan"], "--plan-sha256", cell["plan_sha256"], "--prepared", plan["prepared"],
        "--url", plan["url"], "--served-model", "task-model", "--request-timeout", str(plan["request_timeout"]),
        "--output", str(Path(cell["directory"]) / mode)]
    if mode == "score":
        command += ["--preflight", str(Path(cell["directory"]) / "preflight")]
    return command


def make_plan(args):
    need(sha(args.request) == args.request_sha256, "Root request changed before CPU freeze")
    q = read(args.request)
    need(q.get("campaign_kind") in ("proof", "quality"), "Explicit new proof or quality campaign required")
    need(type(q["stop_unix"]) in (int, float) and type(q["pod_created_unix"]) in (int, float)
         and math.isfinite(q["stop_unix"]) and math.isfinite(q["pod_created_unix"])
         and q["stop_unix"] == q["pod_created_unix"] + 36000
         and time.time() + 300 < q["stop_unix"], "Use this owned pod creation+10h cutoff")
    root, prepared = safe_path(q["output"]), safe_path(q["prepared"])
    need(not root.exists() and not prepared.exists(), "Fresh campaign/preparation paths required; no replacement")
    files = {}
    for name, expected in q["expected_files"].items():
        path = safe_path(name)
        need(path.is_file() and sha(path) == expected, "Actual frozen input differs: " + name)
        files[name] = expected
    need(files.get(str(Path(__file__).resolve())) == sha(__file__), "Controller must be request-pinned")
    for key, expected in (("base_driver", DRIVER_SHA), ("lifecycle_helper", LIFECYCLE_SHA), ("task_harness", TASK_SHA),
        ("task_adapter", ADAPTER_SHA), ("launch_builder", BUILDER_SHA), ("private_freeze", SELECTION_FREEZE_SHA), ("adapter_freeze", ADAPTER_FREEZE_SHA), ("ledger_helper", LEDGER_SHA)):
        need(files[q[key]] == expected, "Canonical dependency differs: " + key)
    for key in ("task_adapter", "launch_builder", "private_freeze", "adapter_freeze", "gpu_gate", "public_kernel_gate", "source_proof", "checkpoint_proof", "model_ready", "allocation_proof", "global_assignment", "vllm", "ledger_helper", "ledger_base"):
        need(q[key] in files, "Unpinned required input: " + key)
    lib = module(q["lifecycle_helper"], LIFECYCLE_SHA, "stochastic_frozen_lifecycle")
    source = lib.source_identity(q["source_proof"], q["source_root"], q["public_site"])
    checkpoint = lib.checkpoint_identity(q["checkpoint_proof"], q["model"])
    allocation, deployment = read(q["allocation_proof"]), read(q["source_proof"])
    need(allocation.get("verified_live") is True and allocation.get("gpu_count") == 8 and "H100" in allocation.get("gpu_name", "")
         and allocation.get("namespace") == "itay" and all(allocation.get(k) == deployment["pod_identity"].get(k) for k in ("namespace", "pod", "uid", "node", "image")), "Source/allocation not this owned eight-H100 host")
    seeds, assignment = assignment_gate(q, allocation)
    need(q["campaign_kind"] != "proof" or len(seeds) == 1, "One weighted6 training proof per assigned host/seed")
    ready = read(q["model_ready"])
    need(ready.get("complete") is True and ready["proof_sha256"] == sha(q["checkpoint_proof"])
         and read(q["public_kernel_gate"]).get("complete") is True, "Fresh model/public CUDA gate incomplete")
    frozen = read(q["private_freeze"])
    need(frozen.get("complete") is True, "Private source/helper freeze incomplete")
    needed = [Path(q["private_site"]) / "stochastic_tail" / name for name in PRIVATE_FILES]
    need(all(files.get(str(p)) == frozen["files"]["stochastic_tail/" + p.name]["sha256"] for p in needed),
         "Actual installed private package must match every reviewed source-freeze hash")
    gate_helper = str(Path(q["launch_builder"]).with_name("gpu_gate.py"))
    need(files.get(gate_helper) == frozen["files"]["gpu_gate.py"]["sha256"],
         "Same reviewed 180-case CUDA gate helper must be request-pinned beside builder")
    need(Path(q["ledger_base"]).name == "audit_stochastic_ledgers_v1.py"
         and Path(q["ledger_helper"]).name == "audit_stochastic_ledgers_v2.py"
         and Path(q["ledger_helper"]).parent == Path(q["ledger_base"]).parent
         and files[q["ledger_base"]] == "5f17ce847ad56c38accfe3f53d98d587efc102bb5871fe096f5263f0533d6dc5",
         "Stage the unchanged original source-import helper beside the new native-seal auditor")
    training_proof_gate(q, files)
    # Guarantee imported private package comes from the installed isolated site,
    # even when builder is stored beside an otherwise identical source folder.
    original_path = sys.path[:]
    sys.path[:] = [q["private_site"], q["public_site"]] + [p for p in sys.path if Path(p or os.getcwd()).resolve() != Path(q["launch_builder"]).parent.resolve()]
    root.mkdir(parents=True, exist_ok=False)
    try:
        builder = module(q["launch_builder"], files[q["launch_builder"]], "stochastic_frozen_launch_builder")
        need(Path(sys.modules["stochastic_tail.plugin"].__file__).resolve() == Path(q["private_site"]) / "stochastic_tail/plugin.py", "Builder imported a different private package")
        need(builder.ARMS == PARAMETERS, "Private builder nine-arm methods differ")
        cells = []
        for seed in seeds:
            order = ORDERS[seed]
            for position, arm in enumerate(order if q["campaign_kind"] == "quality" else ["weighted6"]):
                directory = root / f"seed{seed}" / f"position{position}-{arm}"
                launch_plan = root / "frozen-launch-plans" / f"seed{seed}-position{position}-{arm}.json"
                builder.build(SimpleNamespace(output=launch_plan, private_site=Path(q["private_site"]), public_site=Path(q["public_site"]),
                    public_source_proof=Path(q["source_proof"]), gpu_gate=Path(q["gpu_gate"]), checkpoint_proof=Path(q["checkpoint_proof"]),
                    model=Path(q["model"]), evidence=directory / "ledger", arm=arm, seed=seed, purpose=q["campaign_kind"]))
                value = read(launch_plan)
                need(value["arm"] == arm and value["parameters"]["seed"] == seed and value["purpose"] == q["campaign_kind"]
                     and value["record_calls"] is True, "Private per-cell plan differs")
                cells.append({"seed": seed, "position": position, "arm": arm, "directory": str(directory),
                              "plan": str(launch_plan), "plan_sha256": sha(launch_plan)})
                files[str(launch_plan)] = sha(launch_plan)
    finally:
        sys.path[:] = original_path
    dataset_files = sorted(p for p in Path(q["datasets"]).rglob("*") if p.is_file())
    need(dataset_files and all(str(p) in files for p in dataset_files), "Every dataset/reference file must be input-pinned")
    plan = {key: q[key] for key in ("output", "prepared", "private_site", "public_site", "source_root", "source_proof", "checkpoint_proof", "model", "model_ready", "public_kernel_gate", "gpu_gate", "private_freeze", "adapter_freeze", "allocation_proof", "base_driver", "lifecycle_helper", "task_harness", "task_adapter", "launch_builder", "vllm", "datasets", "stop_unix", "pod_created_unix", "global_assignment", "ledger_helper", "ledger_base")}
    plan.update(schema=SCHEMA, observer_revision=4, campaign_kind=q["campaign_kind"], training_proofs=q.get("training_proofs", []), created_unix=time.time(), source_commit=COMMIT, revision=REVISION,
        files=files, source_identity=source, checkpoint_identity=checkpoint, allocation=allocation, cells=cells, orders=ORDERS,
        expected_prompt_hashes=PROMPTS, server_command=server_command(plan), url="http://127.0.0.1:8077",
        prepare_timeout=600, boot_timeout=2400, proof_timeout=600, preflight_timeout=600, score_timeout=2700, audit_timeout=600, cleanup_margin=CLEANUP_MARGIN,
        request_timeout=600, seeds=seeds, max_fresh_servers=len(cells), global_required_cells=27, max_replacement_servers=0,
        observer_scope="MRV2 actual-instance observer and pause/RPC terminal seal; training preflight resume changes scheduling. V3 outcomes excluded. Weighted6 train proof certifies observer/control path, not nine-method accuracy or speed.",
        request_sha256=args.request_sha256, request_path=str(args.request.resolve()),
        order_scope="Rotations0/3/6 over9 labels; each arm early/middle/late once, partial position balance only",
        analysis_scope="Global full1319GSM8K+10042HellaSwag27-cell matrix; three seed-plus-host configuration units, seed/host variability inseparable; local receipts only assignedcells, no item/round pseudoreplication/best-seed selection",
        causal_scope="Priced PAR teacher scores may use future tokens; stochastic keyed by scheduling/startup/dummy call counters; no strict causal likelihood or request invariance claim",
        method_scope="Uniform-all-hot vs weighted-hot-top32 changes range and weights; combined method comparison. Lambda0 plain4/6 controls separate removal of tile load coupling from stochastic selection.")
    plan["prepare_command"] = ["python3", plan["task_harness"], "prepare", "--datasets", plan["datasets"], "--model", plan["model"],
        "--checkpoint-proof", plan["checkpoint_proof"], "--source-proof", plan["source_proof"], "--production-root", plan["source_root"], "--output", plan["prepared"]]
    files[str(args.request.resolve())] = args.request_sha256
    write_new(root / "plan.json", plan)
    print(json.dumps({"complete": True, "plan": str(root / "plan.json"), "plan_sha256": sha(root / "plan.json"), "cells": len(cells), "GPU_model_requests": 0}))


def make_campaign_class(base, lib):
    class Campaign(base.Campaign):
        def command(self, command, log, env, timeout, server=None):
            remaining = self.plan["stop_unix"] - CLEANUP_MARGIN - time.time()
            need(remaining > 0, "Compute cutoff reached; cleanup margin reserved")
            return super().command(command, log, env, min(timeout, remaining), server)

        def validate(self, allow_past_cutoff=False):
            p = self.plan
            seeds, _ = assignment_gate(p, p["allocation"])
            need(p["schema"] == SCHEMA and p["observer_revision"] == 4 and p["campaign_kind"] in ("proof", "quality") and p["orders"] == ORDERS
                 and p["max_fresh_servers"] == (9 if p["campaign_kind"] == "quality" else 1) * len(seeds) and p["max_replacement_servers"] == 0
                 and p["global_required_cells"] == 27 and p["stop_unix"] == p["pod_created_unix"] + 36000
                 and (allow_past_cutoff or time.time() < p["stop_unix"] - CLEANUP_MARGIN), "Frozen design/cutoff changed")
            need(p["server_command"] == server_command(p) and (p["prepare_timeout"], p["boot_timeout"], p["preflight_timeout"], p["score_timeout"], p["request_timeout"], p["proof_timeout"], p["audit_timeout"], p["cleanup_margin"]) == (600, 2400, 600, 2700, 600, 600, 600, CLEANUP_MARGIN), "Fixed commands/deadlines drift")
            need(p["expected_prompt_hashes"] == PROMPTS, "Canonical preparation IDs changed")
            need([(c["seed"], c["position"], c["arm"]) for c in p["cells"]] == [(s, i, a) for s in seeds for i, a in enumerate(ORDERS[s] if p["campaign_kind"] == "quality" else ["weighted6"])], "Exact assigned-cell order changed")
            for name, expected in p["files"].items():
                need(sha(name) == expected, "Frozen campaign input changed: " + name)
            training_proof_gate(p, p["files"])
            need(lib.source_identity(p["source_proof"], p["source_root"], p["public_site"]) == p["source_identity"], "Public source lineage drift")
            need(lib.checkpoint_identity(p["checkpoint_proof"], p["model"]) == p["checkpoint_identity"], "Fresh pinned checkpoint/stat drift")
            need(read(p["model_ready"])["complete"] is True and read(p["gpu_gate"])["complete"] is True, "Readiness proof incomplete")

        def launch_cell(self, cell):
            self.validate()
            directory = Path(cell["directory"])
            need(directory == self.root / f"seed{cell['seed']}" / f"position{cell['position']}-{cell['arm']}", "Cell output path differs")
            directory.mkdir(parents=True, exist_ok=False)
            entry = {"complete": False, **cell, "campaign_plan_sha256": sha(self.plan_path), "started_unix": time.time(), "finalization_errors": []}
            index = len(self.state["launches"])
            self.state["launches"].append({"complete": False, **cell})
            self.state["stage"] = f"seed{cell['seed']} position{cell['position']} {cell['arm']}"
            self.save()
            try:
                base.gpu_idle(wait_seconds=30)
                with socket.socket() as sock:
                    need(sock.connect_ex(("127.0.0.1", 8077)) != 0, "Dedicated HTTP port already occupied")
                env = environment(self.plan, cell)
                entry["server_command"] = self.plan["server_command"]
                entry["server_environment"] = {k: env[k] for k in sorted(env) if k.startswith(("PLACEMENT_", "STOCHASTIC_", "VLLM_")) or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NO_PROXY")}
                with (directory / "server.log").open("xb") as stream:
                    self.server = subprocess.Popen(self.plan["server_command"], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    self.server_id = base.process_record(self.server.pid)
                    need(self.server_id and self.server_id["pgrp"] == self.server.pid, "Cannot bind owned fresh server")
                    entry["server_process"] = self.server_id
                    self.state["active_server"] = {"process": self.server_id, "cell": cell}
                    self.save()
                    deadline = min(time.time() + self.plan["boot_timeout"], self.plan["stop_unix"] - CLEANUP_MARGIN)
                    while True:
                        need(self.server.poll() is None and time.time() < deadline, "Server startup failed/deadline")
                        try:
                            with urllib.request.urlopen(self.plan["url"] + "/health", timeout=2) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        self.save()
                        time.sleep(2)
                    entry["healthy_unix"] = time.time()
                    for mode in (("proof",) if self.plan["campaign_kind"] == "proof" else ("preflight", "score")):
                        self.validate()
                        command = adapter_command(self.plan, cell, mode)
                        client_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
                        entry[mode] = self.command(command, directory / (mode + ".log"), client_env, self.plan[mode + "_timeout"], self.server)
                        result = read(directory / mode / "result.json")
                        need(result.get("complete") is True and result["experiment_plan_sha256"] == cell["plan_sha256"]
                             and result["arm"] == cell["arm"] and result["router_seed"] == cell["seed"]
                             and result["worker_pids"] and len(result["worker_identity_records"]) == 4, "Private adapter/result activation differs")
                        entry[mode + "_result_sha256"] = sha(directory / mode / "result.json")
                    if self.plan["campaign_kind"] == "quality":
                        pre, score = read(directory / "preflight/result.json"), read(directory / "score/result.json")
                        need(pre["preflight_pass"] is True and pre["worker_identity_records"] == score["worker_identity_records"]
                             and pre["worker_pids"] == score["worker_pids"], "Same-server private worker contract differs")
                    entry["workload_complete"] = True
            except BaseException:
                entry["workload_error"] = traceback.format_exc()
            finally:
                for label in ("child", "server"):
                    child, identity = getattr(self, label), getattr(self, label + "_id")
                    if child is not None:
                        if label == "server":
                            try:
                                members = base.group_members(child.pid)
                                entry["owned_group_members_before_cleanup"] = members
                                if entry.get("workload_complete") is True:
                                    pids = read(directory / ("proof" if self.plan["campaign_kind"] == "proof" else "score") / "result.json")["worker_pids"]
                                    need(set(pids) <= {m["pid"] for m in members}
                                         and all(m["pgrp"] == child.pid for m in members), "Private workers not in owned server group before closure")
                            except BaseException:
                                entry["finalization_errors"].append({"step": "worker_group_membership", "error": traceback.format_exc()})
                        try:
                            entry[label + "_cleanup"] = base.stop_owned(child, identity)
                            setattr(self, label, None);setattr(self, label + "_id", None)
                            self.state.pop("active_command" if label == "child" else "active_server", None)
                        except BaseException:
                            entry["finalization_errors"].append({"step": label + "_cleanup", "error": traceback.format_exc()})
                for label, action in (("source_model_inputs", lambda: self.validate(allow_past_cutoff=True)), ("gpu_idle", lambda: base.gpu_idle(wait_seconds=30))):
                    try:
                        action()
                    except BaseException:
                        entry["finalization_errors"].append({"step": label, "error": traceback.format_exc()})
                try:
                    entry["server_log_sha256"] = sha(directory / "server.log")
                    entry["closed_ledger_files_sha256"] = {str(p): sha(p) for p in sorted((directory / "ledger").glob("pid*.jsonl"))}
                    need(entry["closed_ledger_files_sha256"], "No private worker/call ledger preserved")
                except BaseException:
                    entry["finalization_errors"].append({"step": "raw_log_ledger_retention", "error": traceback.format_exc()})
                entry["complete"] = entry.get("workload_complete") is True and "workload_error" not in entry and not entry["finalization_errors"]
                entry["finished_unix"] = time.time()
                write_new(directory / "launch.json", entry)
                self.state["launches"][index] = {"complete": False, "execution_complete": entry["complete"], **cell, "receipt": str(directory / "launch.json"), "receipt_sha256": sha(directory / "launch.json")}
                self.save()
            need(entry["complete"], "Cell failed; immutable raw/receipt retained, no replacement launch")
            # The execution receipt is immutable before raw replay. Campaign
            # acceptance is separate, so a failed auditor cannot bless a cell
            # and neither receipt needs a circular self hash or a rewrite.
            self.state["launches"][index]["complete"] = False
            self.save()
            audit_path = directory / "raw-ledger-audit.json"
            command = ["python3", self.plan["ledger_helper"], "--launch", str(directory / "launch.json"),
                "--launch-sha256", sha(directory / "launch.json"), "--campaign", str(self.plan_path),
                "--campaign-sha256", sha(self.plan_path), "--adapter", self.plan["task_adapter"],
                "--adapter-sha256", ADAPTER_SHA, "--output", str(audit_path)]
            try:
                audit_process = self.command(command, directory / "raw-ledger-audit.log", environment(self.plan), self.plan["audit_timeout"])
                report = read(audit_path)
                need(report.get("complete") is True and report["launch_sha256"] == sha(directory / "launch.json")
                     and report["campaign_sha256"] == sha(self.plan_path)
                     and report["helper_sha256"] == self.plan["files"][self.plan["ledger_helper"]]
                     and report["purpose"] == self.plan["campaign_kind"], "Strict closed native ledger acceptance failed")
                self.state["launches"][index].update(complete=True, raw_ledger_audit=str(audit_path),
                    raw_ledger_audit_sha256=sha(audit_path), raw_ledger_audit_process=audit_process)
            except BaseException:
                self.state["launches"][index]["raw_ledger_audit_error"] = traceback.format_exc()
                if audit_path.is_file():
                    self.state["launches"][index].update(raw_ledger_audit=str(audit_path), raw_ledger_audit_sha256=sha(audit_path))
                raise
            finally:
                self.save()

        def run(self):
            try:
                self.validate()
                base.gpu_idle()
                self.state["controller_process"] = base.process_record(os.getpid())
                self.state["stage"] = "Canonical CPU prompt preparation"
                self.save()
                self.state["prepare"] = self.command(self.plan["prepare_command"], self.root / "prepare.log", environment(self.plan), self.plan["prepare_timeout"])
                preparation = read(Path(self.plan["prepared"]) / "preparation.json")
                need(preparation["complete"] is True and all(preparation[k] == v for k, v in PROMPTS.items()), "Fresh canonical preparation IDs differ")
                self.state["shared_preparation_sha256"] = sha(Path(self.plan["prepared"]) / "preparation.json")
                for cell in self.plan["cells"]:
                    self.launch_cell(cell)
                need(len(self.state["launches"]) == self.plan["max_fresh_servers"] and all(r["complete"] for r in self.state["launches"]), "Assigned local cells incomplete")
                self.state.update(complete=True, execution_complete=True, local_assigned_cells_complete=True,
                    local_completed_cells=len(self.state["launches"]), global_required_cells=27, global_matrix_audited=False,
                    observer_revision=4, campaign_kind=self.plan["campaign_kind"], closed_ledgers_audited=True,
                    stage="Assigned local cells and strict closed native-seal ledgers complete; global barrier/analysis pending")
            except BaseException:
                self.state.update(error=traceback.format_exc(), stage="Failed; raw retained, no partial effect or replacement")
                raise
            finally:
                errors = []
                for label in ("child", "server"):
                    child, identity = getattr(self, label), getattr(self, label + "_id")
                    if child is not None:
                        try:
                            self.state["final_" + label + "_cleanup"] = base.stop_owned(child, identity)
                        except BaseException:
                            errors.append({"step": label, "error": traceback.format_exc()})
                try:
                    base.gpu_idle(wait_seconds=30)
                except BaseException:
                    errors.append({"step": "gpu_idle", "error": traceback.format_exc()})
                if errors:
                    self.state["complete"] = False
                self.state.update(cleanup_complete=not errors, cleanup_errors=errors, finished_unix=time.time())
                self.save()
                write_new(self.root / "receipt.json", self.state)
                self.lock.close()
                need(not errors, "Owned final cleanup not verified")
    return Campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.mode == "plan":
        make_plan(args)
        return
    need(sha(args.plan) == args.plan_sha256, "Frozen campaign plan changed")
    plan = read(args.plan)
    need(plan["files"][str(Path(__file__).resolve())] == sha(__file__), "Controller source differs from freeze")
    base = module(plan["base_driver"], DRIVER_SHA, "stochastic_frozen_task_driver")
    lib = module(plan["lifecycle_helper"], LIFECYCLE_SHA, "stochastic_frozen_lifecycle")
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN);signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError("Owned stochastic campaign received " + signal.Signals(signum).name)
    signal.signal(signal.SIGTERM, cancel);signal.signal(signal.SIGINT, cancel)
    make_campaign_class(base, lib)(args.plan).run()


if __name__ == "__main__":
    main()
