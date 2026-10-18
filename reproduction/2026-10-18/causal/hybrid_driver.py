#!/usr/bin/env python3
"""Bounded owned-process controller for genuine EP4+EP4 NIXL PD experiments.

No cluster mutations. Three modes use separate fresh server pairs: instrumented
training fidelity proof; nine clean timing pairs in three paired sweeps; and
three actual hybrid full-GSM8K quality pairs. Every mode has an immutable plan,
finite deadline, exclusive GPU lock, process identity checks and owned cleanup.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from common_v2 import (MODEL_ID, REVISION, SOURCE_COMMIT, encoded, file_sha, read, require, sha, write_new)
from hybrid_timing import run_round

LIFECYCLE_SHA = "2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3"
ARMS = {"stock": (None, None), "keep6p8": (6, 8), "keep4p8": (4, 8)}
SWEEPS = [["stock", "keep6p8", "keep4p8"], ["keep6p8", "keep4p8", "stock"], ["keep4p8", "stock", "keep6p8"]]


def lifecycle(path):
    require(file_sha(path) == LIFECYCLE_SHA, "Reviewed lifecycle/parser dependency changed")
    spec = importlib.util.spec_from_file_location("frozen_prefill_lifecycle", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def commands(plan, role):
    producer = role == "prefill"
    kv = {"kv_connector": "NixlConnector", "kv_role": "kv_producer" if producer else "kv_consumer",
          "kv_load_failure_policy": "fail"}
    return [plan["vllm"], "serve", plan["model"], "--served-model-name", "task-model", "--host", "127.0.0.1",
            "--port", str(8100 if producer else 8200), "--data-parallel-rpc-port", str(29550 if producer else 29650),
            "--data-parallel-size", "4", "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
            "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput", "--dtype", "bfloat16",
            "--expert-placement-strategy", "linear", '--compilation-config={"cudagraph_mode":"NONE"}',
            "--max-model-len", "8192", "--seed", "0", "--max-num-batched-tokens", "16384", "--max-num-seqs", "256",
            "--no-enable-prefix-caching", "--logprobs-mode", "raw_logprobs", "--enable-prompt-tokens-details",
            "--gpu-memory-utilization", str(plan["gpu_util"]), "--kv-transfer-config", encoded(kv).decode()]


def environment(plan, role, arm, probe=None):
    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(("PLACEMENT_", "VLLM_", "TASK_BENCH_", "PAPER_PROBE_", "PREFILL_BENCH_", "HYBRID_", "CAUSAL_"))
                or "PROFILER" in key or key in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")):
            env.pop(key)
    env.update(PYTHONPATH=plan["site_root"], VLLM_PLUGINS="placement_aware_router", OMP_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NVSHMEM_QP_DEPTH="4096",
               CUDA_VISIBLE_DEVICES="0,1,2,3" if role == "prefill" else "4,5,6,7",
               VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1", VLLM_NIXL_SIDE_CHANNEL_PORT="5600" if role == "prefill" else "5610",
               UCX_RCACHE_MAX_UNRELEASED="1024", NO_PROXY="localhost,127.0.0.1")
    if role == "prefill" and ARMS[arm][0] is not None:
        env.update(PLACEMENT_KEEP=str(ARMS[arm][0]), PLACEMENT_LOAD_PENALTY=str(ARMS[arm][1]))
    if probe is not None:
        env.update(PLACEMENT_PROBE=plan["phase_probe"], HYBRID_PROBE_DIR=str(probe), HYBRID_ROLE=role,
                   HYBRID_ARM=arm if role == "prefill" else "stock")
    else:
        require("PLACEMENT_PROBE" not in env and not any("PROFILER" in k for k in env), "Clean timing env contains instrumentation")
    return env


def make_plan(args):
    lib = lifecycle(args.lifecycle_helper)
    allocation = read(args.allocation_proof)
    require(allocation.get("verified_live") and allocation.get("namespace") == "itay" and allocation.get("gpu_count") == 8
            and "H100" in allocation.get("gpu_name", ""), "Fresh exact eight-H100 allocation proof required")
    deployed = read(args.source_proof).get("pod_identity", {})
    require(all(deployed.get(key) == allocation.get(key) for key in ("namespace", "pod", "uid", "node", "image")),
            "Allocation and source deployment identify different pod")
    directory = Path(__file__).resolve().parent
    source = lib.source_identity(args.source_proof, args.production_root, args.site_root)
    checkpoint = lib.checkpoint_identity(args.checkpoint_proof, args.model)
    runtime = read(args.runtime_preflight)
    require(runtime.get("complete") is True and runtime["packages"]["vllm"].split("+")[0] == "0.31.0"
            and runtime["packages"]["nixl"] == "1.4.1" and runtime["nixl_agent_constructed"] is True,
            "Actual exact-version NIXL/UCX dependency preflight required")
    for item in runtime["module_files"].values():
        require(file_sha(item["path"]) == item["sha256"], "Installed native connector code differs from preflight")
    require(file_sha(args.preflight_inputs) == "183888c4a3d0402ab97806d453449dbf740e445901b80ee3d5e156ffc7dd7017",
            "Training input file differs")
    full = read(args.quality_inputs)
    require(file_sha(args.quality_inputs) == "6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b"
            and len(full) == 41487 and len([r for r in full if r["task"] == "gsm8k"]) == 1319,
            "Require canonical full task request file")
    original = lib.verify_prompts(args.prefill_prompts)
    prompts = [{"request_id": i * 4 + j, "token_ids": row["token_ids"][j * 1024:(j + 1) * 1024]}
               for i, row in enumerate(original) for j in range(4)]
    require(len(prompts) == 256 and all(len(r["token_ids"]) == 1024 for r in prompts), "Synthetic deployment workload shape differs")
    prerequisites = []
    for path in args.prerequisite_receipts:
        receipt = read(path)
        require(receipt.get("complete") is True and receipt.get("cleanup_complete") is True, "Prior campaign not completed and cleaned: " + str(path))
        prerequisites.append({"path": str(path.resolve()), "sha256": file_sha(path)})
    require(len(prerequisites) >= 3 and len({p["path"] for p in prerequisites}) == len(prerequisites),
            "Need completed prefill timing, profile and outer-controller receipts")
    must_be_exited = []
    perf_receipts = [(path, read(path)) for path in args.prerequisite_receipts if len(read(path).get("launches", [])) == 36]
    profile_receipts = [(path, read(path)) for path in args.prerequisite_receipts if "main_guard" in read(path)]
    require(len(perf_receipts) == len(profile_receipts) == 1, "Exactly one completed36launch primary timing and one linked profile receipt required")
    perf_path, perf_receipt = perf_receipts[0]
    profile_path, profile_receipt = profile_receipts[0]
    guard = profile_receipt["main_guard"]
    require(guard["receipt_sha256"] == file_sha(perf_path) and guard["complete_launches"] == 36
            and guard["controller_original_exited"] is True and guard["main_lock_held"] is True
            and len(profile_receipt["launches"]) == 3 and all(x["complete"] for x in profile_receipt["launches"]),
            "Profile receipt does not close the same completed timing campaign")
    for _, receipt in (perf_receipts + profile_receipts):
        identity = receipt["controller_process"]
        must_be_exited.append({"pid": int(identity["pid"]), "start_ticks": str(identity["start_ticks"])})
    for value in args.must_have_exited:
        pid, start_ticks = value.split(":")
        item = {"pid": int(pid), "start_ticks": str(start_ticks)}
        live = lib.process_record(item["pid"])
        require(not live or str(live["start_ticks"]) != item["start_ticks"] or live["state"] == "Z", "Prior owned controller still alive")
        must_be_exited.append(item)
    for item in must_be_exited:
        live = lib.process_record(item["pid"])
        require(not live or str(live["start_ticks"]) != item["start_ticks"] or live["state"] == "Z", "Prior timing/profile process still alive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "prompts256x1024.json", prompts)
    plan = {"schema": "real-nixl-hybrid-v1", "output": str(output), "source_commit": SOURCE_COMMIT,
            "model_id": MODEL_ID, "checkpoint_revision": REVISION, "model": str(args.model.resolve()),
            "source_root": str(args.production_root.resolve()), "site_root": str(args.site_root.resolve()),
            "source_proof": str(args.source_proof.resolve()), "checkpoint_proof": str(args.checkpoint_proof.resolve()),
            "source_identity": source, "checkpoint_identity": checkpoint,
            "allocation": allocation, "allocation_proof": str(args.allocation_proof.resolve()),
            "runtime_preflight": str(args.runtime_preflight.resolve()), "lifecycle_helper": str(args.lifecycle_helper.resolve()),
            "prerequisite_receipts": prerequisites, "must_have_exited": must_be_exited,
            "vllm": str(args.vllm.resolve()), "phase_probe": str(directory / "hybrid_phase_probe.py"),
            "helper_dir": str(directory), "prompt_file": str(output / "prompts256x1024.json"),
            "preflight_inputs": str(args.preflight_inputs.resolve()), "quality_inputs": str(args.quality_inputs.resolve()),
            "warmup_rounds": 3, "measured_rounds": 3, "timed_sweeps": SWEEPS, "quality_arms": list(ARMS),
            "output_tokens": 128, "logical_requests": 256, "concurrency": 64, "client_waves": 4,
            "prefill_url": "http://127.0.0.1:8100", "decode_url": "http://127.0.0.1:8200",
            "gpu_util": args.gpu_util, "stop_unix": args.stop_unix, "boot_timeout": args.boot_timeout,
            "request_timeout": args.request_timeout, "quality_timeout": args.quality_timeout,
            "quality_max_tokens": 2048, "quality_concurrency": 64,
            "performance_estimand": "Mean of three fixed measured256x1024to128forced cohorts; treatment/stock within each fresh three-arm sweep; three sweeps descriptive only",
            "timing_scope": "Real NIXL producer/consumer and HTTP proxy-equivalent client, fixed8GPUs, rawcachedtokenAPIcoverageeachrequest, no probes/profilers/CUDAeventhooks",
            "native_metrics": "Default native metrics/logging kept uniformly enabled, promptcachedtokendetails mandatory; no synthetic cached-token override",
            "quality_scope": "Three fresh actual hybrid server pairs, all1319 GSM8K, canonical8shot IDs/stops/greedy/naturalEOS/max2048; one launch perarm"}
    plan["server_commands"] = {role: commands(plan, role) for role in ("prefill", "decode")}
    inputs = [Path(__file__).resolve(), *[directory / name for name in
              ("common_v2.py", "hybrid_timing.py", "hybrid_client.py", "hybrid_phase_probe.py", "audit_hybrid.py", "score_hybrid_gsm.py")],
              args.lifecycle_helper.resolve(), args.source_proof.resolve(), args.checkpoint_proof.resolve(),
              args.allocation_proof.resolve(), args.runtime_preflight.resolve(), args.preflight_inputs.resolve(),
              args.quality_inputs.resolve(), output / "prompts256x1024.json", args.vllm.resolve()]
    plan["files"] = {str(path): file_sha(path) for path in inputs}
    plan["files"].update({item["path"]: item["sha256"] for item in runtime["module_files"].values()})
    plan["files"].update({item["path"]: item["sha256"] for item in prerequisites})
    require(time.time() + 300 < args.stop_unix, "Compute cutoff too close")
    write_new(output / "plan.json", plan)
    return plan


class Driver:
    def __init__(self, path, mode):
        self.path, self.plan, self.mode = path, read(path), mode
        self.lib = lifecycle(self.plan["lifecycle_helper"])
        self.root = Path(self.plan["output"]) / mode
        self.root.mkdir(parents=True, exist_ok=False)
        self.lock = (Path(self.plan["output"]) / "exclusive-gpu-controller.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.children, self.identities, self.logs = {}, {}, {}
        self.state = {"complete": False, "cleanup_complete": False, "mode": mode, "plan_sha256": file_sha(path),
                      "controller_process": self.lib.process_record(os.getpid()), "launches": [], "started_unix_ns": time.time_ns()}
        self.save()

    def save(self):
        self.state["updated_unix_ns"] = time.time_ns()
        self.lib.save(self.root / "status.json", self.state)

    def validate(self):
        require(time.time() < self.plan["stop_unix"], "Absolute compute cutoff reached")
        for path, expected in self.plan["files"].items():
            require(file_sha(path) == expected, "Frozen input changed: " + path)
        require(self.lib.source_identity(self.plan["source_proof"], self.plan["source_root"], self.plan["site_root"]) == self.plan["source_identity"], "Source drift")
        require(self.lib.checkpoint_identity(self.plan["checkpoint_proof"], self.plan["model"]) == self.plan["checkpoint_identity"], "Model proof/stats drift")
        require(self.plan["server_commands"] == {role: commands(self.plan, role) for role in ("prefill", "decode")}
                and self.plan["timed_sweeps"] == SWEEPS, "Commands or fixed schedule drift")
        for identity in self.plan["must_have_exited"]:
            live = self.lib.process_record(identity["pid"])
            require(not live or str(live["start_ticks"]) != identity["start_ticks"] or live["state"] == "Z", "Prior controller still alive")

    def run_command(self, args, output, timeout):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        with output.open("xb") as stream:
            child = subprocess.Popen([sys.executable, *args], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            identity = self.lib.process_record(child.pid)
            require(identity, "Cannot bind helper process")
            try:
                until = min(time.time() + timeout, self.plan["stop_unix"])
                while child.poll() is None:
                    require(time.time() < until and all(p.poll() is None for p in self.children.values()), "Helper deadline/server failure")
                    self.save()
                    time.sleep(1)
                require(child.returncode == 0, "Helper failed; inspect " + str(output))
            finally:
                self.lib.stop_owned(child, identity)

    def start_pair(self, directory, arm, instrumented):
        self.validate()
        self.lib.gpu_idle()
        for port in (8100, 8200):
            with socket.socket() as sock:
                require(sock.connect_ex(("127.0.0.1", port)) != 0, "Hybrid HTTP port already occupied")
        for role in ("prefill", "decode"):
            probe = directory / (role + "-probe") if instrumented else None
            env = environment(self.plan, role, arm, probe)
            # This CPU-only import resolution binds both clean and instrumented
            # processes to the same pristine site path and plugin entry point.
            identity_plan = {**self.plan, "environment": {}, "max_num_seqs": 256}
            import_proof = self.lib.cpu_import_identity(identity_plan, env)
            stream = (directory / (role + "-server.log")).open("xb")
            self.logs[role] = stream
            child = subprocess.Popen(self.plan["server_commands"][role], env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.children[role], self.identities[role] = child, self.lib.process_record(child.pid)
            require(self.identities[role] and self.identities[role]["pgrp"] == child.pid, "Cannot bind owned server group")
            self.state["active_servers"] = self.identities
            write_new(directory / (role + "-launch.json"), {"role": role, "arm": arm if role == "prefill" else "stock",
                      "command": self.plan["server_commands"][role], "environment": {k: env[k] for k in sorted(env)
                         if k in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH", "UCX_RCACHE_MAX_UNRELEASED")
                         or k.startswith(("PLACEMENT_", "VLLM_", "HYBRID_"))},
                      "process": self.identities[role], "cpu_import_proof": import_proof})
            self.save()
        until = min(time.time() + self.plan["boot_timeout"], self.plan["stop_unix"])
        pending = {"prefill", "decode"}
        while pending:
            require(time.time() < until and all(p.poll() is None for p in self.children.values()), "Pair startup failure/deadline")
            for role in list(pending):
                try:
                    with urllib.request.urlopen(self.plan[role + "_url"] + "/health", timeout=2) as response:
                        if response.status == 200:
                            pending.remove(role)
                except OSError:
                    pass
            self.save()
            if pending:
                time.sleep(2)
        for role in ("prefill", "decode"):
            write_new(directory / (role + "-contract.json"), {"complete": True, "role": role,
                "arm": arm if role == "prefill" else "stock", "ep": 4, "source_commit": SOURCE_COMMIT,
                "checkpoint_revision": REVISION, "checkpoint_proof_sha256": file_sha(self.plan["checkpoint_proof"]),
                "source_proof_sha256": file_sha(self.plan["source_proof"]), "model_path": self.plan["model"],
                "kv_connector": "NixlConnector", "kv_load_failure_policy": "fail", "prefix_caching": False,
                "ready_scope": "Owned native server healthy; actual phase proof independently audited after closure",
                "process": self.identities[role], "allocation": self.plan["allocation"], "plan_sha256": file_sha(self.path)})

    def close_pair(self):
        result, errors = {}, []
        for role in list(self.children):
            try:
                result[role] = self.lib.stop_owned(self.children[role], self.identities[role])
                del self.children[role]
                self.identities.pop(role, None)
                self.logs.pop(role).close()
            except BaseException:
                errors.append(traceback.format_exc())
        self.state["active_servers"] = self.identities
        self.save()
        require(not errors, "Owned pair cleanup failed: " + str(errors))
        result["gpu_idle"] = self.lib.gpu_idle(wait=60)
        return result

    def client(self, directory, arm, mode):
        inputs = self.plan["preflight_inputs"] if mode == "smoke" else self.plan["quality_inputs"]
        out = directory / "client"
        args = [self.plan["helper_dir"] + "/hybrid_client.py", "--mode", mode, "--prepared", inputs,
                "--prepared-sha256", file_sha(inputs), "--prefill-contract", str(directory / "prefill-contract.json"),
                "--decode-contract", str(directory / "decode-contract.json"), "--prefill-url", self.plan["prefill_url"],
                "--decode-url", self.plan["decode_url"], "--arm", arm, "--launch-id", directory.name,
                "--max-tokens", "128" if mode == "smoke" else "2048", "--concurrency", "4" if mode == "smoke" else "64",
                "--max-requests", "4" if mode == "smoke" else "1319", "--stop-unix", str(self.plan["stop_unix"]),
                "--expected-split", "train" if mode == "smoke" else "test", "--output", str(out)]
        self.run_command(args, directory / "client.log", self.plan["quality_timeout"])
        return out, inputs

    def phase_audit(self, directory, out, inputs):
        audit = directory / "phase-audit.json"
        self.run_command([self.plan["helper_dir"] + "/audit_hybrid.py", "--prepared", inputs, "--result", str(out),
                          "--prefill-probe", str(directory / "prefill-probe"), "--decode-probe", str(directory / "decode-probe"),
                          "--output", str(audit)], directory / "audit.log", 300)
        self.run_command([self.plan["helper_dir"] + "/score_hybrid_gsm.py", "--prepared", inputs, "--result", str(out),
                          "--phase-audit", str(audit), "--output", str(directory / "gsm-score.json")], directory / "score.log", 300)
        return {"path": str(audit), "sha256": file_sha(audit)}

    def proof_gate(self):
        receipt = read(Path(self.plan["output"]) / "proof" / "receipt.json")
        require(receipt["complete"] and receipt["cleanup_complete"] and receipt["plan_sha256"] == file_sha(self.path)
                and len(receipt["launches"]) == 3 and {r["arm"] for r in receipt["launches"]} == set(ARMS), "All three actual handoff proofs required")
        for entry in receipt["launches"]:
            require(entry["complete"] and file_sha(entry["receipt"]) == entry["sha256"], "Handoff proof launch changed")
            launch = read(entry["receipt"])
            require(launch["arm"] == entry["arm"] and launch["complete"] is True
                    and launch["plan_sha256"] == file_sha(self.path), "Proof launch arm/plan differs")
            require(file_sha(launch["phase_audit"]["path"]) == launch["phase_audit"]["sha256"]
                    and read(launch["phase_audit"]["path"])["complete"], "Phase proof changed/incomplete")
            proof = read(launch["phase_audit"]["path"])
            for filename, expected in proof["phase_file_sha256"].items():
                require(file_sha(filename) == expected, "Underlying closed phase evidence changed")
            require(file_sha(Path(launch["directory"]) / "client" / "result.json") == proof["campaign_result_sha256"],
                    "Underlying handoff client result changed")
        self.state["proof_receipt_sha256"] = file_sha(Path(self.plan["output"]) / "proof" / "receipt.json")

    def launch(self, sweep, position, arm):
        self.validate()
        directory = self.root / f"sweep{sweep:02d}-position{position}-{arm}"
        directory.mkdir(exist_ok=False)
        entry = {"complete": False, "arm": arm, "sweep": sweep, "position": position,
                 "directory": str(directory), "plan_sha256": file_sha(self.path), "rounds": []}
        self.state["stage"] = directory.name
        attempt_index = len(self.state["launches"])
        self.state["launches"].append({"complete": False, "arm": arm, "sweep": sweep, "position": position,
                                       "directory": str(directory), "receipt_pending": True})
        self.save()
        failure = None
        try:
            self.start_pair(directory, arm, self.mode != "timed")
            if self.mode == "timed":
                prompts = read(self.plan["prompt_file"])
                for phase in ("warmup", "measured"):
                    for number in range(3):
                        require(time.time() + self.plan["request_timeout"] + 30 < self.plan["stop_unix"], "Insufficient request cutoff margin")
                        require(all(p.poll() is None for p in self.children.values()), "Native pair died before round")
                        round_plan = {**self.plan, "launch_id": directory.name}
                        value = run_round(round_plan, prompts, self.lib.parse_event, directory / f"{phase}{number:02d}", phase, number)
                        entry["rounds"].append({"phase": phase, "number": number, "elapsed_seconds": value["elapsed_seconds"],
                                                "path": str(directory / f"{phase}{number:02d}" / "round.json"),
                                                "sha256": file_sha(directory / f"{phase}{number:02d}" / "round.json")})
                        self.save()
                entry["mean_measured_seconds"] = statistics.fmean(r["elapsed_seconds"] for r in entry["rounds"] if r["phase"] == "measured")
            else:
                out, inputs = self.client(directory, arm, "smoke" if self.mode == "proof" else "generate")
            entry["workload_complete"] = True
        except BaseException as error:
            failure = error
            entry["workload_error"] = traceback.format_exc()
        finally:
            finalization_errors = []
            try:
                entry["closure"] = self.close_pair()
            except BaseException:
                finalization_errors.append({"stage": "owned_pair_closure", "error": traceback.format_exc()})
            for role in ("prefill", "decode"):
                path = directory / (role + "-server.log")
                try:
                    if path.exists():
                        log = path.read_text(errors="replace")
                        active = "placement-router: placement-aware routing active" in log
                        valid = active == (role == "prefill" and arm != "stock") and "placement-router: routing left unchanged" not in log
                        entry[role + "_activation_contract"] = valid
                        entry[role + "_server_log_sha256"] = file_sha(path)
                        if not valid:
                            finalization_errors.append({"stage": role + "_activation", "error": "Native activation/fallback log contract differs"})
                    else:
                        finalization_errors.append({"stage": role + "_log", "error": "Server never created its log"})
                except BaseException:
                    finalization_errors.append({"stage": role + "_log_inspection", "error": traceback.format_exc()})
            if failure is None and not finalization_errors and self.mode != "timed":
                try:
                    entry["phase_audit"] = self.phase_audit(directory, out, inputs)
                except BaseException as error:
                    failure = error
                    entry["phase_audit_error"] = traceback.format_exc()
            try:
                self.validate()
            except BaseException:
                finalization_errors.append({"stage": "post_source_model_validation", "error": traceback.format_exc()})
            entry["finalization_errors"] = finalization_errors
            entry["complete"] = failure is None and not finalization_errors and entry.get("workload_complete") is True
            entry["finished_unix_ns"] = time.time_ns()
            write_new(directory / "launch.json", entry)
            self.state["launches"][attempt_index] = {"complete": entry["complete"], "arm": arm, "sweep": sweep, "position": position,
                 "receipt": str(directory / "launch.json"), "sha256": file_sha(directory / "launch.json")}
            self.save()
        require(entry["complete"], "Hybrid launch failed; inspect preserved immutable receipt " + str(directory / "launch.json"))

    def run(self):
        try:
            self.validate()
            if self.mode != "proof":
                self.proof_gate()
            schedule = SWEEPS if self.mode == "timed" else [list(ARMS)]
            for sweep, order in enumerate(schedule):
                for position, arm in enumerate(order):
                    self.launch(sweep, position, arm)
            self.state["complete"] = True
        except BaseException:
            self.state["error"] = traceback.format_exc()
            raise
        finally:
            try:
                self.state["emergency_closure"] = self.close_pair()
                self.state["cleanup_complete"] = True
            except BaseException:
                self.state["cleanup_error"] = traceback.format_exc()
                self.state["complete"] = False
            self.state["finished_unix_ns"] = time.time_ns()
            self.save()
            write_new(self.root / "receipt.json", self.state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in ("output", "model", "production-root", "site-root", "source-proof", "checkpoint-proof", "allocation-proof",
                 "runtime-preflight", "lifecycle-helper", "vllm", "prefill-prompts", "preflight-inputs", "quality-inputs"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stop-unix", type=float, required=True)
    p.add_argument("--gpu-util", type=float, default=.88)
    p.add_argument("--boot-timeout", type=int, default=2400)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--quality-timeout", type=int, default=7200)
    p.add_argument("--prerequisite-receipts", type=Path, nargs="+", required=True)
    p.add_argument("--must-have-exited", nargs="+", required=True, help="Owned prior process identities as PID:start_ticks")
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--mode", choices=("proof", "timed", "quality"), required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
    else:
        def cancel(signum, frame):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            raise InterruptedError("Owned hybrid controller interrupted")
        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        Driver(args.plan, args.mode).run()
