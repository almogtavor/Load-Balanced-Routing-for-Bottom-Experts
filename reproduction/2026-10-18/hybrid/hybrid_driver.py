#!/usr/bin/env python3
"""Bounded owned-process controller for genuine EP4+EP4 NIXL PD experiments.

No cluster mutations. Three modes use separate fresh server pairs: instrumented
training native-phase proof with monolithic outputs retained diagnostically; nine clean timing pairs in three paired sweeps; and
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
from native_runner_contract import installed_sources
from native_request_id import installed_sources as request_id_sources
import recovery_provenance as recovery
import native_accounting as accounting
import prospective_gate as prospective

LIFECYCLE_SHA = "2b1832a73ec1d84ef16746c99a8ad94c82f4300ef7464fc4d2250e63b66b6db3"
ARMS = {"stock": (None, None), "keep6p8": (6, 8), "keep4p8": (4, 8)}
SWEEPS = [["stock", "keep6p8", "keep4p8"], ["keep6p8", "keep4p8", "stock"], ["keep4p8", "stock", "keep6p8"]]
FAILED_PRIORS = {"/work/results/profile-waiter-v1.json": (12347, "5683784"),
                 "/work/results/hybrid-outer-driver-v1.json": (35159, "6002531"),
                 "/work/results/prefill-profiles-v1/receipt.json": (35030, "5996801"),
                 "/work/results/prefill-profiles-v2/receipt.json": (36199, "6124979"),
                 "/work/results/prefill-profiles-v3/receipt.json": (37050, "6200412"),
                 "/work/results/hybrid-v2/proof/receipt.json": (40768, "6385036"),
                 "/work/results/hybrid-outer-driver-v3.json": (40764, "6384903")}
FAILED_SHAS = {"/work/results/profile-waiter-v1.json": "2a58d98169cd9d5aed45b0989103b04debb9df902d6b32a4956196bf52f7ae2d",
               "/work/results/hybrid-outer-driver-v1.json": "fd9384b46b531cf1623cbc77b8646df139039436b29489b9762aa38a2e35180a",
               "/work/results/prefill-profiles-v1/receipt.json": "8248cd8684a535ecd6b8446b329a02cdaabc19c9d17a496a1a8aedac22daa246",
               "/work/results/prefill-profiles-v2/receipt.json": "8a891170a632e82f79e3012435772d1bdb04b93d180c2df7d7590385aa6e889e",
               "/work/results/prefill-profiles-v3/receipt.json": "eb14d91b42ea63d336805ad7818c66f42888ab690cd05eb9b84eb41f23875ad3",
               "/work/results/hybrid-v2/proof/receipt.json":"878101ff1d7b2178a4d866d393976477dd2e325be71c2b6b28dc6b9cb1323496",
               "/work/results/hybrid-outer-driver-v3.json":"98e67057a50b4d92665b9389beb9964b872f4830c47bf7cdde92956fb63ad883"}


def failed_prior_gate(paths, lib, recovery_binding):
    require(recovery_binding["original_pod"] == recovery.OLD and recovery_binding["prior_controller_closure"] == recovery.CLOSURE, "Archived process namespace closure required")
    require({recovery.original_path(p.resolve()) for p in paths} == set(FAILED_PRIORS), "Retain exactly all seven failed observer/outer receipts")
    result = []
    for path in paths:
        value = read(path)
        original = recovery.original_path(path.resolve())
        require(file_sha(path) == FAILED_SHAS[original], "Retained failed V1 receipt changed")
        identity = value.get("controller_process", value.get("process"))
        pid, ticks = FAILED_PRIORS[original]
        require(value.get("complete") is False and value.get("cleanup_complete") is True
                and (value.get("finished_unix") or value.get("finished_unix_ns"))
                and identity and identity["pid"] == identity["pgrp"] == pid and str(identity["start_ticks"]) == ticks
                and (value.get("cleanup_errors") == [] or (original=="/work/results/hybrid-v2/proof/receipt.json" and "cleanup_errors" not in value))
                and not value.get("cleanup_error")
                and not value.get("active_child"), "Failed prior campaign lacks bound terminal closure")
        if original=="/work/results/hybrid-v2/proof/receipt.json":
            require(value["mode"]=="proof" and not value["active_servers"] and len(value["launches"])==1
                    and value["launches"][0]["complete"] is False
                    and value["emergency_closure"]["gpu_idle"]["returncode"]==0
                    and not value["emergency_closure"]["gpu_idle"]["stdout"].strip(), "Original failed native-ID proof did not close")
            old=value["launches"][0]
            require(old["receipt"]=="/work/results/hybrid-v2/proof/sweep00-position0-stock/launch.json"
                    and old["sha256"]=="f38e7d01162c32c0ed678969a97e514abc07ac7ab910bbb12711e7d46316745f"
                    and file_sha(recovery.prior_path(old["receipt"]))==old["sha256"], "Original failed stock launch changed")
            failed=read(recovery.prior_path(old["receipt"]))
            require(failed["complete"] is False and failed["finalization_errors"]==[]
                    and all(failed["closure"][role]["remaining_members"]==[] for role in ("prefill","decode")),
                    "Original failed stock roles not closed")
        result.append({"path": str(path.resolve()), "original_path": original, "sha256": file_sha(path),
                       "closure_scope": recovery.CLOSURE,
                       "controller_process": {"pid": pid, "start_ticks": ticks}})
    return result


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
    recovery_binding = recovery.verify(args.recovery_proof, args.recovery_proof_sha256, allocation)
    admission = prospective.verify(lib)
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
    native_sources = installed_sources(Path(runtime["module_files"]["vllm.config.kv_transfer"]["path"]).parents[1])
    native_accounting_sources = accounting.installed_sources(native_sources["root"])
    failed_priors = failed_prior_gate(args.failed_prior_receipts, lib, recovery_binding)
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
    recovered_prior_controllers = []
    perf_receipts = [(path, read(path)) for path in args.prerequisite_receipts if len(read(path).get("launches", [])) == 36]
    profile_receipts = [(path, read(path)) for path in args.prerequisite_receipts if "main_guard" in read(path)]
    require(len(perf_receipts) == len(profile_receipts) == 1, "Exactly one completed36launch primary timing and one linked profile receipt required")
    perf_path, perf_receipt = perf_receipts[0]
    profile_path, profile_receipt = profile_receipts[0]
    profile_plan_path = profile_path.parent / "plan.json"
    require(read(profile_plan_path)["schema"] == "headline-prefill-profiles-v4"
            and file_sha(profile_plan_path) == profile_receipt["plan_sha256"], "Require corrected actual-V2 profile plan")
    guard = profile_receipt["main_guard"]
    require(guard["receipt_sha256"] == file_sha(perf_path) and guard["complete_launches"] == 36
            and guard["controller_original_exited"] is True and guard["main_lock_held"] is True
            and len(profile_receipt["launches"]) == 3 and all(x["complete"] for x in profile_receipt["launches"]),
            "Profile receipt does not close the same completed timing campaign")
    for _, receipt in (perf_receipts + profile_receipts):
        identity = receipt["controller_process"]
        recovered_prior_controllers.append({"pid": int(identity["pid"]), "start_ticks": str(identity["start_ticks"]), "closure_scope": recovery.CLOSURE})
    require(not args.must_have_exited, "Old process IDs cannot be checked in the replacement namespace")
    for item in prerequisites:
        item["original_path"] = recovery.original_path(item["path"])
        item["closure_scope"] = recovery.CLOSURE
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "prompts256x1024.json", prompts)
    plan = {"schema": "real-nixl-hybrid-v6", "output": str(output), "source_commit": SOURCE_COMMIT,
            "model_id": MODEL_ID, "checkpoint_revision": REVISION, "model": str(args.model.resolve()),
            "source_root": str(args.production_root.resolve()), "site_root": str(args.site_root.resolve()),
            "source_proof": str(args.source_proof.resolve()), "checkpoint_proof": str(args.checkpoint_proof.resolve()),
            "source_identity": source, "checkpoint_identity": checkpoint,
            "allocation": allocation, "allocation_proof": str(args.allocation_proof.resolve()),
            "runtime_preflight": str(args.runtime_preflight.resolve()), "lifecycle_helper": str(args.lifecycle_helper.resolve()),
            "prerequisite_receipts": prerequisites, "must_have_exited": [],
            "recovery_binding": recovery_binding, "recovered_prior_controllers": recovered_prior_controllers,
            "hardware_scope": "Matched same-host eight-H200 hybrid experiment; separate from archived H100 prefill measurements",
            "failed_prior_receipts": failed_priors, "native_runner_sources": native_sources,
            "prospective_admission": admission, "prospective_policy": prospective.POLICY,
            "native_accounting_sources": native_accounting_sources,
            "request_id_sources": request_id_sources(Path(native_sources["files"]["config/vllm.py"]).parents[1]) ,
            "runner_policy": "Observe actual native default V2; no model-runner environment override, all roles/arms identical",
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
            "timing_scope": "Real NIXL producer/consumer and HTTP proxy-equivalent client, fixed8GPUs, sourcecorrectnativecached0+offclockNIXLcountbytecoverage, no probes/profilers/CUDAeventhooks",
            "native_metrics": "Default native metrics/logging uniformly enabled; cached0producerprefixhits; deterministic offclock<=5s count/bytes drain, allpolls retained",
            "quality_scope": "Three fresh actual hybrid server pairs, all1319 GSM8K, canonical8shot IDs/stops/greedy/naturalEOS/max2048; one launch perarm"}
    plan["server_commands"] = {role: commands(plan, role) for role in ("prefill", "decode")}
    inputs = [Path(__file__).resolve(), *[directory / name for name in
              ("native_accounting.py", "audit_native_metrics.py", "prospective_gate.py", "audit_hybrid_timing_raw_v6.py", "recovery_provenance.py", "common_v2.py", "hybrid_timing.py", "hybrid_client.py", "hybrid_phase_probe.py", "audit_hybrid.py", "score_hybrid_gsm.py", "native_runner_contract.py", "native_request_id.py")],
              args.lifecycle_helper.resolve(), args.source_proof.resolve(), args.checkpoint_proof.resolve(),
              args.allocation_proof.resolve(), args.recovery_proof.resolve(), args.runtime_preflight.resolve(), args.preflight_inputs.resolve(),
              args.quality_inputs.resolve(), output / "prompts256x1024.json", args.vllm.resolve()]
    plan["files"] = {str(path): file_sha(path) for path in inputs}
    plan["files"].update(prospective.FILES)
    plan["files"].update({path:native_accounting_sources["sha256"][name] for name,path in native_accounting_sources["files"].items()})
    plan["files"].update({path:plan["request_id_sources"]["sha256"][name] for name,path in plan["request_id_sources"]["files"].items()})
    plan["files"][str(recovery.prior_path("/work/results/hybrid-v2/proof/sweep00-position0-stock/launch.json"))]="f38e7d01162c32c0ed678969a97e514abc07ac7ab910bbb12711e7d46316745f"
    plan["files"].update({item["path"]: item["sha256"] for item in runtime["module_files"].values()})
    plan["files"].update({item["path"]: item["sha256"] for item in prerequisites})
    plan["files"].update({item["path"]: item["sha256"] for item in failed_priors})
    plan["files"][str(profile_plan_path)] = file_sha(profile_plan_path)
    plan["files"].update({path: native_sources["sha256"][name] for name, path in native_sources["files"].items()})
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
                and self.plan["timed_sweeps"] == SWEEPS and self.plan["schema"] == "real-nixl-hybrid-v6", "Commands or fixed schedule drift")
        require(prospective.verify(self.lib)==self.plan["prospective_admission"] and self.plan["prospective_policy"]==prospective.POLICY, "Prospective source/diagnostic policy drift")
        require(accounting.installed_sources(self.plan["native_accounting_sources"]["root"])==self.plan["native_accounting_sources"], "Native accounting source drift")
        require(installed_sources(self.plan["native_runner_sources"]["root"]) == self.plan["native_runner_sources"],
                "Native V2 runner source drift")
        require(failed_prior_gate([Path(r["path"]) for r in self.plan["failed_prior_receipts"]], self.lib, self.plan["recovery_binding"])
                == self.plan["failed_prior_receipts"], "Retained failed campaign drift")
        require(not self.plan["must_have_exited"], "No archived PID checks in replacement namespace")
        binding = self.plan["recovery_binding"]
        require(recovery.verify(binding["proof_path"], binding["proof_sha256"], read(self.plan["allocation_proof"])) == binding,
                "Frozen recovery allocation/closure proof changed")

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
                "native_runner_sources": self.plan["native_runner_sources"],
                "runner_contract_sha256": file_sha(Path(self.plan["helper_dir"]) / "native_runner_contract.py"),
                "request_id_sources":self.plan["request_id_sources"],
                "request_id_contract_sha256":file_sha(Path(self.plan["helper_dir"]) / "native_request_id.py"),
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
                entry["owned_group_members_before_cleanup"] = {
                    role: self.lib.group_members(identity["pgrp"]) for role, identity in self.identities.items()}
            except BaseException:
                finalization_errors.append({"stage": "preclosure_owned_groups", "error": traceback.format_exc()})
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
                    phase = read(entry["phase_audit"]["path"])
                    require(phase["actual_v2_runner_on_all8_workers"] is True
                            and phase["decoder_exact_transferred_prefix_then_native_last_prompt_token"] is True
                            and phase["exact_native_external_internal_mapping"] is True,
                            "Actual V2 last-token handoff proof missing")
                    for role in ("prefill", "decode"):
                        members = entry["owned_group_members_before_cleanup"][role]
                        launch = read(directory / (role + "-launch.json"))
                        workers = phase["worker_identity"][role]
                        require(len({r["pid"] for r in members}) == len(members)
                                and all(r["pgrp"] == launch["process"]["pgrp"] and r["state"] != "Z"
                                        and str(r["start_ticks"]).isdigit() for r in members)
                                and {r["pid"] for r in workers} <= {r["pid"] for r in members}
                                and set(phase["request_id_mapping_pids"][role]) <= {r["pid"] for r in members},
                                "Actual V2 proof workers were outside the owned role process group")
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
                 "recovery-proof", "runtime-preflight", "lifecycle-helper", "vllm", "prefill-prompts", "preflight-inputs", "quality-inputs"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--recovery-proof-sha256", required=True)
    p.add_argument("--stop-unix", type=float, required=True)
    p.add_argument("--gpu-util", type=float, default=.88)
    p.add_argument("--boot-timeout", type=int, default=2400)
    p.add_argument("--request-timeout", type=int, default=600)
    p.add_argument("--quality-timeout", type=int, default=7200)
    p.add_argument("--prerequisite-receipts", type=Path, nargs="+", required=True)
    p.add_argument("--failed-prior-receipts", type=Path, nargs="+", required=True)
    p.add_argument("--must-have-exited", nargs="*", default=[], help="Must remain empty: archived prior PIDs belong to deleted original pod")
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
