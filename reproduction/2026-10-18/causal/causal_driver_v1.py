#!/usr/bin/env python3
"""Secondary-host causal campaign after all nine quality launches are closed.

New additive controller. Imports the already frozen hybrid and performance
lifecycle code, never edits v1 helpers. Six fresh training cells close and audit
before a separate held-out freeze is materialized; three fresh held-out cells
then use the serial target-absent reference. No cluster operations.
"""
from __future__ import annotations
import argparse
import importlib.util
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
from common_v2 import (REVISION, SOURCE_COMMIT, encoded, file_sha, read, require, write_new)
import hybrid_driver as hybrid
import causal_probe

FROZEN_HYBRID_DRIVER = "18e2db041543d60fda2dde92bcee54ea9750d05a2d6a6fc5915b3ab1c0ed4271"
FROZEN_CAUSAL = "2d67763c5fffb8154bd95dd10056528943398370313153566f064c13db1791de"
FROZEN_HELPER_MANIFEST = "0b5032b2f4c37204ca2952ac91bcc10201c6217bffc484e2d78d2eb3d6c5d7c0"
TRAINING_CELLS = [(arm, budget) for budget in (16384, 1024) for arm in ("stock", "keep6p8", "keep4p8")]


def import_heldout(path):
    spec = importlib.util.spec_from_file_location("approved_heldout_causal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prior_gate(path, lib, process):
    receipt = read(path)
    require(receipt.get("complete") is True and receipt.get("execution_complete") is True
            and receipt.get("cleanup_complete") is True and not receipt.get("cleanup_errors"),
            "Nine-launch quality campaign is not complete and cleaned")
    rows = receipt["launches"]
    expected = {(replicate, arm) for replicate in (1, 2, 3) for arm in ("stock", "keep6p8", "keep4p8")}
    require(len(rows) == 9 and {(r["replicate"], r["arm"]) for r in rows} == expected
            and all(r["complete"] is True and r["ep"] == 4 for r in rows), "Quality campaign did not close all nine expected cells")
    live = lib.process_record(process["pid"])
    require(not live or str(live["start_ticks"]) != process["start_ticks"] or live["state"] == "Z", "Original quality controller still alive")
    return receipt


def command(plan, budget):
    return [plan["vllm"], "serve", plan["model"], "--served-model-name", "task-model", "--host", "127.0.0.1",
            "--port", "8040", "--data-parallel-rpc-port", "29690", "--data-parallel-size", "4",
            "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1", "--enable-expert-parallel",
            "--all2all-backend", "deepep_high_throughput", "--dtype", "bfloat16", "--expert-placement-strategy", "linear",
            '--compilation-config={"cudagraph_mode":"NONE"}', "--max-model-len", "8192", "--seed", "0",
            "--max-num-batched-tokens", str(budget), "--max-num-seqs", "256", "--enable-chunked-prefill",
            "--no-enable-prefix-caching", "--logprobs-mode", "raw_logprobs", "--gpu-memory-utilization", ".88"]


def make_plan(args):
    root = Path(__file__).resolve().parent
    require(file_sha(root / "hybrid_driver.py") == FROZEN_HYBRID_DRIVER
            and file_sha(root / "causal_probe.py") == FROZEN_CAUSAL
            and file_sha(root / "helper-freeze-v1.json") == FROZEN_HELPER_MANIFEST, "Frozen v1 dependency changed")
    lib = hybrid.lifecycle(args.lifecycle_helper)
    process = {"pid": 167, "start_ticks": "5555001"}
    prior_gate(args.quality_receipt, lib, process)
    allocation = read(args.allocation_proof)
    require(allocation.get("verified_live") and allocation.get("namespace") == "itay" and allocation.get("gpu_count") == 8
            and "H100" in allocation.get("gpu_name", ""), "Whole eight-H100 quality-host allocation proof required")
    deployed = read(args.source_proof).get("pod_identity", {})
    require(all(deployed.get(k) == allocation.get(k) for k in ("namespace", "pod", "uid", "node", "image")), "Wrong quality pod source deployment")
    require(file_sha(args.preflight_inputs) == "183888c4a3d0402ab97806d453449dbf740e445901b80ee3d5e156ffc7dd7017"
            and file_sha(args.full_inputs) == "6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b", "Canonical task inputs changed")
    source = lib.source_identity(args.source_proof, args.production_root, args.site_root)
    checkpoint = lib.checkpoint_identity(args.checkpoint_proof, args.model)
    require(time.time() + 300 < args.stop_unix, "Compute cutoff too close")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    causal_probe.prepare(SimpleNamespace(prior_preflight=args.preflight_inputs,
        prior_preflight_sha256=file_sha(args.preflight_inputs), checkpoint_proof=args.checkpoint_proof,
        source_proof=args.source_proof, production_root=args.production_root, output=output / "training-prepared"))
    plan = {"schema": "secondary-causal-v1", "output": str(output), "model": str(args.model.resolve()),
            "source_root": str(args.production_root.resolve()), "site_root": str(args.site_root.resolve()),
            "source_proof": str(args.source_proof.resolve()), "checkpoint_proof": str(args.checkpoint_proof.resolve()),
            "source_identity": source, "checkpoint_identity": checkpoint, "allocation": allocation,
            "quality_receipt": str(args.quality_receipt.resolve()), "quality_controller": process,
            "lifecycle_helper": str(args.lifecycle_helper.resolve()), "helper_dir": str(root), "vllm": str(args.vllm.resolve()),
            "preflight_inputs": str(args.preflight_inputs.resolve()), "full_inputs": str(args.full_inputs.resolve()),
            "training_prepared": str(output / "training-prepared"), "training_cells": TRAINING_CELLS,
            "training_requests_per_cell": 493, "heldout_helper": str(args.heldout_helper.resolve()),
            "heldout_review": str(args.heldout_review.resolve()), "heldout_prepared": str(output / "heldout-prepared"),
            "heldout_arms": ["stock", "keep6p8", "keep4p8"], "url": "http://127.0.0.1:8040",
            "stop_unix": args.stop_unix, "boot_timeout": 2400, "training_timeout": 3600, "heldout_timeout": 18000,
            "serial_policy": "One completed HTTP request at a time; no other live evaluation requests; target/future never in any concurrent modelinput",
            "phases": "training6complete+rawaudit+ownedclosure -> separate heldout freeze -> heldout3freshservers",
            "GPU_policy": "Whole8GPUhost idle before each fresh server; only GPU0..3 allocated to EP4; no other evaluation service runs"}
    files = [Path(__file__).resolve(), root / "helper-freeze-v1.json", *[root / name for name in
             ("common_v2.py", "common_v3.py", "hybrid_driver.py", "hybrid_timing.py", "hybrid_client.py", "causal_probe.py", "causal_contract.py", "causal_worker_probe.py")],
             args.lifecycle_helper.resolve(), args.allocation_proof.resolve(), args.source_proof.resolve(), args.checkpoint_proof.resolve(),
             args.quality_receipt.resolve(), args.preflight_inputs.resolve(), args.full_inputs.resolve(), args.vllm.resolve(),
             args.heldout_helper.resolve(), args.heldout_review.resolve(), output / "training-prepared" / "protocol.json"]
    plan["files"] = {str(p): file_sha(p) for p in files}
    write_new(output / "plan.json", plan)


class CausalDriver(hybrid.Driver):
    def validate(self):
        require(time.time() < self.plan["stop_unix"], "Absolute causal compute cutoff reached")
        for path, expected in self.plan["files"].items():
            require(file_sha(path) == expected, "Frozen causal input changed: " + path)
        require(self.lib.source_identity(self.plan["source_proof"], self.plan["source_root"], self.plan["site_root"]) == self.plan["source_identity"], "Source drift")
        require(self.lib.checkpoint_identity(self.plan["checkpoint_proof"], self.plan["model"]) == self.plan["checkpoint_identity"], "Checkpoint drift")
        prior_gate(self.plan["quality_receipt"], self.lib, self.plan["quality_controller"])
        require([list(v) for v in TRAINING_CELLS] == self.plan["training_cells"], "Training cell schedule changed")

    def start_server(self, directory, arm, budget):
        self.validate()
        self.lib.gpu_idle()
        with socket.socket() as sock:
            require(sock.connect_ex(("127.0.0.1", 8040)) != 0, "Dedicated causal HTTP port occupied")
        env = hybrid.environment(self.plan, "prefill", arm)
        for key in ("VLLM_NIXL_SIDE_CHANNEL_HOST", "VLLM_NIXL_SIDE_CHANNEL_PORT", "UCX_RCACHE_MAX_UNRELEASED"):
            env.pop(key, None)
        env.update(PLACEMENT_PROBE=self.plan["helper_dir"] + "/causal_worker_probe.py", CAUSAL_PROBE_DIR=str(directory / "probe"),
                   CAUSAL_PROBE_ARM=arm, CAUSAL_SOURCE_PROOF=self.plan["source_proof"], CAUSAL_EXPECTED_BUDGET=str(budget))
        import_identity = self.lib.cpu_import_identity(self.plan, env)
        stream = (directory / "server.log").open("xb")
        self.logs["causal"] = stream
        child = subprocess.Popen(command(self.plan, budget), env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children["causal"], self.identities["causal"] = child, self.lib.process_record(child.pid)
        require(self.identities["causal"] and self.identities["causal"]["pgrp"] == child.pid, "Cannot bind owned causal server")
        self.state["active_servers"] = self.identities
        write_new(directory / "server-launch.json", {"command": command(self.plan, budget), "process": self.identities["causal"],
                  "environment": {k: env[k] for k in sorted(env) if k.startswith(("PLACEMENT_", "CAUSAL_", "VLLM_"))
                                  or k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "OMP_NUM_THREADS", "NVSHMEM_QP_DEPTH")},
                  "cpu_import_identity": import_identity, "allocation": self.plan["allocation"]})
        self.save()
        deadline = min(time.time() + self.plan["boot_timeout"], self.plan["stop_unix"])
        while True:
            require(child.poll() is None and time.time() < deadline, "Causal server startup failure/deadline")
            try:
                with urllib.request.urlopen(self.plan["url"] + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            self.save()
            time.sleep(2)

    def training_gate(self):
        receipt_path = Path(self.plan["output"]) / "training" / "receipt.json"
        receipt = read(receipt_path)
        require(receipt.get("complete") and receipt.get("cleanup_complete") and receipt["plan_sha256"] == file_sha(self.path)
                and len(receipt["launches"]) == 6
                and {(r["arm"], r["budget"]) for r in receipt["launches"]} == set(TRAINING_CELLS),
                "All six training cells must close before held-out freeze")
        analyses = []
        for row in receipt["launches"]:
            require(row["complete"] and file_sha(row["receipt"]) == row["sha256"], "Training launch proof changed")
            launch = read(row["receipt"])
            require(launch["complete"] and launch["arm"] == row["arm"] and launch["budget"] == row["budget"]
                    and launch["plan_sha256"] == file_sha(self.path), "Training launch arm/budget/plan differs")
            path = Path(launch["analysis"]["path"])
            require(file_sha(path) == launch["analysis"]["sha256"] and read(path)["complete"], "Training analysis incomplete/changed")
            require(file_sha(Path(launch["directory"]) / "client" / "result.json") == launch["client_result_sha256"]
                    and file_sha(Path(launch["directory"]) / "contract.json") == launch["native_contract_sha256"],
                    "Underlying training client/native contract changed")
            for filename, expected in launch["probe_files_sha256"].items():
                require(file_sha(filename) == expected, "Underlying native training proof changed")
            analyses.append(path)
        return receipt_path, analyses

    def freeze_heldout(self):
        self.validate()
        receipt, analyses = self.training_gate()
        require(not self.children, "No model may be active while locking the held-out protocol")
        self.lib.gpu_idle()
        review = read(self.plan["heldout_review"])
        require(review.get("complete") is True and review.get("heldout_ready") is True
                and review["heldout_helper_sha256"] == file_sha(self.plan["heldout_helper"]), "Independent held-out helper review required")
        module = import_heldout(self.plan["heldout_helper"])
        module.prepare(SimpleNamespace(full_requests=Path(self.plan["full_inputs"]), full_requests_sha256=file_sha(self.plan["full_inputs"]),
            training_analyses=analyses, checkpoint_proof_sha256=file_sha(self.plan["checkpoint_proof"]),
            source_proof_sha256=file_sha(self.plan["source_proof"]), output=Path(self.plan["heldout_prepared"])))
        write_new(Path(self.plan["output"]) / "heldout-freeze-receipt.json", {"complete": True,
                  "training_receipt_sha256": file_sha(receipt), "protocol_sha256": file_sha(Path(self.plan["heldout_prepared"]) / "protocol.json"),
                  "selected_requests_sha256": file_sha(Path(self.plan["heldout_prepared"]) / "selected-requests.json"),
                  "plan_sha256": file_sha(self.path), "review_sha256": file_sha(self.plan["heldout_review"]),
                  "created_unix_ns": time.time_ns(), "no_heldout_model_queries_before_freeze": True})

    def launch_cell(self, index, arm, budget):
        directory = self.root / f"cell{index:02d}-{arm}-budget{budget}"
        directory.mkdir(exist_ok=False)
        attempt = len(self.state["launches"])
        self.state["launches"].append({"complete": False, "arm": arm, "budget": budget, "directory": str(directory)})
        self.state["stage"] = directory.name
        self.save()
        entry = {"complete": False, "arm": arm, "budget": budget, "mode": self.mode, "directory": str(directory),
                 "plan_sha256": file_sha(self.path), "finalization_errors": []}
        failure = None
        try:
            self.start_server(directory, arm, budget)
            tools = Path(self.plan["helper_dir"])
            common = ["--prepared", self.plan["preflight_inputs"], "--prepared-sha256", file_sha(self.plan["preflight_inputs"])]
            self.run_command([str(tools / "causal_contract.py"), "warmup", *common, "--url", self.plan["url"],
                "--launch-id", directory.name, "--stop-unix", str(self.plan["stop_unix"]), "--output", str(directory / "warmup")], directory / "warmup.log", 1200)
            self.run_command([str(tools / "causal_contract.py"), "build", "--checkpoint-proof", self.plan["checkpoint_proof"],
                "--source-proof", self.plan["source_proof"], "--probe-dir", str(directory / "probe"), "--arm", arm,
                "--budget", str(budget), "--output", str(directory / "contract.json")], directory / "contract.log", 300)
            helper = str(tools / "causal_probe.py") if self.mode == "training" else self.plan["heldout_helper"]
            prepared = self.plan["training_prepared"] if self.mode == "training" else self.plan["heldout_prepared"]
            count = 493 if self.mode == "training" else read(Path(prepared) / "protocol.json")["total_requests"]
            args = [helper, "run", "--prepared", prepared, "--server-contract", str(directory / "contract.json"),
                    "--url", self.plan["url"], "--launch-id", directory.name, "--arm", arm, "--max-requests", str(count),
                    "--stop-unix", str(self.plan["stop_unix"]), "--output", str(directory / "client")]
            args += ["--chunking-label", f"budget{budget}"] if self.mode == "training" else ["--exclusive-server-ack", "no-other-live-evaluation-requests"]
            self.run_command(args, directory / "client.log", self.plan["training_timeout"] if self.mode == "training" else self.plan["heldout_timeout"])
            entry["client_result_sha256"] = file_sha(directory / "client" / "result.json")
            entry["workload_complete"] = True
        except BaseException as error:
            failure, entry["workload_error"] = error, traceback.format_exc()
        finally:
            try:
                entry["closure"] = self.close_pair()
            except BaseException:
                entry["finalization_errors"].append({"step": "owned_closure", "error": traceback.format_exc()})
            try:
                log = (directory / "server.log").read_text(errors="replace")
                require(("placement-router: placement-aware routing active" in log) == (arm != "stock")
                        and "placement-router: routing left unchanged" not in log, "Routing activation/fallback differs")
                entry["server_log_sha256"] = file_sha(directory / "server.log")
                self.validate()
            except BaseException:
                entry["finalization_errors"].append({"step": "source_activation_validation", "error": traceback.format_exc()})
            if failure is None and not entry["finalization_errors"]:
                try:
                    self.run_command([helper, "analyze", "--prepared", prepared, "--result", str(directory / "client"),
                        "--output", str(directory / "analysis.json")], directory / "analysis.log", 3600)
                    entry["analysis"] = {"path": str(directory / "analysis.json"), "sha256": file_sha(directory / "analysis.json")}
                    entry["native_contract_sha256"] = file_sha(directory / "contract.json")
                    entry["probe_files_sha256"] = {str(p): file_sha(p) for p in sorted((directory / "probe").glob("*.json"))}
                except BaseException as error:
                    failure, entry["analysis_error"] = error, traceback.format_exc()
            entry["complete"] = failure is None and not entry["finalization_errors"] and entry.get("workload_complete") is True
            entry["finished_unix_ns"] = time.time_ns()
            write_new(directory / "launch.json", entry)
            self.state["launches"][attempt] = {"complete": entry["complete"], "arm": arm, "budget": budget,
                "receipt": str(directory / "launch.json"), "sha256": file_sha(directory / "launch.json")}
            self.save()
        require(entry["complete"], "Causal cell failed; preserve immutable launch receipt, no replacement")

    def run(self):
        try:
            self.validate()
            if self.mode == "heldout":
                self.training_gate()
                freeze = read(Path(self.plan["output"]) / "heldout-freeze-receipt.json")
                require(freeze["complete"] and freeze["plan_sha256"] == file_sha(self.path)
                        and freeze["protocol_sha256"] == file_sha(Path(self.plan["heldout_prepared"]) / "protocol.json"), "Post-training held-out freeze missing/drifted")
            cells = TRAINING_CELLS if self.mode == "training" else [(arm, 16384) for arm in self.plan["heldout_arms"]]
            for index, (arm, budget) in enumerate(cells):
                self.launch_cell(index, arm, budget)
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


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    for name in ("output", "model", "production-root", "site-root", "source-proof", "checkpoint-proof", "allocation-proof",
                 "lifecycle-helper", "vllm", "preflight-inputs", "full-inputs", "quality-receipt", "heldout-helper", "heldout-review"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stop-unix", type=float, default=1792336755)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--mode", choices=("training", "heldout"), required=True)
    p = sub.add_parser("freeze-heldout")
    p.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args)
        return
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise InterruptedError("Owned causal controller interrupted")
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    if args.command == "freeze-heldout":
        # Constructor creates a distinct administrative receipt directory and
        # takes the same exclusive lock, without starting a model.
        driver = CausalDriver(args.plan, "freeze-heldout")
        driver.freeze_heldout()
        driver.state.update(complete=True, cleanup_complete=True)
        driver.save()
        write_new(driver.root / "receipt.json", driver.state)
    else:
        CausalDriver(args.plan, args.mode).run()


if __name__ == "__main__":
    main()
