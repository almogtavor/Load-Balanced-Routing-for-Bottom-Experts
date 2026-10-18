#!/usr/bin/env python3
"""CPU-only manifest builder for one isolated experimental server, no install.

Root owns resource lifecycle and task clients. Every worker refuses startup until
the actual installed private overlay matches the complete same-source CUDA gate.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from stochastic_tail.plugin import MODEL_ID, REVISION, PUBLIC, PUBLIC_COMMIT, private_hashes, read_plan, NATIVE_FILES

ARMS = {"stock": ("stock", None, None), "par6": ("par", 6, 8.0), "par4": ("par", 4, 8.0),
        "plain6": ("par", 6, 0.0), "plain4": ("par", 4, 0.0),
        "uniform6": ("uniform_hot", 6, None), "uniform4": ("uniform_hot", 4, None),
        "weighted6": ("weighted_hot", 6, None), "weighted4": ("weighted_hot", 4, None)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def validate_gate(path):
    """Bind GPU proof to this frozen helper and its complete, fixed case matrix."""
    gate = json.loads(Path(path).read_text())
    need(gate.get("schema") == "stochastic-tail-kernel-gate-v1" and gate.get("complete") is True
         and gate.get("cuda_required") is True and "H100" in gate.get("gpu", "")
         and gate.get("private_source_sha256") == private_hashes()
         and gate.get("gate_helper_sha256") == sha(Path(__file__).with_name("gpu_gate.py")),
         "GPU gate source/helper/CUDA/runtime identity differs")
    need(gate.get("weighted_key_tolerance") == 2e-5 and gate.get("weighted_tail_reference_regret_tolerance") == 4e-5,
         "CUDA numeric tolerances changed")
    checks = {"native_head", "distinct_experts", "dtype_and_sentinel", "original_weights",
              "ragged_and_padding", "rng_words_and_repeat", "fanout", "ties_and_range"}
    need(set(gate.get("checks", {})) == checks and all(v is True for v in gate["checks"].values()),
         "CUDA checks missing/incomplete")
    expected = []
    modes = ("uniform_hot", "weighted_hot", "uniform_range")
    for dtype in ("torch.float32", "torch.bfloat16", "torch.float16"):
        for ranks in (4, 8):
            for keep in (4, 6):
                for seed in (0, 1, 2):
                    for mode in modes:
                        expected.append((17, keep, ranks, mode, seed, dtype, "torch.int64", ranks-1, 13, 47))
    for tokens in (1, 2, 15, 16, 33, 257, 2051):
        for mode in modes:
            for dtype in ("torch.int32", "torch.int64", "torch.uint32"):
                expected.append((tokens, 4, 4, mode, 0, "torch.float32", dtype, 0, 0, 0))
    for keep in (4, 6, 8):
        for mode in modes:
            expected.append((33, keep, 4, mode, 0, "torch.float32", "torch.int64", 0, 13, 47))
    fields = ("T", "keep", "ranks", "mode", "seed", "logits_dtype", "indices_dtype", "physical_rng_rank", "call", "layer")
    observed = Counter(tuple(c[k] for k in fields) for c in gate.get("cases", []))
    need(observed == Counter(expected) and len(gate["cases"]) == 180, "CUDA case inventory differs from frozen180 cases")
    need(all(0 <= c["max_weighted_key_error"] <= 2e-5 and 0 <= c["weighted_near_tie_tail_rows"] <= c["T"]
             for c in gate["cases"]), "CUDA case diagnostics invalid")
    return gate


def build(args):
    need(not args.output.exists(), "Choose a new plan filename; do not overwrite")
    validate_gate(args.gpu_gate)
    deployment = json.loads(args.public_source_proof.read_text())
    need(deployment["source_commit"] == PUBLIC_COMMIT
         and Path(deployment["site_root"]).resolve() == args.public_site.resolve(), "Public lineage deployment/site differs")
    need(all(sha(args.public_site / name) == expected for name, expected in PUBLIC.items()), "Pristine public lineage drift")
    entries = [(e.name, e.value) for d in importlib.metadata.distributions(path=[str(args.private_site)])
               for e in d.entry_points if e.group == "vllm.general_plugins"]
    need(entries == [("stochastic_tail_experiment", "stochastic_tail.plugin:register")], "Actual isolated entrypoint metadata differs")
    mode, keep, price = ARMS[args.arm]
    checkpoint = json.loads(args.checkpoint_proof.read_text())
    signatures = {f["filename"]: [(args.model / f["filename"]).stat().st_size,
                                  (args.model / f["filename"]).stat().st_mtime_ns]
                  for f in checkpoint["safetensors"]}
    native_paths = sorted(NATIVE_FILES)
    native = importlib.metadata.distribution("vllm")
    native_source_files = {name: str(Path(native.locate_file(name)).resolve()) for name in native_paths}
    runtime = {"ep": 4, "dp": 4, "tp": 1, "pp": 1, "max_model_len": 8192,
               "max_num_seqs": 256, "max_num_batched_tokens": 16384, "graphs": "NONE",
               "prefix_caching": False, "all2all_backend": "deepep_high_throughput",
               "dtype": "torch.bfloat16", "expert_placement": "linear", "seed": 0,
               "logprobs_mode": "raw_logprobs", "use_v2_model_runner": True,
               "worker_extension_cls": "stochastic_tail.worker_extension.CheckpointExtension", "server_dev_mode": True}
    value = {"schema": "stochastic-tail-launch-v4", "observer_revision": 4, "created_unix_ns": time.time_ns(),
             "arm": args.arm, "purpose": args.purpose, "record_calls": args.purpose != "prefill_timing",
             "parameters": {"mode": mode, "keep": keep, "seed": args.seed,
                            "range_limit": 32, "temperature": 1.0, "ranks": 4, "load_penalty": price},
             "private_site": str(args.private_site.resolve()), "public_site": str(args.public_site.resolve()),
             "private_source_sha256": private_hashes(), "public_lineage_sha256": PUBLIC,
             "native_source_files": native_source_files,
             "native_source_sha256": {name: sha(path) for name, path in native_source_files.items()},
             "public_source_proof": {"path": str(args.public_source_proof.resolve()), "sha256": sha(args.public_source_proof)},
             "gpu_gate": {"path": str(args.gpu_gate.resolve()), "sha256": sha(args.gpu_gate)},
             "checkpoint_proof": {"path": str(args.checkpoint_proof.resolve()), "sha256": sha(args.checkpoint_proof)},
             "checkpoint_stat_signatures": signatures,
             "model_id": MODEL_ID, "revision": REVISION, "model_path": str(args.model.resolve()),
             "runtime": runtime, "evidence_directory": str(args.evidence.resolve()),
             "call_ledger_is_instrumentation": args.purpose != "prefill_timing",
             "rng": "lowbias32 unsigned wrap; seed/physical-DP-rank/native-layer/per-router-call/local-row/expert;23-bit open U; lower-ID ties",
             "request_invariance": False, "observer_control_revision": "v4 native MRV2 contexts and pause/RPC checkpoint/resume; do not pool with failed v3 observer outcomes",
             "sampling_tuning": "None: seeds0/1/2, tau1, allowed-rank32 fixed before any held-out evaluation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    try:
        read_plan(args.output, sha(args.output))
    except BaseException:
        # The failed plan is deliberately retained for diagnosis.
        raise
    environment = {"VLLM_PLUGINS": "stochastic_tail_experiment", "VLLM_SERVER_DEV_MODE": "1",
                   "PYTHONPATH": str(args.private_site.resolve()) + ":" + str(args.public_site.resolve()),
                   "STOCHASTIC_EXPERIMENT_PLAN": str(args.output.resolve()),
                   "STOCHASTIC_EXPERIMENT_PLAN_SHA256": sha(args.output)}
    if mode == "par":
        environment.update(PLACEMENT_KEEP=str(keep), PLACEMENT_LOAD_PENALTY=str(price))
    print(json.dumps({"plan": str(args.output.resolve()), "sha256": sha(args.output),
                      "environment": environment, "remove_parent_environment_prefixes": ["PLACEMENT_", "STOCHASTIC_", "PAPER_PROBE_", "PREFILL_BENCH_"],
                      "required_server_arguments": ["--data-parallel-size", "4", "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
                          "--enable-expert-parallel", "--all2all-backend", "deepep_high_throughput", "--dtype", "bfloat16",
                          "--expert-placement-strategy", "linear", '--compilation-config={"cudagraph_mode":"NONE"}',
                          "--max-model-len", "8192", "--max-num-batched-tokens", "16384", "--max-num-seqs", "256",
                          "--no-enable-prefix-caching", "--seed", "0", "--logprobs-mode", "raw_logprobs",
                          "--worker-extension-cls", "stochastic_tail.worker_extension.CheckpointExtension"],
                      "note": "Use the experimental identity adapter; frozen deterministic COMMIT/SOURCE activation checks do not apply to private compute callables."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["output", "private-site", "public-site", "public-source-proof", "gpu-gate", "checkpoint-proof", "model", "evidence"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--purpose", choices=["quality", "preflight", "proof", "prefill_timing"], default="quality")
    build(parser.parse_args())
