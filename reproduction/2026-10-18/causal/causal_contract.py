#!/usr/bin/env python3
"""Training warmup and strict all-rank runtime-contract assembly for LP probes."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from common_v2 import (MODEL_ID, REVISION, SOURCE_COMMIT, SOURCE, base_body, choices,
                       file_sha, post, read, require, write_new)


def warmup(args):
    require(file_sha(args.prepared) == args.prepared_sha256, "Training input raw bytes changed")
    rows = [r for r in read(args.prepared) if r["task"] == "gsm8k" and r["split"] == "train"][:4]
    require(len(rows) == 4, "Require four fixed training prompts")
    args.output.mkdir(parents=True, exist_ok=False)
    for i, row in enumerate(rows):
        require(time.time() < args.stop_unix, "Warmup deadline reached")
        body = base_body(args.served_model, row["token_ids"], args.launch_id + f"-identity-{i}")
        response, _ = post(args.url, body, args.output / f"rank{i}", timeout=min(900, max(1, args.stop_unix - time.time())),
                            headers={"X-data-parallel-rank": str(i)})
        choices(response, [row["token_ids"]])
    write_new(args.output / "result.json", {"complete": True, "training_only": True, "requests": 4})


def build(args):
    checkpoint = read(args.checkpoint_proof)
    require(checkpoint["canonical_identity_verified"] and checkpoint["model_id"] == MODEL_ID
            and checkpoint["revision"] == REVISION and len(checkpoint["safetensors"]) == 16
            and checkpoint["weights_verified"] == checkpoint["safetensors"], "Pinned whole-checkpoint proof missing")
    source = read(args.source_proof)
    require(source.get("source_commit", source.get("base_commit")) == SOURCE_COMMIT, "Production source differs")
    paths = sorted(args.probe_dir.glob("rank*_pid*.json"))
    values = [read(path) for path in paths]
    require(len(values) == 4 and {v["ep_rank"] for v in values} == set(range(4))
            and len({v["pid"] for v in values}) == 4, "Require one closed identity from each EP4 worker")
    probe = file_sha(Path(__file__).with_name("causal_worker_probe.py"))
    for path, v in zip(paths, values):
        require(v.get("complete") and v["arm"] == args.arm and v["source_commit"] == SOURCE_COMMIT
                and v["source_sha256"] == SOURCE and v["all_routers_expected"] and v["observed_router_objects"] == 48
                and v["registered_layers"] == 48 and len(v["router_contracts"]) == 48
                and v["deployment_proof_sha256"] == file_sha(args.source_proof) and v["probe_sha256"] == probe,
                "Native routing/source/probe identity differs")
        require(v["ep_world_size"] == v["physical_dp"] == 4 and v["tp"] == v["pp"] == 1
                and v["backend"] == "deepep_high_throughput" and v["expert_placement_strategy"] == "linear"
                and v["enable_expert_parallel"] and v["graphs"] == "NONE" and v["dtype"] == "torch.bfloat16"
                and v["prefix_caching"] is False and v["ubatching"] is False and v["seed"] == 0
                and v["max_model_len"] == 8192 and v["max_num_seqs"] == 256
                and v["max_num_batched_tokens"] == v["expected_budget"] == args.budget
                and v["model_path"] == checkpoint["local_model_path"] and v["logprobs_mode"] == "raw_logprobs"
                and v["vllm_version"].split("+")[0] == "0.31.0", "Native scheduler/model/logprob protocol differs")
        require(path.name == f"rank{v['ep_rank']}_pid{v['pid']}.json", "Rank/PID filename differs")
    write_new(args.output, {"complete": True, "arm": args.arm, "chunking_label": f"budget{args.budget}",
              "source_commit": SOURCE_COMMIT, "checkpoint_revision": REVISION,
              "checkpoint_proof_sha256": file_sha(args.checkpoint_proof), "source_proof_sha256": file_sha(args.source_proof),
              "prefix_caching": False, "max_num_batched_tokens": args.budget,
              "probe_files_sha256": {str(p): file_sha(p) for p in paths},
              "probe_sha256": probe, "model_path": checkpoint["local_model_path"],
              "builder_sha256": file_sha(__file__), "logprobs_mode": "raw_logprobs"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("warmup")
    for name in ("prepared", "output"):
        w.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared-sha256", "url", "launch-id"):
        w.add_argument("--" + name, required=True)
    w.add_argument("--served-model", default="task-model")
    w.add_argument("--stop-unix", type=float, required=True)
    b = sub.add_parser("build")
    for name in ("checkpoint-proof", "source-proof", "probe-dir", "output"):
        b.add_argument("--" + name, type=Path, required=True)
    b.add_argument("--arm", choices=("stock", "keep8", "plain6", "keep6p8", "keep4p8"), required=True)
    b.add_argument("--budget", type=int, choices=(1024, 16384), required=True)
    args = parser.parse_args()
    {"warmup": warmup, "build": build}[args.command](args)
