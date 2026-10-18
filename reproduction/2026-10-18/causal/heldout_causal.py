#!/usr/bin/env python3
"""Frozen256-item HellaSwag validation comparison after the training gate.

Primary prefix scores are SERIAL one-request re-prefills. No concurrent request
contains a later prefix, target, or continuation of any other candidate. Teacher
scores use fixed256-prompt cohorts and are labelled pseudo-scores for priced
tile routing. Both are operational serving policies; this does not claim one
incremental-KV model trajectory or scheduling-invariant likelihoods.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import time
from common_v2 import (REVISION, SOURCE_COMMIT, base_body, choices, encoded, file_sha,
                       next_lp, post, read, require, sha, write_new)
from causal_probe import teacher_body, teacher_scores


def selected_indices(count=10042):
    return sorted(random.Random(0).sample(range(count), 256))


def prepare(args):
    require(file_sha(args.full_requests) == args.full_requests_sha256, "Canonical full input bytes changed")
    analyses = [read(p) for p in args.training_analyses]
    expected = {(arm, f"budget{budget}") for arm in ("stock", "keep6p8", "keep4p8") for budget in (1024, 16384)}
    require(len(analyses) == len(expected) and {(a["arm"], a["chunking_label"]) for a in analyses} == expected
            and all(a.get("complete") and a.get("training_only") for a in analyses), "All six native training diagnostic cells must close first")
    require(all(a["source_commit"] == SOURCE_COMMIT and a["checkpoint_revision"] == REVISION
                and a["source_proof_sha256"] == args.source_proof_sha256
                and a["checkpoint_proof_sha256"] == args.checkpoint_proof_sha256 for a in analyses),
            "Training and held-out identity proofs differ")
    all_rows = [r for r in read(args.full_requests) if r["task"] == "hellaswag"]
    require(len(all_rows) == 40168 and all(r["split"] == "validation" for r in all_rows)
            and len({r["id"] for r in all_rows}) == 40168, "Require full canonical HellaSwag validation candidates")
    indices = selected_indices()
    rows = sorted([r for r in all_rows if r["item_index"] in set(indices)], key=lambda r: (r["item_index"], r["choice"]))
    require(len(rows) == 1024 and all([r["choice"] for r in rows if r["item_index"] == i] == list(range(4)) for i in indices),
            "Frozen subset missing/duplicated candidate choices")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "selected-requests.json", rows)
    targets = sum(r["continuation_tokens"] for r in rows)
    spec = {"schema": "hellaswag-heldout-serial-causal-v1", "complete": True, "items": 256, "candidates": 1024,
            "selection": "random.Random(0).sample(range(10042),256), then sort original row indices; no gold-based filtering",
            "selected_item_indices": indices, "source_commit": SOURCE_COMMIT, "checkpoint_revision": REVISION,
            "full_requests_sha256": file_sha(args.full_requests), "selected_requests_sha256": file_sha(args.output / "selected-requests.json"),
            "training_analyses_sha256": {str(p): file_sha(p) for p in args.training_analyses},
            "checkpoint_proof_sha256": args.checkpoint_proof_sha256, "source_proof_sha256": args.source_proof_sha256,
            "arms": ["stock", "keep6p8", "keep4p8"], "chunking_label": "budget16384", "teacher_cohort": 256,
            "prefix_concurrency": 1, "teacher_requests": 4, "prefix_requests": targets, "total_requests": 4 + targets,
            "prefix_policy": "Serial target-absent re-prefill with raw arbitrary-target next-position LP, no other live evaluation request",
            "teacher_policy": "Teacher-forced candidate-conditioned pseudo-score under tile penalty; fixed256prompt cohorts",
            "classification": "argmax summed continuation LP and LP/preprocessed-ending Python-character-count; first-choice tie break",
            "prefix_generation": "max1ignoreEOStrue,greedyseed0; does not constrain logits or expose target as input",
            "statistical_scope": "Observed fixed256-item comparison conditional on one launch perarm; no population/equivalence claim",
            "helper_sha256": file_sha(__file__), "common_sha256": file_sha(Path(__file__).with_name("common_v2.py")),
            "teacher_helper_sha256": file_sha(Path(__file__).with_name("causal_probe.py"))}
    write_new(args.output / "protocol.json", spec)


def validate(args):
    spec = read(args.prepared / "protocol.json")
    require(spec["helper_sha256"] == file_sha(__file__) and spec["common_sha256"] == file_sha(Path(__file__).with_name("common_v2.py"))
            and spec["teacher_helper_sha256"] == file_sha(Path(__file__).with_name("causal_probe.py")), "Frozen helpers changed")
    require(file_sha(args.prepared / "selected-requests.json") == spec["selected_requests_sha256"], "Subset bytes changed")
    require(spec["selected_item_indices"] == selected_indices() and spec["prefix_concurrency"] == 1, "Selection or serial reference changed")
    return spec, read(args.prepared / "selected-requests.json")


def run(args):
    spec, rows = validate(args)
    contract = read(args.server_contract)
    require(contract.get("complete") and contract["arm"] == args.arm and args.arm in spec["arms"]
            and contract["chunking_label"] == spec["chunking_label"] and contract["prefix_caching"] is False
            and contract["source_commit"] == SOURCE_COMMIT and contract["checkpoint_revision"] == REVISION
            and contract["checkpoint_proof_sha256"] == spec["checkpoint_proof_sha256"]
            and contract["source_proof_sha256"] == spec["source_proof_sha256"], "Ready native contract differs from held-out freeze")
    require(args.exclusive_server_ack == "no-other-live-evaluation-requests" and args.max_requests == spec["total_requests"]
            and args.stop_unix > time.time(), "Serial exclusive-serving acknowledgement and exact finite request bound required")
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "start.json", {"protocol_sha256": file_sha(args.prepared / "protocol.json"),
              "arm": args.arm, "contract": contract, "contract_canonical_sha256": sha(encoded(contract)),
              "launch_id": args.launch_id, "served_model": args.served_model, "stop_unix": args.stop_unix,
              "max_requests": args.max_requests, "prefix_concurrency": 1, "exclusive_server_ack": args.exclusive_server_ack})
    ordinal, teacher, prefix = 0, [], []
    def call(body, label):
        nonlocal ordinal
        require(ordinal < args.max_requests and time.time() < args.stop_unix, "Global request/time bound reached")
        ordinal += 1
        body["request_id"] = args.launch_id + f"-{ordinal:06d}"
        value, _ = post(args.url, body, args.output / "raw" / f"{ordinal:06d}-{label}",
                         timeout=min(args.timeout, max(1, args.stop_unix - time.time())))
        return value
    try:
        for offset in range(0, len(rows), 256):
            selected = rows[offset:offset + 256]
            response = call(teacher_body(selected, args.served_model, ""), "teacher")
            teacher.extend(teacher_scores(selected, response))
        # No executor, async client, future submission or lookahead. Each exact
        # request and response fully completes before the next prefix is built.
        for row in rows:
            lps = []
            for j in range(row["continuation_start"], len(row["token_ids"])):
                ids, target = row["token_ids"][:j], row["token_ids"][j]
                body = base_body(args.served_model, ids, "")
                body.update(ignore_eos=True, logprobs=0, logprob_token_ids=[target], return_tokens_as_token_ids=True)
                response = call(body, "prefix")
                lps.append(next_lp(choices(response, [ids])[0], target))
            prefix.append({"id": row["id"], "token_logprobs": lps})
        require(ordinal == spec["total_requests"], "Request sequence incomplete")
        write_new(args.output / "scores.json", {"teacher": teacher, "prefix": prefix})
        write_new(args.output / "result.json", {"complete": True, "arm": args.arm, "requests": ordinal,
                  "start_sha256": file_sha(args.output / "start.json"), "scores_sha256": file_sha(args.output / "scores.json"),
                  "raw_files_sha256": {str(p.relative_to(args.output)): file_sha(p) for p in sorted((args.output / "raw").glob("*.json"))}})
    except BaseException as error:
        write_new(args.output / "partial-scores.json", {"teacher": teacher, "prefix": prefix})
        write_new(args.output / "failure.json", {"complete": False, "requests_attempted": ordinal,
                  "error_type": type(error).__name__, "error": str(error)})
        raise


def analyze(args):
    spec, rows = validate(args)
    result, start = read(args.result / "result.json"), read(args.result / "start.json")
    require(result.get("complete") and result["start_sha256"] == file_sha(args.result / "start.json")
            and start["protocol_sha256"] == file_sha(args.prepared / "protocol.json")
            and result["requests"] == spec["total_requests"], "Run/protocol incomplete or changed")
    contract = start["contract"]
    require(start["contract_canonical_sha256"] == sha(encoded(contract)) and contract.get("complete") is True
            and result["arm"] == start["arm"] == contract["arm"] and result["arm"] in spec["arms"]
            and contract["chunking_label"] == spec["chunking_label"] and contract["prefix_caching"] is False
            and contract["source_commit"] == SOURCE_COMMIT and contract["checkpoint_revision"] == REVISION
            and contract["checkpoint_proof_sha256"] == spec["checkpoint_proof_sha256"]
            and contract["source_proof_sha256"] == spec["source_proof_sha256"] and start["prefix_concurrency"] == 1
            and start["exclusive_server_ack"] == "no-other-live-evaluation-requests" and start["max_requests"] == spec["total_requests"],
            "Actual server/serial-policy identity differs")
    raw_paths = sorted((args.result / "raw").glob("*.json"))
    require(len(raw_paths) == spec["total_requests"] * 3 and set(result["raw_files_sha256"]) == {str(p.relative_to(args.result)) for p in raw_paths}, "Raw inventory differs")
    for path in raw_paths:
        require(file_sha(path) == result["raw_files_sha256"][str(path.relative_to(args.result))], "Raw HTTP bytes changed")
    ordinal, teacher, prefix = 0, [], []
    def raw(body, label):
        nonlocal ordinal
        ordinal += 1
        body["request_id"] = start["launch_id"] + f"-{ordinal:06d}"
        stem = args.result / "raw" / f"{ordinal:06d}-{label}"
        envelope, transport = read(stem.with_suffix(".request.json")), read(stem.with_suffix(".transport.json"))
        require(envelope["body"] == body and envelope["sha256"] == sha(encoded(body)) and envelope["headers"] == {},
                "Actual wire request violates serial target-absent policy")
        require(transport["response_complete"] is True and transport["http_status"] == 200
                and transport["response_sha256"] == file_sha(stem.with_suffix(".response.json")), "Response incomplete/changed")
        return read(stem.with_suffix(".response.json"))
    for offset in range(0, len(rows), 256):
        selected = rows[offset:offset + 256]
        teacher.extend(teacher_scores(selected, raw(teacher_body(selected, start["served_model"], ""), "teacher")))
    for row in rows:
        lps = []
        for j in range(row["continuation_start"], len(row["token_ids"])):
            ids, target = row["token_ids"][:j], row["token_ids"][j]
            body = base_body(start["served_model"], ids, "")
            body.update(ignore_eos=True, logprobs=0, logprob_token_ids=[target], return_tokens_as_token_ids=True)
            lps.append(next_lp(choices(raw(body, "prefix"), [ids])[0], target))
        prefix.append({"id": row["id"], "token_logprobs": lps})
    scores = {"teacher": teacher, "prefix": prefix}
    require(ordinal == spec["total_requests"] and file_sha(args.result / "scores.json") == result["scores_sha256"]
            and read(args.result / "scores.json") == scores, "Raw re-scoring differs")
    grouped, candidate_details = defaultdict(list), []
    for row, t, p in zip(rows, teacher, prefix):
        require(row["id"] == t["id"] == p["id"] and len(t["token_logprobs"]) == len(p["token_logprobs"]) == row["continuation_tokens"],
                "Candidate identity/token span differs")
        item = {"id": row["id"], "item_index": row["item_index"], "choice": row["choice"], "gold": row["gold"],
                "teacher": math.fsum(t["token_logprobs"]), "prefix": math.fsum(p["token_logprobs"]),
                "normalization_chars": row["normalization_chars"]}
        grouped[row["item_index"]].append(item)
        candidate_details.append(item)
    items = []
    for index in spec["selected_item_indices"]:
        candidates = grouped[index]
        require([r["choice"] for r in candidates] == list(range(4)) and len({r["gold"] for r in candidates}) == 1, "Bad item choices/gold")
        item = {"item_index": index, "gold": candidates[0]["gold"]}
        for policy in ("teacher", "prefix"):
            for norm in (False, True):
                key = policy + ("_norm" if norm else "_raw")
                selected = max(candidates, key=lambda c: c[policy] / c["normalization_chars"] if norm else c[policy])
                item[key + "_prediction"] = selected["choice"]
                item[key + "_correct"] = selected["choice"] == item["gold"]
        items.append(item)
    correct = {key: sum(r[key + "_correct"] for r in items) for key in ("teacher_raw", "teacher_norm", "prefix_raw", "prefix_norm")}
    write_new(args.output, {"complete": True, "held_out": True, "n": 256, "arm": result["arm"], "correct": correct,
              "accuracy_percent": {key: 100 * value / 256 for key, value in correct.items()}, "items": items,
              "candidates": candidate_details, "serial_primary_prefix_requests": spec["prefix_requests"],
              "result_sha256": file_sha(args.result / "result.json"), "protocol_sha256": file_sha(args.prepared / "protocol.json"),
              "scope": spec["statistical_scope"], "policy_caveat": "Difference includes teacher cohort packing and future-token-conditioned routing; training invariance probes isolate mechanisms"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--full-requests", type=Path, required=True)
    p.add_argument("--full-requests-sha256", required=True)
    p.add_argument("--training-analyses", type=Path, nargs=6, required=True)
    p.add_argument("--checkpoint-proof-sha256", required=True)
    p.add_argument("--source-proof-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    for name in ("prepared", "server-contract", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("url", "arm", "launch-id", "exclusive-server-ack"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--served-model", default="task-model")
    p.add_argument("--stop-unix", type=float, required=True)
    p.add_argument("--max-requests", type=int, required=True)
    p.add_argument("--timeout", type=int, default=900)
    p = sub.add_parser("analyze")
    for name in ("prepared", "result", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)
