#!/usr/bin/env python3
"""Training-only causal LP, suffix-invariance and scheduling diagnostics.

No model/cluster lifetime operations. Prefix requests never include the target
or later tokens and never constrain generation. Exact arbitrary-target LPs use
vLLM0.31's logprob_token_ids. They are serial single-prompt requests, so their
conditional score has no other candidate text in its HTTP cohort. Re-prefilling
each prefix defines a serving policy, not the likelihood of an incremental cache
trajectory; that distinction remains in every result.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import statistics
import time
from common_v2 import (MODEL_ID, REVISION, SOURCE_COMMIT, SOURCE, base_body, choices,
                    encoded, file_sha, next_lp, post, prompt_lp, read, require, sha, write_new)


def prepare(args):
    require(file_sha(args.prior_preflight) == args.prior_preflight_sha256, "Frozen training inputs changed")
    rows = [r for r in read(args.prior_preflight) if r["task"] == "hellaswag"]
    require(len(rows) == 16 and len({r["id"] for r in rows}) == 16, "Exactly four frozen training items required")
    require(all(r["split"] == "train-preflight" and r["item_index"] in range(4) for r in rows),
            "Held-out data must not enter training preflight")
    rows = sorted(rows, key=lambda r: (r["item_index"], r["choice"]))
    checkpoint, source = read(args.checkpoint_proof), read(args.source_proof)
    require(checkpoint["canonical_identity_verified"] and checkpoint["model_id"] == MODEL_ID
            and checkpoint["revision"] == REVISION and checkpoint["weights_verified"] == checkpoint["safetensors"]
            and len(checkpoint["safetensors"]) == 16, "Pinned checkpoint full-shard proof required")
    require(source.get("source_commit", source.get("base_commit")) == SOURCE_COMMIT, "Wrong production commit")
    for relative, expected in SOURCE.items():
        require(file_sha(args.production_root / relative) == expected, "Production source differs: " + relative)
    # Neighbour schedule includes 64 requests without looking beyond the four
    # preregistered training items: four labelled replicas of the same 16 rows.
    teacher_rows = []
    for replica in range(4):
        for row in rows:
            teacher_rows.append({**row, "probe_id": row["id"] + ":replica" + str(replica), "replica": replica})
    spec = {"schema_version": 1, "training_only": True, "rows": rows, "teacher_rows": teacher_rows,
            "orders": ["original", "reverse", "permutation1729"], "cohorts": [1, 16, 64],
            "prefix_scoring": "Serial single-prompt re-prefill, target absent; logprob_token_ids only",
            "prefix_score_scope": "Conditional on serving policy and configured chunking; not one incremental KV trajectory",
            "suffix_probe_positions": "unique continuation_start, +1, +7, penultimate, clipped to permit nonempty suffix",
            "suffix_mutation": "Reverse suffix after scored position; if unchanged replace first suffix token with13/220",
            "prefix_score_all_continuation_tokens": True,
            "generation": "Teacher max1naturalEOS; prefix max1ignoreEOStrue ensures one returned LP even atEOS; raw logits unchanged",
            "no_model_input_contains_gold": True, "analysis_tolerance_abs_logprob": 1e-5,
            "statistical_scope": "Diagnostic finite training inputs; no quality/equivalence or throughput claim",
            "model_id": MODEL_ID, "revision": REVISION, "source_commit": SOURCE_COMMIT,
            "source_files": SOURCE, "checkpoint_proof_sha256": file_sha(args.checkpoint_proof),
            "source_proof_sha256": file_sha(args.source_proof), "prior_preflight_sha256": file_sha(args.prior_preflight),
            "prior_preflight_canonical_sha256": sha(encoded(read(args.prior_preflight))),
            "independent_training_items": 4, "teacher_schedule_replicas_per_candidate": 4,
            "harness_sha256": file_sha(__file__), "common_sha256": file_sha(Path(__file__).with_name("common_v2.py"))}
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output / "protocol.json", spec)


def teacher_body(rows, model, request_id):
    body = base_body(model, rows[0]["token_ids"], request_id)
    body["prompt"] = [r["token_ids"] for r in rows]
    body["prompt_logprobs"] = 0
    return body


def teacher_scores(rows, response):
    values = choices(response, [r["token_ids"] for r in rows])
    return [{"id": row.get("probe_id", row["id"]), "source_id": row["id"],
             "token_logprobs": [prompt_lp(choice, row["token_ids"], j)
                                for j in range(row["continuation_start"], len(row["token_ids"]))]}
            for row, choice in zip(rows, values)]


def positions(row):
    start, last = row["continuation_start"], len(row["token_ids"]) - 2
    return sorted({min(last, start + offset) for offset in (0, 1, 7)} | {last})


def mutate_suffix(ids, position):
    require(0 < position < len(ids) - 1, "Need preceding context and nonempty suffix")
    mutated = ids[:position + 1] + list(reversed(ids[position + 1:]))
    if mutated == ids:
        mutated[position + 1] = 13 if ids[position + 1] != 13 else 220
    require(mutated[:position + 1] == ids[:position + 1] and len(mutated) == len(ids) and mutated != ids,
            "Invalid suffix-only mutation")
    return mutated


def run(args):
    spec = read(args.prepared / "protocol.json")
    require(spec["harness_sha256"] == file_sha(__file__)
            and spec["common_sha256"] == file_sha(Path(__file__).with_name("common_v2.py")), "Frozen helper changed")
    contract = read(args.server_contract)
    require(contract.get("complete") is True and contract["source_commit"] == SOURCE_COMMIT
            and contract["checkpoint_revision"] == REVISION, "Server contract not pinned/ready")
    require(contract["arm"] == args.arm and contract["chunking_label"] == args.chunking_label,
            "Requested arm/chunking differs from contract")
    require(contract["checkpoint_proof_sha256"] == spec["checkpoint_proof_sha256"]
            and contract["source_proof_sha256"] == spec["source_proof_sha256"], "Prepared/runtime identity proof differs")
    require(contract["prefix_caching"] is False, "Prefix-only probes require prefix cache disabled")
    require(args.stop_unix > time.time() and args.max_requests > 0, "Missing/expired global bound")
    args.output.mkdir(parents=True, exist_ok=False)
    records, ordinal = [], 0
    def call(body, label):
        nonlocal ordinal
        require(ordinal < args.max_requests and time.time() < args.stop_unix, "Global request/time bound reached")
        ordinal += 1
        body["request_id"] = args.launch_id + f"-{ordinal:05d}"
        result, transport = post(args.url, body, args.output / "raw" / f"{ordinal:05d}-{label}",
                                 timeout=min(args.timeout, max(1, args.stop_unix - time.time())))
        return result, transport
    write_new(args.output / "start.json", {"complete": False, "protocol_sha256": file_sha(args.prepared / "protocol.json"),
              "server_contract": contract, "server_contract_sha256": file_sha(args.server_contract),
              "server_contract_canonical_sha256": sha(encoded(contract)),
              "launch_id": args.launch_id, "arm": args.arm, "chunking_label": args.chunking_label,
              "served_model": args.served_model,
              "started_unix_ns": time.time_ns(), "helper_sha256": file_sha(__file__),
              "common_sha256": file_sha(Path(__file__).with_name("common_v2.py")),
              "max_requests": args.max_requests, "stop_unix": args.stop_unix})
    try:
        # Same inputs/order repeated to quantify observed stock/treatment drift.
        for repeat in range(2):
            for row in spec["rows"]:
                response, _ = call(teacher_body([row], args.served_model, ""), "singleton")
                score = teacher_scores([row], response)[0]
                records.append({"kind": "singleton", "repeat": repeat, **score})
        for order in spec["orders"]:
            ordered = list(spec["teacher_rows"])
            if order == "reverse":
                ordered.reverse()
            elif order.startswith("permutation"):
                random.Random(1729).shuffle(ordered)
            for cohort in spec["cohorts"]:
                for offset in range(0, len(ordered), cohort):
                    selected = ordered[offset:offset + cohort]
                    response, _ = call(teacher_body(selected, args.served_model, ""), "schedule")
                    records.extend({"kind": "schedule", "order": order, "cohort": cohort, "offset": offset, **score}
                                   for score in teacher_scores(selected, response))
        for row in spec["rows"]:
            for j in range(row["continuation_start"], len(row["token_ids"])):
                prefix, target = row["token_ids"][:j], row["token_ids"][j]
                body = base_body(args.served_model, prefix, "")
                body.update(logprobs=0, logprob_token_ids=[target], return_tokens_as_token_ids=True, ignore_eos=True)
                require(body["prompt"] == row["token_ids"][:j] and len(body["prompt"]) == j,
                        "Target/future leaked into prefix request")
                response, _ = call(body, "prefix")
                choice = choices(response, [prefix])[0]
                records.append({"kind": "prefix", "id": row["id"], "position": j,
                                "prefix_tokens": j, "target_id": target, "logprob": next_lp(choice, target)})
            for j in positions(row):
                changed = mutate_suffix(row["token_ids"], j)
                mutation_row = {**row, "token_ids": changed}
                response, _ = call(teacher_body([mutation_row], args.served_model, ""), "suffix")
                choice = choices(response, [changed])[0]
                records.append({"kind": "suffix", "id": row["id"], "position": j,
                                "mutated_token_ids_sha256": sha(encoded(changed)),
                                "identical_prefix_through_position": True, "logprob": prompt_lp(choice, changed, j)})
        write_new(args.output / "records.json", records)
        write_new(args.output / "result.json", {"complete": True, "requests": ordinal,
                  "records_sha256": file_sha(args.output / "records.json"), "arm": args.arm,
                  "chunking_label": args.chunking_label, "training_only": True,
                  "start_sha256": file_sha(args.output / "start.json"),
                  "raw_files_sha256": {str(path.relative_to(args.output)): file_sha(path)
                                       for path in sorted((args.output / "raw").glob("*.json"))},
                  "prefix_scope": spec["prefix_score_scope"], "finished_unix_ns": time.time_ns()})
    except BaseException as error:
        write_new(args.output / "partial-records.json", records)
        write_new(args.output / "failure.json", {"complete": False, "requests_attempted": ordinal,
                  "error_type": type(error).__name__, "error": str(error)})
        raise


def validate_records(spec, records):
    """Exact preregistered key sets and lengths, before any zip or aggregation."""
    rows = {r["id"]: r for r in spec["rows"]}
    expected = set()
    for row_id, row in rows.items():
        expected.update(("singleton", row_id, repeat) for repeat in range(2))
        expected.update(("prefix", row_id, j) for j in range(row["continuation_start"], len(row["token_ids"])))
        expected.update(("suffix", row_id, j) for j in positions(row))
    expected.update(("schedule", order, cohort, r["probe_id"])
                    for order in spec["orders"] for cohort in spec["cohorts"] for r in spec["teacher_rows"])
    observed = set()
    for record in records:
        kind = record.get("kind")
        if kind == "singleton":
            key, row_id = (kind, record["id"], record["repeat"]), record["id"]
        elif kind in ("prefix", "suffix"):
            key, row_id = (kind, record["id"], record["position"]), record["id"]
        elif kind == "schedule":
            key, row_id = (kind, record["order"], record["cohort"], record["id"]), record["source_id"]
        else:
            raise RuntimeError("Unknown record kind")
        require(key in expected and key not in observed and row_id in rows, "Unknown/duplicate record key")
        observed.add(key)
        if kind in ("singleton", "schedule"):
            values = record["token_logprobs"]
            require(len(values) == rows[row_id]["continuation_tokens"], "Wrong per-token LP vector length")
        else:
            values = [record["logprob"]]
        require(all(type(v) in (int, float) and math.isfinite(v) and -9998 < v <= 1e-6 for v in values),
                "Nonfinite/invalid logprob in records")
    require(observed == expected, "Missing preregistered records")
    return True


def rescore_raw(spec, start, folder, receipt):
    """Rebuild the entire registered request sequence and scores from raw HTTP."""
    raw_paths = sorted((folder / "raw").glob("*.json"))
    require(set(receipt["raw_files_sha256"]) == {str(p.relative_to(folder)) for p in raw_paths}, "Raw request inventory differs")
    for path in raw_paths:
        require(file_sha(path) == receipt["raw_files_sha256"][str(path.relative_to(folder))], "Raw HTTP bytes changed")
    ordinal, records = 0, []
    def original(body, label):
        nonlocal ordinal
        ordinal += 1
        body["request_id"] = start["launch_id"] + f"-{ordinal:05d}"
        stem = folder / "raw" / f"{ordinal:05d}-{label}"
        request = read(stem.with_suffix(".request.json"))
        require(request["body"] == body and request["sha256"] == sha(encoded(body)) and request["headers"] == {},
                "Actual wire request differs from registered prefix/teacher protocol")
        transport = read(stem.with_suffix(".transport.json"))
        require(transport["response_complete"] is True and transport["http_status"] == 200
                and transport["response_sha256"] == file_sha(stem.with_suffix(".response.json")), "Raw response incomplete/changed")
        return read(stem.with_suffix(".response.json"))
    model = start["served_model"]
    for repeat in range(2):
        for row in spec["rows"]:
            score = teacher_scores([row], original(teacher_body([row], model, ""), "singleton"))[0]
            records.append({"kind": "singleton", "repeat": repeat, **score})
    for order in spec["orders"]:
        ordered = list(spec["teacher_rows"])
        if order == "reverse":
            ordered.reverse()
        elif order.startswith("permutation"):
            random.Random(1729).shuffle(ordered)
        for cohort in spec["cohorts"]:
            for offset in range(0, len(ordered), cohort):
                rows = ordered[offset:offset + cohort]
                scores = teacher_scores(rows, original(teacher_body(rows, model, ""), "schedule"))
                records.extend({"kind": "schedule", "order": order, "cohort": cohort, "offset": offset, **score} for score in scores)
    for row in spec["rows"]:
        for j in range(row["continuation_start"], len(row["token_ids"])):
            prefix, target = row["token_ids"][:j], row["token_ids"][j]
            body = base_body(model, prefix, "")
            body.update(logprobs=0, logprob_token_ids=[target], return_tokens_as_token_ids=True, ignore_eos=True)
            choice = choices(original(body, "prefix"), [prefix])[0]
            records.append({"kind": "prefix", "id": row["id"], "position": j, "prefix_tokens": j,
                            "target_id": target, "logprob": next_lp(choice, target)})
        for j in positions(row):
            changed = mutate_suffix(row["token_ids"], j)
            choice = choices(original(teacher_body([{**row, "token_ids": changed}], model, ""), "suffix"), [changed])[0]
            records.append({"kind": "suffix", "id": row["id"], "position": j,
                            "mutated_token_ids_sha256": sha(encoded(changed)), "identical_prefix_through_position": True,
                            "logprob": prompt_lp(choice, changed, j)})
    require(ordinal == receipt["requests"] and len(raw_paths) == ordinal * 3, "Actual request count/file set differs")
    return records


def analyze(args):
    spec, records = read(args.prepared / "protocol.json"), read(args.result / "records.json")
    require(read(args.result / "result.json")["complete"], "Incomplete run")
    require(file_sha(args.result / "records.json") == read(args.result / "result.json")["records_sha256"], "Records changed")
    receipt, start = read(args.result / "result.json"), read(args.result / "start.json")
    require(receipt["start_sha256"] == file_sha(args.result / "start.json")
            and start["protocol_sha256"] == file_sha(args.prepared / "protocol.json")
            and start["helper_sha256"] == spec["harness_sha256"] == file_sha(__file__)
            and start["common_sha256"] == spec["common_sha256"] == file_sha(Path(__file__).with_name("common_v2.py")),
            "Run/preparation/source bindings changed")
    contract = start["server_contract"]
    require(start["server_contract_canonical_sha256"] == sha(encoded(contract)), "Embedded server contract changed")
    require(receipt["arm"] == start["arm"] == contract["arm"]
            and receipt["chunking_label"] == start["chunking_label"] == contract["chunking_label"]
            and contract["source_proof_sha256"] == spec["source_proof_sha256"]
            and contract["checkpoint_proof_sha256"] == spec["checkpoint_proof_sha256"], "Run arm/chunking/proof binding differs")
    validate_records(spec, records)
    require(records == rescore_raw(spec, start, args.result, receipt), "Stored scores differ from raw response re-scoring")
    single = {(r["id"], r["repeat"]): r["token_logprobs"] for r in records if r["kind"] == "singleton"}
    rows = {r["id"]: r for r in spec["rows"]}
    deltas = defaultdict(list)
    details = []
    for row_id, row in rows.items():
        base = single[(row_id, 0)]
        deltas["repeat"].extend(b - a for a, b in zip(base, single[(row_id, 1)]))
        prefix = sorted([r for r in records if r["kind"] == "prefix" and r["id"] == row_id], key=lambda r: r["position"])
        require([r["position"] for r in prefix] == list(range(row["continuation_start"], len(row["token_ids"]))),
                "Prefix score missing/duplicated target positions")
        causal = [r["logprob"] for r in prefix]
        deltas["prefix_minus_teacher"].extend(b - a for a, b in zip(base, causal))
        details.append({"id": row_id, "teacher_sum": math.fsum(base), "prefix_sum": math.fsum(causal),
                        "teacher_normalized": math.fsum(base) / row["normalization_chars"],
                        "prefix_normalized": math.fsum(causal) / row["normalization_chars"]})
    for r in records:
        if r["kind"] == "suffix":
            base = single[(r["id"], 0)][r["position"] - rows[r["id"]]["continuation_start"]]
            deltas["suffix_minus_original"].append(r["logprob"] - base)
        elif r["kind"] == "schedule":
            base = single[(r["source_id"], 0)]
            deltas[f"schedule_{r['order']}_cohort{r['cohort']}_minus_singleton"].extend(
                b - a for a, b in zip(base, r["token_logprobs"]))
    summary = {name: {"n": len(values), "max_abs": max(map(abs, values)),
                      "mean_abs": statistics.fmean(map(abs, values)),
                      "above_1e_minus5": sum(abs(v) > 1e-5 for v in values)} for name, values in deltas.items()}
    write_new(args.output, {"complete": True, "training_only": True, "diagnostics": summary,
              "arm": receipt["arm"], "chunking_label": receipt["chunking_label"],
              "checkpoint_revision": REVISION, "source_commit": SOURCE_COMMIT,
              "checkpoint_proof_sha256": spec["checkpoint_proof_sha256"], "source_proof_sha256": spec["source_proof_sha256"],
              "protocol_sha256": file_sha(args.prepared / "protocol.json"),
              "candidate_scores": details, "no_accuracy_or_significance_claim": True,
              "interpretation": "Compare treatment with matched stock repeat and stock full-versus-prefix controls; nonzero differences alone do not isolate their numerical cause.",
              "prefix_scope": spec["prefix_score_scope"], "result_sha256": file_sha(args.result / "result.json")})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--prior-preflight", type=Path, required=True)
    prep.add_argument("--prior-preflight-sha256", required=True)
    prep.add_argument("--checkpoint-proof", type=Path, required=True)
    prep.add_argument("--source-proof", type=Path, required=True)
    prep.add_argument("--production-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    launch = sub.add_parser("run")
    for name in ("prepared", "server-contract", "output"):
        launch.add_argument("--" + name, type=Path, required=True)
    for name in ("url", "launch-id", "arm", "chunking-label"):
        launch.add_argument("--" + name, required=True)
    launch.add_argument("--served-model", default="task-model")
    launch.add_argument("--timeout", type=int, default=900)
    launch.add_argument("--max-requests", type=int, required=True)
    launch.add_argument("--stop-unix", type=float, required=True)
    analysis = sub.add_parser("analyze")
    for name in ("prepared", "result", "output"):
        analysis.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
