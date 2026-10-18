#!/usr/bin/env python3
"""Report-only paired summaries after the complete fresh causal raw audit.

This does not replace the frozen raw readers or inspect an active experiment.
It checks their complete 6+3 result, recomputes every chosen answer from saved
candidate scores, and applies the prospective conditional-item report protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import traceback

ARMS = ("stock", "keep6p8", "keep4p8")
KEYS = ("prefix_norm", "prefix_raw", "teacher_norm", "teacher_raw")
N = 256
REPLICATES = 10000
PROTOCOL_SHA = "9b1094cd0d3521efc595cb51c45976228e3d08c53f7de8861a1bdfba7a1a73ab"


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_indices():
    return sorted(random.Random(0).sample(range(10042), N))


def validate_arm(arm, data):
    need(data.get("complete") is True and data.get("held_out") is True
         and data.get("arm") == arm and data.get("n") == N
         and data.get("serial_primary_prefix_requests") == 29611,
         "Complete frozen256-item causal arm required")
    items, candidates = data["items"], data["candidates"]
    need(len(items) == N and [v["item_index"] for v in items] == selected_indices()
         and len(candidates) == N * 4 and len({v["id"] for v in candidates}) == N * 4,
         "Exact selected item/candidate identity inventory required")
    grouped = {v["item_index"]: [] for v in items}
    for row in candidates:
        need(row["item_index"] in grouped and type(row["choice"]) is int
             and 0 <= row["choice"] < 4 and type(row["gold"]) is int
             and 0 <= row["gold"] < 4 and type(row["normalization_chars"]) is int
             and row["normalization_chars"] > 0
             and all(type(row[k]) in (int, float) and math.isfinite(row[k])
                     for k in ("teacher", "prefix")), "Invalid candidate score/identity")
        grouped[row["item_index"]].append(row)
    predictions, correctness, scores = ({key: [] for key in KEYS} for _ in range(3))
    for item in items:
        rows = grouped[item["item_index"]]
        need([v["choice"] for v in rows] == [0, 1, 2, 3]
             and {v["gold"] for v in rows} == {item["gold"]}, "Choice ordering/gold differs")
        for key in KEYS:
            policy, scoring = key.split("_")
            values = [v[policy] / v["normalization_chars"] if scoring == "norm"
                      else v[policy] for v in rows]
            # Python max retains the first candidate on an exact tie, as frozen.
            prediction = max(range(4), key=values.__getitem__)
            correct = prediction == item["gold"]
            need(type(item[key + "_prediction"]) is int
                 and item[key + "_prediction"] == prediction
                 and type(item[key + "_correct"]) is bool
                 and item[key + "_correct"] == correct, "Stored chosen answer differs from scores")
            predictions[key].append(prediction)
            correctness[key].append(correct)
            scores[key].append(math.fsum(values) / 4)
    need(set(data["correct"]) == set(KEYS) and set(data["accuracy_percent"]) == set(KEYS),
         "All four scoring endpoints required")
    for key in KEYS:
        count = sum(correctness[key])
        need(type(data["correct"][key]) is int and data["correct"][key] == count
             and math.isclose(data["accuracy_percent"][key], 100 * count / N,
                              rel_tol=0, abs_tol=1e-12), "Stored accuracy differs from paired items")
    return {"predictions": predictions, "correctness": correctness, "scores": scores}


def paired_accuracy(treatment, baseline, treatment_predictions, baseline_predictions):
    gain = sum(x and not y for x, y in zip(treatment, baseline))
    loss = sum(y and not x for x, y in zip(treatment, baseline))
    indices = selected_indices()
    return {"n": N, "treatment_correct": sum(treatment), "baseline_correct": sum(baseline),
            "paired_gain_items": gain, "paired_loss_items": loss,
            "both_correct_items": sum(x and y for x, y in zip(treatment, baseline)),
            "both_wrong_items": sum(not x and not y for x, y in zip(treatment, baseline)),
            "changed_chosen_answer_items": sum(x != y for x, y in
                                               zip(treatment_predictions, baseline_predictions)),
            "paired_gain_item_indices": [i for i, x, y in zip(indices, treatment, baseline) if x and not y],
            "paired_loss_item_indices": [i for i, x, y in zip(indices, treatment, baseline) if y and not x],
            "changed_chosen_answer_item_indices": [i for i, x, y in
                zip(indices, treatment_predictions, baseline_predictions) if x != y],
            "difference_pp": 100 * (gain - loss) / N}


def intervals(vectors, replicates=REPLICATES):
    """Common fixed-seed paired resamples; nearest-rank empirical bounds."""
    need(replicates > 0 and all(len(v) == N and all(math.isfinite(x) for x in v)
                               for v in vectors.values()), "Invalid paired sensitivity vectors")
    sparse = {key: [(i, v) for i, v in enumerate(values) if v != 0]
              for key, values in vectors.items()}
    draws = {key: [] for key in vectors}
    rng = random.Random(0)
    for _ in range(replicates):
        weights = [0] * N
        for _ in range(N):
            weights[rng.randrange(N)] += 1
        for key, values in sparse.items():
            draws[key].append(math.fsum(v * weights[i] for i, v in values) / N)
    result = {}
    for key, values in draws.items():
        values.sort()
        result[key] = {"lower": values[math.ceil(.025 * replicates) - 1],
                       "upper": values[math.ceil(.975 * replicates) - 1]}
    return result


def summarize(raw_audit, expected_plan_sha256):
    need(raw_audit.get("complete") is True and raw_audit.get("all_checks_pass") is True
         and raw_audit.get("training_launches") == 6 and raw_audit.get("heldout_launches") == 3
         and set(raw_audit["heldout"]) == set(ARMS), "All fresh6+3 raw audits required before report")
    need(len(expected_plan_sha256) == 64 and all(c in "0123456789abcdef" for c in expected_plan_sha256)
         and raw_audit.get("plan_sha256") == expected_plan_sha256,
         "Report must bind the independently selected fresh operational plan")
    rows = {arm: validate_arm(arm, raw_audit["heldout"][arm]["analysis"]) for arm in ARMS}
    reports, vectors = {}, {}
    def compare(name, left_arm, left_key, right_arm, right_key, primary=False):
        left, right = rows[left_arm], rows[right_arm]
        accuracy = paired_accuracy(left["correctness"][left_key], right["correctness"][right_key],
                                   left["predictions"][left_key], right["predictions"][right_key])
        score_vector = [x - y for x, y in zip(left["scores"][left_key], right["scores"][right_key])]
        reports[name] = {"left": {"arm": left_arm, "endpoint": left_key},
                         "right": {"arm": right_arm, "endpoint": right_key},
                         "primary_accuracy_contrast": primary, "accuracy": accuracy,
                         "candidate_score_difference": {"n": N, "mean": math.fsum(score_vector) / N,
                             "unit": "nats_per_normalization_character" if left_key.endswith("norm") else "nats",
                             "weighting": "Mean across four candidates within each item, then across paired items"}}
        vectors[name + "/accuracy_pp"] = [100 * (int(x) - int(y)) for x, y in
                                           zip(left["correctness"][left_key], right["correctness"][right_key])]
        vectors[name + "/candidate_score"] = score_vector
    for arm in ARMS[1:]:
        for key in KEYS:
            compare(arm + "-minus-stock/" + key, arm, key, "stock", key, key == "prefix_norm")
    for arm in ARMS:
        for scoring in ("norm", "raw"):
            compare(arm + "/teacher-minus-prefix/" + scoring, arm, "teacher_" + scoring,
                    arm, "prefix_" + scoring)
    sensitivity = intervals(vectors)
    for key, row in reports.items():
        row["accuracy"]["conditional_item_sensitivity_95"] = sensitivity[key + "/accuracy_pp"]
        row["candidate_score_difference"]["conditional_item_sensitivity_95"] = sensitivity[key + "/candidate_score"]
    return {"complete": True, "schema": "fresh-causal-paired-report-v1", "n": N,
            "selected_item_indices": selected_indices(),
            "per_arm": {arm: {key: {"correct": sum(rows[arm]["correctness"][key]), "n": N,
                         "accuracy_percent": 100 * sum(rows[arm]["correctness"][key]) / N}
                         for key in KEYS} for arm in ARMS},
            "contrasts": reports, "item_sensitivity": {"seed": 0, "replicates": REPLICATES,
                "method": "Common paired item bootstrap; nearest-rank2.5%/97.5% bounds",
                "interpretation": "Conditional item sensitivity only, unadjusted descriptive intervals; not launch/host variance or equivalence."},
            "qualifications": ["Fixed256-item subset, one fresh server per arm; not full10042 or unseen evaluation.",
                "Serial full-reprefill policy differs from incremental KV and mixed prefill/decode serving.",
                "Teacher-minus-prefix combines candidate conditioning, cohort shape and numerical execution; stock control retained.",
                "Four-candidate score shifts do not by themselves imply improved chosen-answer accuracy."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--raw-audit-sha256", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        need(sha(args.raw_audit) == args.raw_audit_sha256 and sha(args.protocol) == PROTOCOL_SHA,
             "Frozen raw-audit/report protocol binding differs")
        result = summarize(json.loads(args.raw_audit.read_bytes()), args.expected_plan_sha256)
        result.update(raw_audit_sha256=args.raw_audit_sha256, report_protocol_sha256=PROTOCOL_SHA,
                      report_source_sha256=sha(__file__), fresh_plan_sha256=args.expected_plan_sha256)
    except BaseException:
        result = {"complete": False, "effects": None, "error": traceback.format_exc()}
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    raise SystemExit(0 if result["complete"] else 2)


if __name__ == "__main__":
    main()
