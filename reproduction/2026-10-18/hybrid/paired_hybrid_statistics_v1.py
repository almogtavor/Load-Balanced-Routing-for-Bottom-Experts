#!/usr/bin/env python3
"""Independent fixed nine-launch statistics for real prefill/decode handoff.

No incomplete matrix, dropped round, filtered request, or replacement launch is
accepted. Source/HTTP/KV/raw integrity must pass a separate independent adapter.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ("stock", "keep6p8", "keep4p8")
ORDERS = (ARMS, ("keep6p8", "keep4p8", "stock"), ("keep4p8", "stock", "keep6p8"))
T975_DF2 = 4.302652729749464
APPLICATION_TOKENS = 256 * (1024 + 128)


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("Every retained cohort duration must be positive and finite")
    return float(value)


def interval(values):
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError("Exactly three finite paired log ratios required")
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    half = T975_DF2 * sd / math.sqrt(3)
    return {"paired_sweeps": 3, "df": 2, "raw_log_rate_ratios": values,
            "raw_rate_ratios": [math.exp(x) for x in values],
            "mean_log_rate_ratio": mean, "sample_sd_log_rate_ratio": sd,
            "throughput_change_percent": 100 * math.expm1(mean),
            "throughput_nominal_pointwise_95ci_percent": [100 * math.expm1(mean - half), 100 * math.expm1(mean + half)],
            "batch_time_change_percent": 100 * math.expm1(-mean),
            "batch_time_nominal_pointwise_95ci_percent": [100 * math.expm1(-mean - half), 100 * math.expm1(-mean + half)],
            "interpretation": "Descriptive small-n nominal paired log-t interval; three fresh sweep units on one host, no equivalence or SLO guarantee"}


def analyze(records):
    issues, cells, ids, launches = [], {}, set(), []
    if not isinstance(records, list):
        return {"complete": False, "effects": None, "problems": ["Expected normalized list"]}
    if len(records) != 9:
        issues.append(f"Expected exactly nine fresh pairs, got {len(records)}")
    for i, r in enumerate(records):
        try:
            sweep, position, arm = r["sweep"], r["position"], r["arm"]
            if type(sweep) is not int or sweep not in range(3) or type(position) is not int or position not in range(3):
                raise ValueError("Integer sweep/position must be 0..2")
            if ORDERS[sweep][position] != arm or (sweep, arm) in cells:
                raise ValueError("Duplicate cell or nonfrozen arm/period")
            identity = r["launch_id"]
            if not isinstance(identity, str) or not identity.strip() or identity in ids:
                raise ValueError("Fresh unique launch ID required")
            warm, measured = r["warmup_seconds"], r["measurement_seconds"]
            if len(warm) != 3 or len(measured) != 3:
                raise ValueError("Retain exactly three fixed warmup and three measured cohorts")
            warm, measured = [positive(v) for v in warm], [positive(v) for v in measured]
            seconds = statistics.fmean(measured)
            cells[sweep, arm] = seconds
            ids.add(identity)
            launches.append({"sweep": sweep, "position": position, "arm": arm, "launch_id": identity,
                "warmup_seconds": warm, "measurement_seconds": measured, "mean_batch_seconds": seconds,
                "application_tokens_per_second": APPLICATION_TOKENS / seconds,
                "input_tokens_per_second": 262144 / seconds, "consumer_output_tokens_per_second": 32768 / seconds})
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            issues.append(f"Record {i}: {error}")
    expected = {(s, a) for s in range(3) for a in ARMS}
    if set(cells) != expected:
        issues.append("Full three-sweep by three-arm matrix required")
    if issues:
        return {"complete": False, "effects": None, "verified_launch_count": len(cells), "problems": issues,
                "interpretation": "Incomplete or invalid fixed matrix; no partial effect or best sweep selected"}
    effects = {arm + "_vs_stock": interval([math.log(cells[s, "stock"]) - math.log(cells[s, arm]) for s in range(3)])
               for arm in ARMS[1:]}
    return {"complete": True, "effects": effects, "verified_launch_count": 9,
        "launches": sorted(launches, key=lambda x: (x["sweep"], x["position"])),
        "launch_estimator": "Arithmetic mean of all three fixed measured cohort wall durations",
        "paired_anchor": "Same-sweep stock-prefill plus stock-decode real handoff pair",
        "workload": {"logical_requests": 256, "input_tokens_each": 1024, "consumer_output_tokens_each": 128,
                     "concurrency": 64, "sequential_client_waves": 4, "application_tokens_per_cohort": APPLICATION_TOKENS},
        "token_scope": "Count each application input and final consumer output once; exclude discarded producer token and duplicate transferred prompt",
        "units": "Three fresh three-arm sweeps; requests, rounds, ranks and token chunks are not independent replicate units",
        "raw_integrity": "Statistics only; independent exact plan/source/launch closure/KV proof/all-wire audit still required"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite prior statistics")
    raw = args.records.read_bytes()
    result = analyze(json.loads(raw))
    result["normalized_records_sha256"] = hashlib.sha256(raw).hexdigest()
    result["statistics_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if result["complete"] else 2)


if __name__ == "__main__":
    main()
