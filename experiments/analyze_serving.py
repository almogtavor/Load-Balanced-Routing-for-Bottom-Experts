#!/usr/bin/env python3
"""Analyze independent, uninstrumented serve_profile.py server launches.

  python experiments/analyze_serving.py runs/waldorf-20261017/qwen-ep4

Writes serving_analysis.json/.md and serving_launches.csv plus
serving_comparisons.csv under the experiment directory. The experiment manifest
defines expected arms, sweeps, shapes, and rounds; CLI overrides are available
for recovered data without a manifest. Incomplete data are reported and exit 2
unless --allow-incomplete is supplied. Reports are still written on exit 2.

The independent repeat unit is a fresh server launch, paired with stock from
the same sweep. Each launch contributes the median of ALL its measured rounds.
Warmups and profile-only launches are excluded by protocol, never by speed.
Latency entries are medians of round-specific percentiles, not percentiles of
pooled requests. No request or round is counted as an independent repeat.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

DEFAULT_ARMS = ["stock", "noop8", "plain6", "balanced6p1"]
DEFAULT_SHAPES = ["decode", "prefill"]
THROUGHPUT = ["output_tokens_per_second", "total_tokens_per_second", "requests_per_second"]
LATENCIES = ["ttft_ms", "tpot_ms", "request_latency_ms"]
QUANTILES = ["p50", "p95", "p99"]
METRICS = THROUGHPUT + [f"{name}_{q}" for name in LATENCIES for q in QUANTILES]
OPTIONAL_METRICS = [f"itl_ms_{q}" for q in QUANTILES]
ALL_METRICS = METRICS + OPTIONAL_METRICS
# Two-sided 95% Student-t critical values. For an unlisted df, use the next
# smaller listed df, giving a conservative interval without a SciPy dependency.
T95 = {
    1: 12.7062047364, 2: 4.30265272991, 3: 3.18244630528,
    4: 2.7764451052, 5: 2.57058183564, 6: 2.44691185114,
    7: 2.36462425159, 8: 2.3060041352, 9: 2.26215716285,
    10: 2.22813885196, 11: 2.20098516008, 12: 2.17881282966,
    13: 2.16036865646, 14: 2.14478668792, 15: 2.13144954556,
    16: 2.11990529922, 17: 2.10981557783, 18: 2.10092204024,
    19: 2.09302405441, 20: 2.08596344727, 21: 2.07961384473,
    22: 2.0738730679, 23: 2.06865761042, 24: 2.06389856163,
    25: 2.05953855275, 26: 2.05552943864, 27: 2.05183051648,
    28: 2.0484071418, 29: 2.04522964213, 30: 2.0422724563,
    40: 2.02107539031, 60: 2.00029782201, 120: 1.97993040505,
}


def finite(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def mean_ci(values):
    """A t interval for the mean at the paired-launch unit, no pseudo-repeats."""
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None,
                "df": None, "t_critical": None, "critical_value_df": None}
    out = {"n": n, "mean": statistics.mean(values), "sd": None, "ci95": None,
           "df": n - 1, "t_critical": None, "critical_value_df": None}
    if n > 1:
        df = n - 1
        critical_df = sorted(T95)[bisect.bisect_right(sorted(T95), df) - 1]
        t = T95[critical_df]
        sd = statistics.stdev(values)
        radius = t * sd / math.sqrt(n)
        out.update(sd=sd, ci95=[out["mean"] - radius, out["mean"] + radius],
                   t_critical=t, critical_value_df=critical_df)
    return out


def paired_statistics(pairs):
    baselines = [p.get("baseline_value", p.get("stock_value")) for p in pairs]
    ratios = [p["arm_value"] / baseline for p, baseline in zip(pairs, baselines)]
    arithmetic = mean_ci(ratios)
    logs = mean_ci([math.log(r) for r in ratios])
    geometric = math.exp(logs["mean"]) if logs["mean"] is not None else None
    log_ci = [math.exp(v) for v in logs["ci95"]] if logs["ci95"] else None
    return {"n_paired_launches": len(pairs), "ratios": ratios,
            "arithmetic_mean_ratio": arithmetic,
            "mean_log_ratio": logs, "geometric_mean_ratio": geometric,
            "geometric_mean_change_pct": (geometric - 1) * 100 if geometric is not None else None,
            "geometric_ratio_ci95": log_ci,
            "geometric_change_pct_ci95": [(v - 1) * 100 for v in log_ci] if log_ci else None,
            "mean_absolute_difference": mean_ci([p["arm_value"] - baseline for p, baseline in zip(pairs, baselines)])}


def read_json(path):
    try:
        return json.loads(path.read_text()), None
    except (OSError, ValueError) as error:
        return None, f"Cannot read {path}: {error}"


def expected_plan(root, overrides):
    manifest, error = read_json(root / "manifest.json")
    issues = []
    if manifest is None:
        issues.append({"severity": "warning", "message": error + "; expected plan uses CLI/defaults."})
        identity, config = {}, {}
    else:
        identity = manifest.get("identity", {})
        config = identity.get("config", {})
    arms = overrides.get("arms") or config.get("arms") or DEFAULT_ARMS
    shapes = overrides.get("shapes") or config.get("shapes") or DEFAULT_SHAPES
    repeats = overrides.get("expected_repeats") or config.get("repeats") or 3
    rounds = overrides.get("rounds") or config.get("rounds") or 2
    schedule = identity.get("schedule")
    if schedule and not (overrides.get("arms") or overrides.get("expected_repeats")):
        expected = [(row["repeat"], arm) for row in schedule for arm in row["arms"]]
    else:
        expected = [(repeat, arm) for repeat in range(repeats) for arm in arms]
    if "stock" not in arms:
        issues.append({"severity": "error", "message": "Expected plan has no stock baseline."})
    if len(set(expected)) != len(expected):
        issues.append({"severity": "error", "message": "Manifest schedule contains duplicate launch identities."})
    workloads = {w["name"]: w for w in identity.get("workloads", [])}
    metadata = (manifest or {}).get("metadata", {})
    recorded_model = config.get("model")
    model_match = re.search(r"models--([^/]+)/snapshots/([^/]+)", recorded_model or "")
    scope = {"model": model_match.group(1).replace("--", "/") if model_match else recorded_model,
             "experiment_phase": config.get("phase"),
             "experimental_selector": config.get("selector"),
             "arm_definitions": identity.get("arm_definitions", {}),
             "recorded_model_path_or_id": recorded_model,
             "model_snapshot": model_match.group(2) if model_match else config.get("revision"),
             "dp": config.get("dp"), "tp": 1, "expert_parallel": True,
             "all2all_backend": config.get("backend", "deepep_high_throughput"), "batch_tokens": config.get("batch_tokens"),
             "max_num_seqs": config.get("max_num_seqs"), "max_model_len": config.get("max_model_len"),
             "gpu_memory_utilization": config.get("gpu_util"), "server_args": config.get("server_arg"),
             "corpus_file": config.get("text_file"), "corpus_sha256": identity.get("corpus_sha256"),
             "packages": metadata.get("packages", {}),
             "hardware": metadata.get("hardware", {}).get("output"),
             "topology": metadata.get("topology", {}).get("output"),
             "schedule": schedule,
             "runner_script_sha256": identity.get("script_sha256")}
    return {"arms": arms, "shapes": shapes, "rounds_per_shape": rounds,
            "expected_launches": [{"repeat": r, "arm": a} for r, a in expected],
            "workloads": workloads, "scope": scope, "manifest_present": manifest is not None,
            "manifest_file": str(root / "manifest.json"),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
            if manifest is not None else None}, issues


def round_metrics(item):
    values = {name: item.get(name) for name in THROUGHPUT}
    for name in LATENCIES:
        for q in QUANTILES:
            values[f"{name}_{q}"] = (item.get(name) or {}).get(q)
    # A subset's inter-chunk gaps must not be called all-request token ITL.
    all_single = (item.get("requests_with_single_token_chunks") == item.get("requests")
                  and isinstance(item.get("requests"), int) and item["requests"] > 0)
    for q in QUANTILES:
        values[f"itl_ms_{q}"] = (item.get("itl_ms") or {}).get(q) if all_single else None
    return values


def required_metrics(configuration):
    # A one-token pure-prefill request has no per-output-token gap to measure.
    return [metric for metric in METRICS
            if configuration.get("output_tokens") != 1 or not metric.startswith("tpot_ms_")]


def round_errors(item, configuration, index):
    prefix = f"round {index}"
    errors = []
    values = round_metrics(item)
    for name in required_metrics(configuration):
        value = values[name]
        if not finite(value, positive=name in THROUGHPUT):
            errors.append(f"{prefix}: missing/nonfinite/invalid metric {name}")
    if not finite(item.get("elapsed_seconds"), positive=True):
        errors.append(f"{prefix}: elapsed_seconds must be positive")
    requests = item.get("requests")
    if not isinstance(requests, int) or isinstance(requests, bool) or requests <= 0:
        errors.append(f"{prefix}: invalid request count")
        return errors
    for name in ("requests", "concurrency"):
        if configuration.get(name) is None or item.get(name) != configuration.get(name):
            errors.append(f"{prefix}: {name} differs from workload configuration")
    expected = {"prompt_tokens": requests * configuration.get("prompt_tokens", 0),
                "completion_tokens": requests * configuration.get("output_tokens", 0)}
    for name, value in expected.items():
        if not value or item.get(name) != value:
            errors.append(f"{prefix}: {name} differs from configured token budget")
    if "records" in item and len(item["records"]) != requests:
        errors.append(f"{prefix}: request record count differs from requests")
    elapsed = item.get("elapsed_seconds")
    if finite(elapsed, positive=True):
        rates = {"output_tokens_per_second": expected["completion_tokens"] / elapsed,
                 "total_tokens_per_second": sum(expected.values()) / elapsed,
                 "requests_per_second": requests / elapsed}
        for name, value in rates.items():
            if finite(item.get(name)) and not math.isclose(item[name], value, rel_tol=1e-6, abs_tol=1e-6):
                errors.append(f"{prefix}: {name} disagrees with tokens/requests and elapsed time")
    for name in LATENCIES:
        if name == "tpot_ms" and configuration.get("output_tokens") == 1:
            continue
        quantiles = (item.get(name) or {})
        values = [quantiles.get(q) for q in QUANTILES]
        if all(finite(v) for v in values) and values != sorted(values):
            errors.append(f"{prefix}: {name} percentiles are not ordered")
        if quantiles.get("count") != requests:
            errors.append(f"{prefix}: {name} sample count differs from requests")
    if item.get("requests_with_single_token_chunks") == requests:
        itl = item.get("itl_ms") or {}
        expected_gaps = requests * max(configuration.get("output_tokens", 0) - 1, 0)
        if expected_gaps and (itl.get("count") != expected_gaps
                              or not all(finite(itl.get(q)) for q in QUANTILES)):
            errors.append(f"{prefix}: verified single-token streams have missing/inconsistent ITL statistics")
    return errors


def read_launch(path, plan):
    matched = re.fullmatch(r"r(\d+)_(.+)", path.parent.name)
    repeat, arm = int(matched.group(1)), matched.group(2)
    item, error = read_json(path)
    launch = {"repeat": repeat, "arm": arm, "result_file": str(path),
              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
              "source_marked_complete": False, "issues": [], "shapes": {}}
    if error:
        launch["issues"].append(error)
        return launch
    launch["source_marked_complete"] = item.get("complete") is True
    launch["source_error"] = item.get("error")
    for name in ("experiment_phase", "arm_definition", "jit_clean_measurements_observed", "runtime_evidence"):
        if name in item:
            launch[name] = item[name]
    if item.get("repeat") != repeat or item.get("arm") != arm:
        launch["issues"].append("Directory name and recorded repeat/arm disagree")
    if not launch["source_marked_complete"]:
        launch["issues"].append("Server launch is not marked complete")
    if item.get("measurement_type") != "uninstrumented_streaming_http":
        launch["issues"].append("Expected uninstrumented_streaming_http measurement_type")
    observed = item.get("shapes", {})
    if set(observed) != set(plan["shapes"]):
        launch["issues"].append(f"Expected shapes {plan['shapes']}; found {list(observed)}")
    for name, shape in observed.items():
        configuration = shape.get("configuration", {})
        rounds = shape.get("rounds", [])
        errors = []
        if len(rounds) != plan["rounds_per_shape"]:
            errors.append(f"Expected {plan['rounds_per_shape']} measured rounds; found {len(rounds)}")
        expected_config = plan["workloads"].get(name)
        if expected_config and configuration != expected_config:
            errors.append("Recorded workload configuration differs from manifest")
        for index, row in enumerate(rounds):
            errors.extend(round_errors(row, configuration, index))
        metrics = {}
        # Preserve absolute launch medians even for partial/invalid launches;
        # eligibility is explicit and no partial launch enters a comparison.
        for metric in ALL_METRICS:
            values = [round_metrics(row)[metric] for row in rounds]
            metrics[metric] = statistics.median(values) if values and all(finite(v) for v in values) else None
        launch["shapes"][name] = {"configuration": configuration,
            "measured_rounds": len(rounds), "warmup_rounds_excluded": len(shape.get("warmups", [])),
            "median_round_metrics": metrics, "issues": errors,
            "measured_requests": sum(row.get("requests", 0) for row in rounds),
            "verified_single_token_requests": sum(row.get("requests_with_single_token_chunks", 0) for row in rounds),
            "itl_available_all_rounds": bool(rounds) and all(
                all(round_metrics(row)[metric] is not None for metric in OPTIONAL_METRICS) for row in rounds),
            "rounds": [{k: v for k, v in row.items() if k != "records"} for row in rounds]}
    eligible = not launch["issues"] and all(not s["issues"] for s in launch["shapes"].values())
    launch["eligible_for_paired_analysis"] = eligible
    return launch


def supplemental_contrast(indexed, expected, plan, arm, baseline):
    """Same-sweep contrasts diagnose incremental changes; stock stays primary."""
    comparisons = []
    sweeps = sorted(repeat for repeat, name in expected if name == arm)
    for shape in plan["shapes"]:
        paired, missing = [], []
        for repeat in sweeps:
            before, after = indexed.get((repeat, baseline)), indexed.get((repeat, arm))
            reason = None
            if not before or not after:
                reason = "Missing " + "/".join(name for name, value in ((baseline, before), (arm, after)) if not value)
            elif not before.get("eligible_for_paired_analysis") or not after.get("eligible_for_paired_analysis"):
                reason = "Incomplete or invalid baseline/arm launch; see launch issues"
            elif before["shapes"][shape]["configuration"] != after["shapes"][shape]["configuration"]:
                reason = "Baseline/arm workload configurations differ"
            if reason:
                missing.append({"repeat": repeat, "reason": reason})
            else:
                paired.append((repeat, before["shapes"][shape], after["shapes"][shape]))
        for metric in ALL_METRICS:
            pairs, unavailable = [], []
            for repeat, before, after in paired:
                a, b = before["median_round_metrics"][metric], after["median_round_metrics"][metric]
                if not finite(a, positive=True) or not finite(b, positive=True):
                    unavailable.append({"repeat": repeat, "reason": "A ratio requires positive absolute metrics"})
                else:
                    pairs.append({"repeat": repeat, "baseline_value": a, "arm_value": b,
                                  "ratio": b / a, "change_pct": (b / a - 1) * 100})
            comparisons.append({"comparison_role": "supplemental", "arm": arm, "baseline": baseline,
                "shape": shape, "metric": metric,
                "direction": "higher_is_better" if metric in THROUGHPUT else "lower_is_better",
                "expected_paired_launches": len(sweeps), "complete_pairs": not (missing or unavailable),
                "missing_pairs": missing + unavailable, "pairs": pairs,
                **paired_statistics(pairs)})
    return comparisons


def analyze(root, overrides=None):
    overrides = overrides or {}
    plan, issues = expected_plan(root, overrides)
    expected = {(row["repeat"], row["arm"]) for row in plan["expected_launches"]}
    launches = []
    for directory in sorted(root.glob("r*_*/")):
        if re.fullmatch(r"r(\d+)_(.+)", directory.name):
            launches.append(read_launch(directory / "result.json", plan))
    indexed = {}
    for launch in launches:
        key = launch["repeat"], launch["arm"]
        if key in indexed:
            launch["eligible_for_paired_analysis"] = False
            indexed[key]["eligible_for_paired_analysis"] = False
            issues.append({"severity": "error", "message": f"Duplicate launch identity {key}"})
        else:
            indexed[key] = launch
        for message in launch["issues"]:
            issues.append({"severity": "error", "message": f"r{key[0]:02d}_{key[1]}: {message}"})
        if launch.get("jit_clean_measurements_observed") is False:
            issues.append({"severity": "warning", "message": f"r{key[0]:02d}_{key[1]}: observed JIT in measured rounds; all rounds retained, steady-state interpretation is limited."})
        for name, shape in launch["shapes"].items():
            for message in shape["issues"]:
                issues.append({"severity": "error", "message": f"r{key[0]:02d}_{key[1]} {name}: {message}"})
    missing = sorted(expected - indexed.keys())
    unexpected = sorted(indexed.keys() - expected)
    for repeat, arm in missing:
        issues.append({"severity": "error", "message": f"Missing planned launch r{repeat:02d}_{arm}/result.json"})
    for repeat, arm in unexpected:
        issues.append({"severity": "error", "message": f"Unexpected launch r{repeat:02d}_{arm}; retained but excluded from planned comparisons"})
    comparisons = []
    for arm in plan["arms"]:
        if arm == "stock":
            continue
        sweeps = sorted(r for r, a in expected if a == arm)
        for shape in plan["shapes"]:
            pairs, missing_pairs = [], []
            for repeat in sweeps:
                stock, changed = indexed.get((repeat, "stock")), indexed.get((repeat, arm))
                reason = None
                if not stock or not changed:
                    reason = "Missing " + "/".join(n for n, v in (("stock", stock), (arm, changed)) if not v)
                elif not stock.get("eligible_for_paired_analysis") or not changed.get("eligible_for_paired_analysis"):
                    reason = "Incomplete or invalid stock/arm launch; see launch issues"
                elif stock["shapes"][shape]["configuration"] != changed["shapes"][shape]["configuration"]:
                    reason = "Stock/arm workload configurations differ"
                if reason:
                    missing_pairs.append({"repeat": repeat, "reason": reason})
                else:
                    pairs.append((repeat, stock["shapes"][shape], changed["shapes"][shape]))
            if missing_pairs:
                issues.append({"severity": "error", "message": f"{arm}/{shape}: missing valid stock pairs for sweeps "
                               + ", ".join(str(p["repeat"]) for p in missing_pairs)})
            for metric in ALL_METRICS:
                metric_pairs, invalid_ratios = [], []
                for repeat, stock, changed in pairs:
                    before, after = stock["median_round_metrics"][metric], changed["median_round_metrics"][metric]
                    if not finite(before, positive=True) or not finite(after, positive=True):
                        invalid_ratios.append({"repeat": repeat, "reason": "A ratio requires positive absolute metrics"})
                    else:
                        metric_pairs.append({"repeat": repeat, "stock_value": before, "arm_value": after,
                                             "ratio": after / before, "change_pct": (after / before - 1) * 100})
                comparisons.append({"comparison_role": "primary", "arm": arm, "baseline": "stock", "shape": shape, "metric": metric,
                    "direction": "higher_is_better" if metric in THROUGHPUT else "lower_is_better",
                    "required_for_completion": metric in required_metrics(plan["workloads"].get(shape, {})),
                    "expected_paired_launches": len(sweeps), "complete_pairs": not (missing_pairs or invalid_ratios),
                    "missing_pairs": missing_pairs + invalid_ratios, "pairs": metric_pairs,
                    **paired_statistics(metric_pairs)})
    supplemental = []
    for arm, baseline in (("balanced6p1", "noop8"), ("balanced6p1", "plain6")):
        if arm in plan["arms"] and baseline in plan["arms"]:
            definitions = plan["scope"]["arm_definitions"]
            if definitions.get(arm, {}).get("selector") != definitions.get(baseline, {}).get("selector"):
                continue  # A no-op control must use the same implementation.
            supplemental.extend(supplemental_contrast(indexed, expected, plan, arm, baseline))
    valid = sum(bool(launch.get("eligible_for_paired_analysis"))
                and (launch["repeat"], launch["arm"]) in expected for launch in launches)
    complete = not any(i["severity"] == "error" for i in issues) and all(
        c["complete_pairs"] or not c["required_for_completion"] for c in comparisons)
    arm_summaries = []
    for shape in plan["shapes"]:
        for arm in plan["arms"]:
            selected = [launch["shapes"][shape] for launch in launches
                        if launch.get("eligible_for_paired_analysis") and launch["arm"] == arm
                        and (launch["repeat"], arm) in expected]
            medians = {}
            for metric in ALL_METRICS:
                values = [item["median_round_metrics"][metric] for item in selected]
                medians[metric] = statistics.median(values) if values and all(finite(v) for v in values) else None
            arm_summaries.append({"arm": arm, "shape": shape, "independent_launches": len(selected),
                "measured_rounds": sum(item["measured_rounds"] for item in selected),
                "measured_requests": sum(item["measured_requests"] for item in selected),
                "verified_single_token_requests": sum(item["verified_single_token_requests"] for item in selected),
                "itl_available_all_rounds": bool(selected) and all(item["itl_available_all_rounds"] for item in selected),
                "median_across_launch_medians": medians})
    independent_ns = sorted({row["n_paired_launches"] for row in comparisons if row["metric"] == "output_tokens_per_second"})
    uncertainty_counts = []
    for n in independent_ns:
        interval = mean_ci([0.0] * n)
        uncertainty_counts.append(f"n={n}, df={n - 1}, t={interval['t_critical']:.3f}" if n > 1
                                  else f"n={n}, no estimable interval")
    schedule = plan["scope"].get("schedule") or []
    positions = {arm: [sum(row["arms"].index(arm) == position for row in schedule if arm in row["arms"])
                       for position in range(len(plan["arms"]))] for arm in plan["arms"]}
    balanced_positions = bool(schedule) and all(len(set(counts)) == 1 for counts in positions.values())
    return {"schema_version": 2, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_directory": str(root), "plan": plan,
            "completion": {"complete": complete, "expected_launches": len(expected),
                           "observed_launches": len(launches), "valid_complete_launches": valid,
                           "measured_rounds_in_valid_launches": sum(item["measured_rounds"] for item in arm_summaries),
                           "measured_requests_in_valid_launches": sum(item["measured_requests"] for item in arm_summaries),
                           "missing_launches": [{"repeat": r, "arm": a} for r, a in missing],
                           "unexpected_launches": [{"repeat": r, "arm": a} for r, a in unexpected]},
            "issues": issues, "launches": launches, "arm_summaries": arm_summaries, "comparisons": comparisons,
            "supplemental_comparisons": supplemental,
            "method": {"repeat_unit": "independent server launch, paired against stock in the same sweep",
                "supplemental_contrasts": "balanced6p1/noop8 examines incremental routing beyond plugin overhead; balanced6p1/plain6 examines adding load balancing. Both pair launches within the same sweep and retain every measured round.",
                "launch_estimator": "median of every measured round; warmups excluded by protocol",
                "latency_estimator": "median of per-round request percentile; no request pooling",
                "arm_absolute_estimator": "median across complete planned launch medians; descriptive, distinct from the paired-ratio effect estimator",
                "itl_validity": "only available when all measured requests in every included round have verified single-token streaming chunks",
                "effect_estimator": "geometric mean arm/stock launch ratios; arithmetic mean ratios also retained",
                "confidence_interval": "two-sided 95% Student-t interval on mean log paired-launch ratios, exponentiated",
                "t_critical_source": "Standard t table; unlisted df use nearest smaller listed df (conservative)",
                "planned_arm_position_counts": positions,
                "planned_arm_positions_fully_balanced": balanced_positions,
                "filtering": "No throughput/latency filtering or outlier removal. Partial/invalid launches retained but never treated as completed repeats."},
            "interpretation_limits": [
                "Paired-launch uncertainty: " + "; ".join(uncertainty_counts) + ". Small independent-launch counts cannot reliably resolve small effects or establish equivalence.",
                "Intervals assume independent approximately normal launch log-ratios. Shared-node interference, drift and rotated launch order can violate those assumptions.",
                "Node exclusivity is not established by these result files; shared CPU, interconnect or networking contention can confound small differences.",
                ("Planned arm positions are fully balanced; time drift and carryover may still confound comparisons." if balanced_positions else
                 "The planned schedule does not fully counterbalance arm positions; same-sweep pairing does not eliminate order-related drift."),
                "CIs are pointwise and unadjusted across arms, shapes and metrics. They are descriptive uncertainty estimates, not a multiple-comparison decision rule.",
                "Median round percentiles summarize the observed request samples; launch-level intervals do not quantify uncertainty in population p95/p99 tail latency.",
                "ITL is based on client chunk arrival gaps only when every chunk is verified one token. Client/server buffering and transport affect it; TPOT is a per-request average across generated tokens.",
                "A constant ratio can produce a zero-width sample interval; that does not prove zero measurement or deployment uncertainty.",
                "Incomplete-pair estimates use only observed complete pairs and are provisional. Missing results may be related to workload behavior.",
                "Closed-loop fixed-concurrency traffic excludes local executor queue time. These controlled workloads do not establish production arrival-process or fleet performance.",
                "Scope is the exact model snapshot, hardware, EP size and corpus in the manifest. These measurements do not establish results for other model families, EP sizes or inter-node topologies.",
                "Profiles are separate instrumented launches and are excluded from this serving analysis."]}


def number(value, digits=2):
    return f"{value:.{digits}f}" if value is not None else "—"


def table(analysis):
    completion = analysis["completion"]
    scope = analysis["plan"]["scope"]
    lines = ["Serving measurements", "", "Status: " + ("COMPLETE" if completion["complete"] else "INCOMPLETE — estimates are provisional"),
             f"{completion['valid_complete_launches']}/{completion['expected_launches']} planned independent launches validated; "
             f"{completion['measured_rounds_in_valid_launches']} measured shape-rounds and "
             f"{completion['measured_requests_in_valid_launches']} measured requests in valid launches.", "",
             f"Model: **{scope['model'] or 'unknown (manifest missing)'}**; snapshot: `{scope['model_snapshot'] or 'unrecorded'}`.",
             f"DP={scope['dp']}; TP={scope['tp']}; expert parallel, {scope['all2all_backend']}; "
             f"vLLM {scope['packages'].get('vllm', 'unrecorded')}; batch token limit {scope['batch_tokens']}.", "",
             "Each value is the median of every measured round from one server launch. Ratios pair each arm with stock from the same sweep.", ""]
    if scope["experiment_phase"]:
        phase = scope["experiment_phase"]
        lines += [f"Stage: **{phase}**; selector: `{scope['experimental_selector']}`. "
                  + ("Exploratory selection/control data; this is not an independent confirmation." if phase in ("screen", "controls")
                     else "This stage uses its own fresh launches; selection data are not pooled."), ""]
    if scope["hardware"]:
        hardware_rows = list(csv.reader(scope["hardware"].splitlines()))[1:]
        names = [row[2].strip() for row in hardware_rows if len(row) >= 3]
        if names:
            lines += ["Recorded GPUs: " + ", ".join(f"{names.count(name)} × {name}" for name in sorted(set(names))) + ".", ""]
    lines += ["| Workload | Input / output tokens per request | Concurrency | Requests per measured round |",
              "|---|---:|---:|---:|"]
    for name, workload in analysis["plan"]["workloads"].items():
        lines.append(f"| {name} | {workload['prompt_tokens']} / {workload['output_tokens']} | "
                     f"{workload['concurrency']} | {workload['requests']} |")
    lines.append("")
    if analysis["issues"]:
        lines += ["Completion and validation issues:", ""]
        lines += [f"- {i['severity'].upper()}: {i['message']}" for i in analysis["issues"]]
        lines.append("")
    lines += ["Paired output throughput (positive change is faster):", "",
              "| Shape | Arm / stock | Paired launches | Geometric change | 95% t interval | Pair status |",
              "|---|---|---:|---:|---:|---|"]
    for item in analysis["comparisons"]:
        if item["metric"] != "output_tokens_per_second":
            continue
        ci = item["geometric_change_pct_ci95"]
        interval = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "unavailable (n < 2)"
        effect = item["geometric_mean_change_pct"]
        lines.append(f"| {item['shape']} | {item['arm']} | {item['n_paired_launches']}/{item['expected_paired_launches']} | "
                     + (f"{effect:+.2f}%" if effect is not None else "—")
                     + f" | {interval} | {'complete' if item['complete_pairs'] else 'MISSING/INVALID'} |")
    if analysis["supplemental_comparisons"]:
        lines += ["", "Supplemental throughput contrasts (same-sweep launch pairing; stock remains the primary deployment baseline):", "",
                  "| Shape | Contrast | Paired launches | Geometric change | 95% t interval | Pair status |",
                  "|---|---|---:|---:|---:|---|"]
        for item in analysis["supplemental_comparisons"]:
            if item["metric"] != "output_tokens_per_second":
                continue
            ci, effect = item["geometric_change_pct_ci95"], item["geometric_mean_change_pct"]
            interval = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "unavailable (n < 2)"
            lines.append(f"| {item['shape']} | {item['arm']} / {item['baseline']} | "
                         f"{item['n_paired_launches']}/{item['expected_paired_launches']} | "
                         + (f"{effect:+.2f}%" if effect is not None else "—")
                         + f" | {interval} | {'complete' if item['complete_pairs'] else 'MISSING/INVALID'} |")
    lines += ["", "Absolute arm summaries (medians across complete launch medians):", "",
              "Latency entries are p50 / p95 / p99 in ms. Counts describe data volume; requests and rounds are not independent repeats.", "",
              "| Shape | Arm | Launches | Rounds / requests | Output tok/s | Request/s | TTFT ms | TPOT ms | Verified token ITL ms |",
              "|---|---|---:|---:|---:|---:|---|---|---|"]
    for item in analysis["arm_summaries"]:
        metrics = item["median_across_launch_medians"]
        latency = [" / ".join(number(metrics[f"{name}_{q}"]) for q in QUANTILES)
                   for name in ("ttft_ms", "tpot_ms", "itl_ms")]
        lines.append(f"| {item['shape']} | {item['arm']} | {item['independent_launches']} | "
                     f"{item['measured_rounds']} / {item['measured_requests']} | "
                     f"{number(metrics['output_tokens_per_second'])} | {number(metrics['requests_per_second'])} | "
                     + " | ".join(latency) + " |")
    lines += ["", "Verified single-token streams: " + "; ".join(
        f"{item['shape']}/{item['arm']} {item['verified_single_token_requests']}/{item['measured_requests']} requests"
        for item in analysis["arm_summaries"]) + ". ITL requires at least two output tokens and all-request token verification in every included round.", "",
              "Paired median-latency changes vs stock (positive means slower; intervals are 95% t intervals at the launch unit):", "",
              "| Shape | Arm | TTFT p50 change [CI] | TPOT p50 change [CI] | Verified ITL p50 change [CI] |",
              "|---|---|---:|---:|---:|"]
    indexed = {(item["shape"], item["arm"], item["metric"]): item for item in analysis["comparisons"]}
    for shape in analysis["plan"]["shapes"]:
        for arm in analysis["plan"]["arms"]:
            if arm == "stock":
                continue
            cells = []
            for name in ("ttft_ms", "tpot_ms", "itl_ms"):
                item = indexed[shape, arm, name + "_p50"]
                effect, ci = item["geometric_mean_change_pct"], item["geometric_change_pct_ci95"]
                cell = f"{effect:+.2f}%" if effect is not None else "—"
                cell += f" [{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else " [CI unavailable]"
                cells.append(cell)
            lines.append(f"| {shape} | {arm} | " + " | ".join(cells) + " |")
    for shape in analysis["plan"]["shapes"]:
        lines += ["", f"Absolute launch metrics — {shape}", "",
                  "Latency cells are p50 / p95 / p99 in ms, each the median of that percentile across rounds.", "",
                  "| Sweep | Arm | Rounds | Output tok/s | TTFT ms | TPOT ms | Verified ITL ms | Request latency ms | Valid complete launch |",
                  "|---:|---|---:|---:|---|---|---|---|---|"]
        for launch in analysis["launches"]:
            if shape not in launch["shapes"]:
                continue
            item = launch["shapes"][shape]
            metrics = item["median_round_metrics"]
            latency = [" / ".join(number(metrics[f"{name}_{q}"]) for q in QUANTILES)
                       for name in ("ttft_ms", "tpot_ms", "itl_ms", "request_latency_ms")]
            lines.append(f"| {launch['repeat']} | {launch['arm']} | {item['measured_rounds']} | "
                         f"{number(metrics['output_tokens_per_second'])} | " + " | ".join(latency)
                         + f" | {'yes' if launch.get('eligible_for_paired_analysis') else 'NO'} |")
    lines += ["", "Interpretation limits:", ""]
    lines += [f"- {note}" for note in analysis["interpretation_limits"]]
    lines += ["", "JSON retains absolute round summaries, source hashes, all launch metrics, exact matched pairs, and arithmetic/log-ratio intervals. CSVs expose launch metrics and comparisons."]
    return "\n".join(lines) + "\n"


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(analysis, root, args):
    destinations = {"output": args.output or root / "serving_analysis.json",
                    "markdown": args.markdown or root / "serving_analysis.md",
                    "csv": args.csv or root / "serving_launches.csv",
                    "comparisons_csv": args.comparisons_csv or root / "serving_comparisons.csv"}
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations["output"].write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    destinations["markdown"].write_text(table(analysis))
    rows = []
    for launch in analysis["launches"]:
        for shape, data in launch["shapes"].items():
            rows.append({"repeat": launch["repeat"], "arm": launch["arm"], "shape": shape,
                         "complete_valid_launch": launch.get("eligible_for_paired_analysis", False),
                         "measured_rounds": data["measured_rounds"], "result_file": launch["result_file"],
                         "measured_requests": data["measured_requests"],
                         "verified_single_token_requests": data["verified_single_token_requests"],
                         "itl_available_all_rounds": data["itl_available_all_rounds"],
                         **data["median_round_metrics"]})
    write_csv(destinations["csv"], rows, ["repeat", "arm", "shape", "complete_valid_launch", "measured_rounds", "result_file",
                                        "measured_requests", "verified_single_token_requests", "itl_available_all_rounds", *ALL_METRICS])
    rows = []
    fields = ["comparison_role", "arm", "baseline", "shape", "metric", "direction", "n_paired_launches", "expected_paired_launches", "complete_pairs",
              "geometric_mean_ratio", "geometric_mean_change_pct", "ratio_ci95_low", "ratio_ci95_high", "arithmetic_mean_ratio", "df", "t_critical"]
    for comparison in analysis["comparisons"] + analysis["supplemental_comparisons"]:
        ci = comparison["geometric_ratio_ci95"] or [None, None]
        rows.append({**{key: comparison[key] for key in fields if key in comparison},
                     "arithmetic_mean_ratio": comparison["arithmetic_mean_ratio"]["mean"],
                     "ratio_ci95_low": ci[0], "ratio_ci95_high": ci[1],
                     "df": comparison["mean_log_ratio"]["df"], "t_critical": comparison["mean_log_ratio"]["t_critical"]})
    write_csv(destinations["comparisons_csv"], rows, fields)
    return destinations


def self_test():
    # Sanity checks cover known mathematical effects and a missing pair. They
    # do not turn request-level samples into replication or test GPU behavior.
    pairs = [{"repeat": i, "stock_value": value, "arm_value": 2 * value}
             for i, value in enumerate((100, 200, 50))]
    stats = paired_statistics(pairs)
    assert stats["n_paired_launches"] == 3
    assert math.isclose(stats["geometric_mean_ratio"], 2)
    assert all(math.isclose(v, 2) for v in stats["geometric_ratio_ci95"])
    assert stats["mean_log_ratio"]["df"] == 2
    assert math.isclose(stats["mean_log_ratio"]["t_critical"], 4.30265272991)
    assert paired_statistics(pairs[:1])["geometric_ratio_ci95"] is None
    assert paired_statistics([])["geometric_mean_ratio"] is None
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        configuration = {"name": "decode", "prompt_tokens": 10, "output_tokens": 10, "concurrency": 2, "requests": 2}
        manifest = {"identity": {"config": {"repeats": 3, "rounds": 2, "arms": ["stock", "plain6"], "shapes": ["decode"]},
                                 "workloads": [configuration]}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        for repeat in range(3):
            for arm in ("stock", "plain6"):
                elapsed = (1 if arm == "stock" else .5) * (repeat + 1)
                latency = {"count": 2, "p50": 5, "p95": 6, "p99": 7}
                row = {"requests": 2, "concurrency": 2, "prompt_tokens": 20, "completion_tokens": 20,
                       "elapsed_seconds": elapsed, "output_tokens_per_second": 20 / elapsed,
                       "total_tokens_per_second": 40 / elapsed, "requests_per_second": 2 / elapsed,
                       **{name: latency for name in LATENCIES}}
                directory = root / f"r{repeat:02d}_{arm}"
                directory.mkdir()
                (directory / "result.json").write_text(json.dumps({"repeat": repeat, "arm": arm, "complete": True,
                    "measurement_type": "uninstrumented_streaming_http", "shapes": {"decode": {"configuration": configuration, "rounds": [row, row]}}}))
        complete = analyze(root)
        assert complete["completion"]["complete"]
        effect = complete["comparisons"][0]
        assert effect["n_paired_launches"] == 3 and math.isclose(effect["geometric_mean_ratio"], 2)
        (root / "r01_stock" / "result.json").unlink()
        partial = analyze(root)
        assert not partial["completion"]["complete"]
        effect = partial["comparisons"][0]
        assert effect["n_paired_launches"] == 2 and effect["missing_pairs"][0]["repeat"] == 1
        assert partial["completion"]["observed_launches"] == 6  # Missing file is still an observed failed launch directory.
    print("Serving statistics sanity checks passed: constant ratios, launch-level df, and missing pairs.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, help="JSON report path")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", type=Path, help="Absolute launch metric CSV path")
    parser.add_argument("--comparisons-csv", type=Path)
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--shapes", nargs="+")
    parser.add_argument("--allow-incomplete", action="store_true", help="Write provisional reports without exit status 2")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.directory is None:
            return
    if args.directory is None:
        parser.error("An experiment directory is required")
    if any(value is not None and value < 1 for value in (args.expected_repeats, args.rounds)):
        parser.error("Expected repeats and rounds must be positive")
    root = args.directory.resolve()
    if not root.is_dir():
        parser.error(f"Experiment directory does not exist: {root}")
    analysis = analyze(root, vars(args))
    destinations = write_reports(analysis, root, args)
    print(table(analysis), end="")
    print("Reports: " + ", ".join(str(path) for path in destinations.values()), file=sys.stderr)
    if not analysis["completion"]["complete"] and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
