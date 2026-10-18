#!/usr/bin/env python3
"""Summarize timestamped campaign telemetry against measured serving windows.

  python experiments/analyze_telemetry.py --telemetry telemetry.jsonl \
      --results-root qwen-ep4 --output telemetry_summary.json

GPU ranges use actual samples inside measured rounds, excluding warmups and gaps.
Cgroup counter deltas use the closest samples enclosing a workload/launch's first
and last measured times; the bracket may include gaps/warmups, and its overhang is
reported. Per-round brackets are diagnostic and must not be added when overlapping.
Ten-second telemetry cannot identify kernel bottlenecks or exclude brief bursts.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import statistics


LIMITS = [
    "GPU measurements are sparse point samples, nominally every 10 seconds. They cannot attribute kernel timings, establish isolation, or rule out brief utilization/power/clock changes.",
    "The monitor timestamps each row before the GPU query. Query latency is not recorded, so GPU sample timestamps have additional uncertainty.",
    "Cgroup counters cover the whole pod/cgroup, including client/server/monitor processes and possibly multiple workers, not an individual model rank.",
    "Counter deltas use enclosing samples; their interval can include workload gaps or warmups. Bracket overhang is reported. Do not sum overlapping per-round intervals.",
    "throttled_usec is reported as a counter delta, not a fraction of wall time or an amount of request latency. The throttled-period fraction describes quota enforcement, not its performance impact.",
]


def counters(text):
    output = {}
    for line in (text or "").splitlines():
        pieces = line.split()
        if len(pieces) == 2:
            try:
                output[pieces[0]] = int(pieces[1])
            except ValueError:
                pass
    return output


def counter_delta(before, after):
    old, new = counters(before), counters(after)
    missing = sorted(set(old) ^ set(new))
    decreases = sorted(k for k in set(old) & set(new) if new[k] < old[k])
    return {"available": bool(old) and bool(new) and not decreases,
            "delta": {k: new[k] - old[k] for k in sorted(set(old) & set(new))} if not decreases else None,
            "missing_counter_keys": missing, "decreased_counter_keys": decreases}


def describe(values):
    return {"samples": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values)} if values else None


def load_telemetry(path):
    samples, invalid = [], []
    with path.open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["unix_seconds"] = float(row["unix_seconds"])
                samples.append(row)
            except (ValueError, TypeError, KeyError):
                invalid.append(number)
    samples.sort(key=lambda r: r["unix_seconds"])
    return samples, invalid


def parse_gpus(row):
    """Return numeric readings and unavailable fields; do not turn N/A into zero."""
    readings, missing = {}, Counter()
    if row.get("query_timeout") or row.get("query_returncode") != 0:
        return readings, missing
    fields = row.get("gpu_fields", [])
    for values in csv.reader(io.StringIO("\n".join(row.get("gpu_csv", [])))):
        if len(values) != len(fields):
            missing["malformed_gpu_csv_rows"] += 1
            continue
        data = dict(zip(fields, (x.strip() for x in values)))
        index = data.get("index")
        if index is None:
            missing["gpu_index"] += 1
            continue
        readings[index] = {}
        for name, value in data.items():
            if name == "index":
                continue
            try:
                readings[index][name] = float(value)
            except ValueError:
                missing[f"gpu{index}:{name}"] += 1
    return readings, missing


def window_summary(samples, windows):
    starts = [r["unix_seconds"] for r in samples]
    start, finish = min(a for a, b in windows), max(b for a, b in windows)
    inside = [row for row in samples if any(a <= row["unix_seconds"] <= b for a, b in windows)]
    interval = [row for row in samples if start <= row["unix_seconds"] <= finish]
    per_gpu = defaultdict(lambda: defaultdict(list))
    unavailable = Counter()
    failed = timed_out = empty = 0
    for row in inside:
        failed += row.get("query_returncode") not in (None, 0)
        timed_out += bool(row.get("query_timeout"))
        readings, missing = parse_gpus(row)
        unavailable.update(missing)
        if not readings:
            empty += 1
        for gpu, fields in readings.items():
            for field, value in fields.items():
                per_gpu[gpu][field].append(value)
    before_i = bisect.bisect_right(starts, start) - 1
    after_i = bisect.bisect_left(starts, finish)
    bracket = {"available": before_i >= 0 and after_i < len(samples),
               "missing_before": before_i < 0, "missing_after": after_i >= len(samples)}
    if bracket["available"]:
        before, after = samples[before_i], samples[after_i]
        cpu = counter_delta(before.get("cpu_stat"), after.get("cpu_stat"))
        memory = counter_delta(before.get("memory_events"), after.get("memory_events"))
        delta = cpu.get("delta") or {}
        periods = delta.get("nr_periods", 0)
        cpu["throttled_period_fraction"] = delta["nr_throttled"] / periods if periods and "nr_throttled" in delta else None
        bracket.update({"start_unix_seconds": before["unix_seconds"],
                        "finish_unix_seconds": after["unix_seconds"],
                        "interval_seconds": after["unix_seconds"] - before["unix_seconds"],
                        "overhang_before_seconds": start - before["unix_seconds"],
                        "overhang_after_seconds": after["unix_seconds"] - finish,
                        "cpu_stat": cpu, "memory_events": memory})
    memory_current = []
    for row in inside:
        try:
            memory_current.append(int(row["memory_current"]))
        except (TypeError, ValueError, KeyError):
            pass
    return {"first_measured_unix_seconds": start, "last_measured_unix_seconds": finish,
            "measured_windows": [{"start": a, "finish": b} for a, b in windows],
            "total_measured_seconds": sum(b - a for a, b in windows),
            "span_seconds": finish - start, "span_gap_seconds": finish - start - sum(b - a for a, b in windows),
            "telemetry_samples_inside_measurements": len(inside),
            "telemetry_samples_in_span_including_gaps": len(interval),
            "gpu_query_failures_inside_measurements": failed,
            "gpu_query_timeouts_inside_measurements": timed_out,
            "gpu_queries_without_status_inside_measurements": sum(r.get("query_returncode") is None and not r.get("query_timeout") for r in inside),
            "samples_without_gpu_readings_inside_measurements": empty,
            "unavailable_gpu_fields": dict(unavailable),
            "gpus": {gpu: {field: describe(values) for field, values in fields.items()}
                     for gpu, fields in sorted(per_gpu.items())},
            "memory_current_bytes_sampled": describe(memory_current),
            "enclosing_counter_bracket": bracket}


def analyze(samples, result_files):
    launches = []
    for path in sorted(result_files):
        result = json.loads(path.read_text())
        if result.get("measurement_type") != "uninstrumented_streaming_http" or not result.get("complete"):
            continue
        launch = {"result_file": str(path), "arm": result["arm"], "repeat": result["repeat"], "workloads": {}}
        launch_windows = []
        for name, shape in result.get("shapes", {}).items():
            windows = [(r["started_unix_ns"] / 1e9, r["finished_unix_ns"] / 1e9)
                       for r in shape.get("rounds", []) if "started_unix_ns" in r and "finished_unix_ns" in r]
            if not windows:
                continue
            summary = window_summary(samples, windows)
            summary["rounds"] = [window_summary(samples, [window]) for window in windows]
            launch["workloads"][name] = summary
            launch_windows.extend(windows)
        if launch_windows:
            launch["all_measured_workloads"] = window_summary(samples, sorted(launch_windows))
            launches.append(launch)
    return launches


def self_test():
    fields = "index,utilization.gpu,power.draw,clocks.sm".split(",")
    def sample(t, periods, throttled, oom=0, bad=False):
        return {"unix_seconds": t, "cpu_stat": f"nr_periods {periods}\nnr_throttled {throttled}\nthrottled_usec {throttled * 100}\n",
                "memory_events": f"oom {oom}\noom_kill {oom}\n", "memory_current": "1000",
                "gpu_fields": fields, "query_returncode": 1 if bad else 0,
                "gpu_csv": ["0, 90, 200, 1500", "1, N/A, 180, 1400"]}
    samples = [sample(0, 0, 0), sample(10, 10, 2), sample(20, 20, 5, oom=1, bad=True), sample(30, 30, 6)]
    summary = window_summary(samples, [(11, 19)])
    bracket = summary["enclosing_counter_bracket"]
    assert bracket["start_unix_seconds"] == 10 and bracket["finish_unix_seconds"] == 20
    assert bracket["cpu_stat"]["delta"]["nr_throttled"] == 3
    assert bracket["cpu_stat"]["throttled_period_fraction"] == .3
    assert bracket["memory_events"]["delta"]["oom_kill"] == 1
    assert not summary["gpus"] and summary["telemetry_samples_inside_measurements"] == 0
    summary = window_summary(samples, [(9, 21)])
    assert summary["gpu_query_failures_inside_measurements"] == 1
    assert summary["gpus"]["0"]["utilization.gpu"]["min"] == 90
    assert "utilization.gpu" not in summary["gpus"]["1"]
    assert summary["unavailable_gpu_fields"]["gpu1:utilization.gpu"] == 1
    assert not window_summary(samples, [(-1, 1)])["enclosing_counter_bracket"]["available"]
    assert counter_delta("nr_periods 5", "nr_periods 2")["delta"] is None
    print("11 telemetry interval/counter/missing-data checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.telemetry:
            return
    if not args.telemetry or not args.results_root or not args.output:
        parser.error("--telemetry, --results-root, and --output are required")
    samples, invalid = load_telemetry(args.telemetry)
    gaps = [b["unix_seconds"] - a["unix_seconds"] for a, b in zip(samples, samples[1:])]
    launches = analyze(samples, args.results_root.glob("r*_*/result.json"))
    result = {"telemetry_file": str(args.telemetry.resolve()), "sample_count": len(samples),
              "invalid_json_or_timestamp_line_numbers": invalid,
              "observed_sampling_interval_seconds": describe(gaps),
              "telemetry_start_unix_seconds": samples[0]["unix_seconds"] if samples else None,
              "telemetry_finish_unix_seconds": samples[-1]["unix_seconds"] if samples else None,
              "total_gpu_query_failures": sum(r.get("query_returncode") not in (None, 0) for r in samples),
              "total_gpu_query_timeouts": sum(bool(r.get("query_timeout")) for r in samples),
              "total_gpu_queries_without_status": sum(r.get("query_returncode") is None and not r.get("query_timeout") for r in samples),
              "limits": LIMITS, "launches": launches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Summarized {len(samples)} samples against {len(launches)} completed launches -> {args.output}")
    for launch in launches:
        for name, window in launch["workloads"].items():
            bracket = window["enclosing_counter_bracket"]
            cpu = bracket.get("cpu_stat", {})
            delta = cpu.get("delta") or {}
            print(f"r{launch['repeat']} {launch['arm']} {name}: GPU samples={window['telemetry_samples_inside_measurements']}; "
                  f"CPU nr_throttled delta={delta.get('nr_throttled')}; "
                  f"throttled-period fraction={cpu.get('throttled_period_fraction')}; "
                  f"enclosing interval={bracket.get('interval_seconds')}s")


if __name__ == "__main__":
    main()
