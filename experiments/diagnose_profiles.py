#!/usr/bin/env python3
"""Audit profiled CPU/GPU timelines and conditional Amdahl budgets.

This complements analyze_profiles.py with interval unions and inter-rank arrival
skew. Missing GPU activity is not proof of CPU saturation: launch overhead,
client pacing, scheduling and dependencies can all create gaps. All conclusions
are about instrumented profiles (OMP4); clean serving used OMP1.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from analyze_profiles import classify, find_raw


def summarize(values):
    if not values:
        return None
    ordered = sorted(values)
    def q(p):
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"count": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "p50": q(.5), "p95": q(.95), "max": ordered[-1]}


def intervals_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def clipped(intervals, start, finish):
    return [(max(start, a), min(finish, b)) for a, b in intervals if a < finish and b > start]


def duration(intervals):
    return sum(b - a for a, b in intervals_union(intervals))


def gaps(intervals, start, finish):
    merged = intervals_union(clipped(intervals, start, finish))
    pointer = start
    output = []
    for lo, hi in merged:
        if lo > pointer:
            output.append(lo - pointer)
        pointer = max(pointer, hi)
    if pointer < finish:
        output.append(finish - pointer)
    return output


def correlation(xs, ys):
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    ax, ay = statistics.mean(xs), statistics.mean(ys)
    return sum((x - ax) * (y - ay) for x, y in zip(xs, ys)) / math.sqrt(
        sum((x - ax) ** 2 for x in xs) * sum((y - ay) ** 2 for y in ys))


def read_rank(path, workloads, rank):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        trace = json.load(source)
    base = trace["baseTimeNanoseconds"]
    gpu = defaultdict(list)
    apis, cpu_ops, user_annotations = [], [], []
    moe = []
    op_durations, annotation_durations = defaultdict(list), defaultdict(list)
    categories = Counter()
    cpu_records = defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        name, cat = event.get("name", "unknown"), event.get("cat", "")
        start, length = event.get("ts", 0), event.get("dur", 0)
        categories[cat] += 1
        interval = (start, start + length)
        if cat in ("cpu_op", "cuda_runtime", "cuda_driver", "user_annotation"):
            cpu_records[(event.get("pid"), event.get("tid"))].append((start, start + length, cat, name))
        if cat == "kernel":
            gpu[classify(name)].append(interval)
        elif cat in ("cuda_runtime", "cuda_driver"):
            apis.append(interval)
            op_durations[name].append(length)
        elif cat == "cpu_op":
            cpu_ops.append(interval)
            if name == "vllm::moe_forward":
                dims = event.get("args", {}).get("Input Dims")
                moe.append({"start": start, "finish": start + length,
                            "input_tokens": dims[0][0] if dims and dims[0] else None,
                            "pid": event.get("pid"), "tid": event.get("tid")})
        elif cat == "user_annotation":
            user_annotations.append(interval)
            annotation_durations[name].append(length)
    for values in gpu.values():
        values.sort()
    moe.sort(key=lambda x: x["start"])
    host_self = defaultdict(lambda: defaultdict(float))
    main_thread = (moe[0]["pid"], moe[0]["tid"]) if moe else None
    def close_host_scope(scope):
        lo, hi, category, label, children = scope
        for phase_name, window in workloads.items():
            ws, we = ((window[k] - base) / 1000 for k in ("started_unix_ns", "finished_unix_ns"))
            bounded = clipped([(lo, hi)], ws, we)
            if bounded:
                exclusive = duration(bounded) - duration(clipped(children, max(lo, ws), min(hi, we)))
                host_self[phase_name][category + ":" + label] += max(0, exclusive)
    stack = []
    for lo, hi, category, label in sorted(cpu_records.get(main_thread, []), key=lambda e: (e[0], -e[1])):
        while stack and lo >= stack[-1][1]:
            close_host_scope(stack.pop())
        if stack:
            stack[-1][4].append((lo, min(hi, stack[-1][1])))
        stack.append([lo, hi, category, label, []])
    while stack:
        close_host_scope(stack.pop())
    phase_data = {}
    for name, window in workloads.items():
        start = (window["started_unix_ns"] - base) / 1000
        finish = (window["finished_unix_ns"] - base) / 1000
        wall = finish - start
        all_gpu = [x for values in gpu.values() for x in clipped(values, start, finish)]
        gpu_union = duration(all_gpu)
        per_category = {category: {"summed_event_us": sum(b - a for a, b in clipped(values, start, finish)),
                                   "union_us": duration(clipped(values, start, finish)),
                                   "calls_starting_in_window": sum(start <= a < finish for a, b in values)}
                        for category, values in gpu.items()}
        local_moe = [x for x in moe if start <= x["start"] < finish]
        phase_data[name] = {"wall_us": wall, "gpu_kernel_union_us": gpu_union,
                           "gpu_kernel_union_fraction": gpu_union / wall,
                           "no_observed_gpu_kernel_fraction": 1 - gpu_union / wall,
                           "gpu_idle_gap_us": summarize(gaps(all_gpu, start, finish)),
                           "gpu_idle_gaps_gt100us": sum(x > 100 for x in gaps(all_gpu, start, finish)),
                           "cpu_torch_op_union_us": duration(clipped(cpu_ops, start, finish)),
                           "cpu_cuda_api_union_us": duration(clipped(apis, start, finish)),
                           "cpu_user_annotation_union_us": duration(clipped(user_annotations, start, finish)),
                           "cpu_moe_forward_duration_us": summarize([x["finish"] - x["start"] for x in local_moe]),
                           "cpu_gap_between_moe_scopes_us": summarize([max(0, b["start"] - a["finish"]) for a, b in zip(local_moe, local_moe[1:])]),
                           "worker_main_thread_exclusive_host_scope_us": dict(sorted(host_self[name].items(), key=lambda pair: pair[1], reverse=True)[:25]),
                           "host_scope_note": "Exclusive traced host-scope elapsed time, not CPU utilization. Opaque MoE/compiled-graph self scopes include untraced Python/C++ work and waits; Python function names are unavailable because with_stack=False.",
                           "source_input_tokens_per_moe_forward": dict(Counter(x["input_tokens"] for x in local_moe)),
                           "categories": per_category}
    return {"rank": rank, "base_time_ns": base, "phases": phase_data,
            "cuda_api_durations_us": {name: summarize(values) for name, values in sorted(op_durations.items(), key=lambda x: sum(x[1]), reverse=True)[:15]},
            "user_annotation_durations_us": {name: summarize(values) for name, values in sorted(annotation_durations.items(), key=lambda x: sum(x[1]), reverse=True)[:15]},
            "trace_categories": dict(categories)}, gpu, moe


def diagnose_result(result_file, roots):
    result = json.loads(result_file.read_text())
    ranks, sequences, moe_sequences = [], {}, {}
    for summary in result["trace_summary"]["traces"]:
        rank = summary.get("dp_rank")
        if rank is None:
            continue
        path = find_raw(summary["path"], [result_file.parent, *roots])
        if path is None:
            raise FileNotFoundError(f"A raw trace is required: {summary['path']}")
        report, gpu, moe = read_rank(path, result["shapes"], rank)
        ranks.append(report)
        sequences[rank], moe_sequences[rank] = gpu, moe
    ranks.sort(key=lambda x: x["rank"])
    if len({r["base_time_ns"] for r in ranks}) != 1:
        raise ValueError("Trace clock origins differ; align origins before comparing ranks")
    base = ranks[0]["base_time_ns"]
    phases = {}
    for phase, window in result["shapes"].items():
        start, finish = ((window[k] - base) / 1000 for k in ("started_unix_ns", "finished_unix_ns"))
        wall = finish - start
        all_gpu = [interval for gpu in sequences.values() for values in gpu.values() for interval in values]
        any_gpu = duration(clipped(all_gpu, start, finish))
        phase_result = {"wall_us": wall, "any_rank_kernel_union_us": any_gpu,
                        "any_rank_kernel_union_fraction": any_gpu / wall,
                        "no_rank_observed_kernel_gap_us": summarize(gaps(all_gpu, start, finish)),
                        "conditional_amdahl": {}, "collective_arrival_skew": {}}
        for label, categories in {"router": ("router_stock", "router_placement"),
                                  "dispatch_payload": ("dispatch_payload",),
                                  "dispatch_and_combine_payload": ("dispatch_payload", "combine_payload"),
                                  "expert_gemm": ("expert_gemm",)}.items():
            max_rank_sum = max(sum(r["phases"][phase]["categories"].get(c, {}).get("summed_event_us", 0) for c in categories) for r in ranks)
            f = max_rank_sum / wall
            phase_result["conditional_amdahl"][label] = {"max_rank_summed_us": max_rank_sum,
                "fraction_of_profiled_wall": f, "ideal_speedup_if_all_removed": 1 / (1 - f) if f < 1 else None,
                "ideal_wall_latency_reduction_if_event_duration_cut20pct": .2 * f,
                "condition": "Illustrative serial-cost budget: assumes this rank's summed events are fully on the critical path, no overlap/shape/scheduler changes, and no induced waiting changes. Not a causal or universal performance bound."}
        for category in ("dispatch_notify_wait", "dispatch_payload", "combine_notify_wait", "combine_payload"):
            values = [sequences[rank].get(category, []) for rank in sorted(sequences)]
            counts = [len(x) for x in values]
            if len(set(counts)) != 1:
                phase_result["collective_arrival_skew"][category] = {"matching_call_counts": False, "counts": counts}
                continue
            arrival, completion, durations, arrival_lead, lead_durations = [], [], [], [], []
            groups = 0
            invalid = 0
            vectors = Counter()
            for i in range(counts[0]):
                group = [v[i] for v in values]
                if not all(start <= a < finish for a, b in group):
                    continue
                groups += 1
                latest_start = max(a for a, b in group)
                arrival.append(latest_start - min(a for a, b in group))
                completion.append(max(b for a, b in group) - min(b for a, b in group))
                invalid += any(b + 20 < latest_start for a, b in group)
                for a, b in group:
                    durations.append(b - a)
                    arrival_lead.append(latest_start - a)
                    lead_durations.append(b - a)
                if i < min(len(moe_sequences[r]) for r in moe_sequences):
                    vectors[tuple(moe_sequences[r][i]["input_tokens"] for r in sorted(moe_sequences))] += 1
            phase_result["collective_arrival_skew"][category] = {
                "matching_call_counts": True, "global_call_counts": counts,
                "groups_inside_workload": groups,
                "arrival_skew_us": summarize(arrival), "completion_skew_us": summarize(completion),
                "rank_kernel_duration_us": summarize(durations),
                "arrival_lead_duration_pearson": correlation(arrival_lead, lead_durations),
                "groups_with_completion_before_latest_start_gt20us": invalid,
                "source_token_vectors": [{"tokens_by_rank": list(k), "calls": v} for k, v in vectors.most_common()],
                "pairing_note": "Same ordinal kernel call across ranks, requiring equal global call counts. This approximates matching collective order; it does not replace an explicit sequence ID."}
        # Exact all-rank source-token strata avoid the local-T-only confound.
        # GPU router and communication calls occur once per MoE invocation on
        # every rank; equal counts/order make ordinal matching checkable.
        source_strata = defaultdict(lambda: defaultdict(list))
        gpu_routers = {rank: sorted(sequences[rank].get("router_stock", []) + sequences[rank].get("router_placement", [])) for rank in sequences}
        shared_count = min([len(v) for v in gpu_routers.values()] + [len(v) for v in moe_sequences.values()])
        for i in range(shared_count):
            routers = [gpu_routers[rank][i] for rank in sorted(sequences)]
            if not all(start <= a < finish for a, b in routers):
                continue
            vector = tuple(moe_sequences[rank][i]["input_tokens"] for rank in sorted(sequences))
            strata = source_strata[vector]
            combined = [sequences[rank]["combine_payload"][i] for rank in sorted(sequences)]
            strata["all_rank_gpu_router_to_combine_envelope_us"].append(max(b for a, b in combined) - min(a for a, b in routers))
            for category in ("dispatch_notify_wait", "dispatch_payload", "combine_notify_wait", "combine_payload"):
                grouped = [sequences[rank][category][i] for rank in sorted(sequences)]
                strata[category + "_rankmax_us"].append(max(b - a for a, b in grouped))
                strata[category + "_all_rank_envelope_us"].append(max(b for a, b in grouped) - min(a for a, b in grouped))
            strata["router_rankmax_us"].append(max(b - a for a, b in routers))
            strata["router_start_skew_us"].append(max(a for a, b in routers) - min(a for a, b in routers))
        phase_result["exact_all_rank_source_token_strata"] = [
            {"source_tokens_by_rank": list(vector), "collective_calls": len(next(iter(metrics.values()))),
             "metrics_us": {key: summarize(values) for key, values in metrics.items()}}
            for vector, metrics in source_strata.items()]
        phase_result["strata_limits"] = "All-rank source-T is controlled within each stratum, but original logits/request allocation, received expert loads, CPU/profiler behavior and rank readiness still differ. Layer calls within one step are not independent experiment repeats. Envelopes include async host/stream gaps, and do not isolate wire transfer."
        phases[phase] = phase_result
    return {"arm": result["arm"], "result_file": str(result_file), "ranks": ranks, "phases": phases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--trace-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profiles = []
    for path in args.results:
        profiles.append(diagnose_result(path, args.trace_root))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"profiles": profiles,
            "limits": "Instrumented OMP4 profiles are not uninstrumented OMP1 performance. GPU gaps do not prove CPU saturation. Kernels overlap; Amdahl estimates are explicitly conditional, not causal bounds. Source-local T does not control received-expert work or other ranks."}, indent=2) + "\n")
        print(f"Analyzed {path}", flush=True)


if __name__ == "__main__":
    main()
