#!/usr/bin/env python3
"""Classify serving GPU profiles without confusing kernel sums with latency.

  python experiments/analyze_profiles.py profile_stock/result.json \
      profile_balanced6p1/result.json --output profile_analysis.json

Uses raw .pt.trace.json[.gz] files when available at their recorded path or under
--trace-root. Otherwise uses the retained top-40 kernel summaries, explicitly
marking them as incomplete lower bounds. The category rules are exclusive and
ordered: FlashAttention is never counted as GEMM/DeepEP combine, and DeepEP notify
kernels are separated from payload kernels. Generic GEMM attribution remains a
candidate classification unless a name identifies the expert implementation.

Profiles mix prefill/decode and instrumented scheduling may differ between arms.
Per-call and per-router-call normalization do not control token counts, batch
shapes, overlap, or rank synchronization; they cannot establish serving speedups.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

CATEGORY_NOTES = {
    "dispatch_notify_wait": "DeepEP dispatch notification/coordination; includes synchronization and bookkeeping, not a pure payload-transfer timer.",
    "dispatch_payload": "DeepEP dispatch payload kernel; duration can still include waits and copies, not just wire/NVLink transfer.",
    "combine_notify_wait": "DeepEP combine notification/coordination; duration can include waiting for other ranks.",
    "combine_payload": "DeepEP combine payload/reduction kernel; not a pure bandwidth timer.",
    "dispatch_layout_pack": "DeepEP dispatch layout calculation and explicit packing helpers.",
    "deepep_other": "Other DeepEP kernels, retained separately for inspection.",
    "expert_gemm": "Kernels named fused_moe_kernel or explicitly grouped/MoE GEMM; generic fused_moe_kernel is Triton expert GEMM in this vLLM configuration.",
    "router_stock": "Stock MoE top-k gating kernels; excludes generic sampling top-k.",
    "router_placement": "Placement-aware Triton _par_select kernel.",
    "expert_pack_sort_align": "Expert token count/sort/permutation/alignment operations.",
    "expert_activation": "MoE-path act_and_mul activation; a name-based attribution.",
    "expert_reduce": "Expert weighted output summation, separate from DeepEP combine.",
    "attention": "Attention/FlashAttention kernels, including FlashAttnFwdCombine and cutlass::device_kernel<flash::...>.",
    "dense_gemm_candidate": "Generic cuBLAS/nvjet/GEMM kernels, excluding named expert and attention kernels. Usually dense projections here; names alone cannot prove the originating layer.",
    "cache": "KV-cache reshape/update kernels.",
    "normalization": "Named RMS/layer norm kernels, possibly fused with adjacent elementwise operations.",
    "sampling": "Sampling and sampled/draft-token bookkeeping.",
    "elementwise_or_other": "Kernels without a more specific identified category; retained by full name.",
}


def classify(name: str) -> str:
    low = name.lower()
    if "deep_ep" in low or "deepep" in low:
        if "notify" in low and "dispatch" in low:
            return "dispatch_notify_wait"
        if "notify" in low and "combine" in low:
            return "combine_notify_wait"
        if "get_dispatch_layout" in low or "::layout::" in low or "pack" in low:
            return "dispatch_layout_pack"
        if re.search(r"::dispatch(?:<|\()", low):
            return "dispatch_payload"
        if re.search(r"::combine(?:<|\()", low):
            return "combine_payload"
        return "deepep_other"
    if "_par_select" in low:
        return "router_placement"
    if re.search(r"topkgating|fused_topk|topk_softmax|topk_sigmoid|fused_moe_router", low):
        return "router_stock"
    if re.search(r"fused_moe_kernel|grouped_gemm|grouped_mm|moe_gemm|moe.*gemm|gemm.*moe", low):
        return "expert_gemm"
    if re.search(r"moe_align|count_and_sort_expert|moe_permute|moe_unpermute|permute.*moe|moe.*permute", low):
        return "expert_pack_sort_align"
    if "act_and_mul" in low:
        return "expert_activation"
    if "moe_sum" in low:
        return "expert_reduce"
    if re.search(r"flashattn|flash_attn|flash_attention|flash::|fmha|paged_attention", low):
        return "attention"
    if re.search(r"sample|sampling|gumbel|draft_token", low):
        return "sampling"
    if re.search(r"nvjet_|cublas|gemm|matmul|grouped_mm|device_kernel<cutlass::", low):
        return "dense_gemm_candidate"
    if re.search(r"reshape_and_cache|cache.*kernel|kv_cache", low):
        return "cache"
    if re.search(r"rms_norm|layer_norm|layernorm", low):
        return "normalization"
    return "elementwise_or_other"


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    def q(fraction):
        pos = (len(values) - 1) * fraction
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"min_us": values[0], "p50_us": q(.5), "p95_us": q(.95),
            "p99_us": q(.99), "max_us": values[-1]}


def finalize_groups(groups):
    output = {}
    for name, item in groups.items():
        durations = item.get("durations")
        calls = item["calls"]
        output[name] = {"calls": calls, "summed_gpu_duration_us": item["total"],
                        "mean_us_per_call": item["total"] / calls if calls else None,
                        "per_call_distribution": distribution(durations),
                        "kernel_names": sorted(item.get("names", []))}
    return output


def add(groups, name, calls, total, durations=None, kernel_name=None):
    item = groups.setdefault(name, {"calls": 0, "total": 0.0, "names": set(), "durations": []})
    item["calls"] += calls
    item["total"] += total
    if durations is not None and item["durations"] is not None:
        item["durations"].extend(durations)
    else:
        item["durations"] = None
    if kernel_name:
        item["names"].add(kernel_name)


def find_raw(recorded_path: str, roots: list[Path]):
    path = Path(recorded_path)
    if path.is_file():
        return path
    for root in roots:
        if root.is_file() and root.name == path.name:
            return root
        if root.is_dir():
            found = list(root.rglob(path.name))
            if found:
                return sorted(found)[0]
            # A downloaded single-rank trace may be renamed for convenience.
            rank = re.search(r"(?:^|_)dp(\d+)(?:_|\.)", path.name)
            if rank:
                for ending in (".pt.trace.json.gz", ".pt.trace.json"):
                    short = root / ("dp" + rank.group(1) + ending)
                    if short.is_file():
                        return short
    return None


def summarize_raw(path: Path, workload_windows=None):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        trace = json.load(f)
    by_kernel, categories = {}, {}
    cpu_events = 0
    kernel_events = 0
    shape_counts = Counter()
    gpu_intervals = []
    kernel_records = []
    moe_scopes = defaultdict(list)
    launches = {}
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", "")).lower()
        name = event.get("name", "unknown")
        if "kernel" in cat:
            duration = float(event.get("dur", 0))
            if duration < 0:
                continue
            kernel_events += 1
            kernel_records.append(event)
            add(by_kernel, name, 1, duration, [duration], name)
            add(categories, classify(name), 1, duration, [duration], name)
            if isinstance(event.get("ts"), (int, float)):
                gpu_intervals.append((event["ts"], event["ts"] + duration))
        elif cat in ("cuda_runtime", "cuda_driver"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = (event.get("pid"), event.get("tid"), event.get("ts", 0))
        elif cat in ("cpu_op", "python_function", "user_annotation"):
            cpu_events += 1
            args = event.get("args", {})
            if name == "vllm::moe_forward":
                dims = args.get("Input Dims") or args.get("Input dims")
                if dims and dims[0] and isinstance(dims[0][0], int):
                    moe_scopes[(event.get("pid"), event.get("tid"))].append(
                        (event["ts"], event["ts"] + event["dur"], dims[0][0]))
            # This histogram exposes batch-shape differences without claiming
            # that temporal proximity identifies a kernel's parent operator.
            if re.search(r"moe_forward|fused_moe|fused_topk|select_expert|topk_softmax", name, re.I):
                dims = args.get("Input Dims") or args.get("Input dims")
                if dims:
                    shape_counts[(name, json.dumps(dims, sort_keys=True))] += 1
    # Attribute each GPU kernel through its CUDA launch correlation to a CPU
    # MoE scope on that launch thread. GPU timestamps alone cannot establish
    # parentage because asynchronous kernels may outlive the CPU scope.
    starts = {}
    for thread, scopes in moe_scopes.items():
        scopes.sort()
        starts[thread] = [scope[0] for scope in scopes]
    by_tokens, by_workload = defaultdict(dict), defaultdict(dict)
    attributed = 0
    base_ns = trace.get("baseTimeNanoseconds")
    for event in kernel_records:
        category = classify(event.get("name", "unknown"))
        duration = float(event.get("dur", 0))
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch:
            thread, launch_ts = launch[:2], launch[2]
            scopes = moe_scopes.get(thread, [])
            position = bisect.bisect_right(starts.get(thread, []), launch_ts) - 1
            if position >= 0 and scopes[position][0] <= launch_ts <= scopes[position][1]:
                token_count = scopes[position][2]
                add(by_tokens[str(token_count)], category, 1, duration, [duration], event.get("name"))
                attributed += 1
        if isinstance(base_ns, int):
            kernel_start_ns = base_ns + event.get("ts", 0) * 1000
            kernel_end_ns = kernel_start_ns + duration * 1000
            for workload, window in (workload_windows or {}).items():
                if window.get("started_unix_ns", math.inf) <= kernel_start_ns and kernel_end_ns <= window.get("finished_unix_ns", -math.inf):
                    add(by_workload[workload], category, 1, duration, [duration], event.get("name"))
                    break
    # Union within one rank is a diagnostic of profiled kernel activity. It is
    # still not an end-to-end critical path or a busy-percentage denominator.
    gpu_intervals.sort()
    merged = []
    for lo, hi in gpu_intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    gpu_union = sum(hi - lo for lo, hi in merged)
    return {"source": str(path), "source_kind": "raw_trace", "complete_kernel_coverage": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observed_cpu_events": cpu_events, "observed_cuda_kernel_events": kernel_events,
            "known_kernel_calls": kernel_events, "known_kernel_call_coverage": 1.0,
            "known_summed_gpu_duration_us": sum(v["total"] for v in categories.values()),
            "kernel_interval_union_us": gpu_union,
            "kernel_span_us": max(end for _, end in gpu_intervals) - gpu_intervals[0][0] if gpu_intervals else 0,
            "categories": finalize_groups(categories), "kernels": finalize_groups(by_kernel),
            "moe_kernel_attribution": {"method": "GPU correlation -> CUDA launch on CPU thread -> enclosing vllm::moe_forward scope",
                "attributed_kernel_calls": attributed,
                "by_source_local_input_tokens": {tokens: finalize_groups(groups) for tokens, groups in by_tokens.items()},
                "limitation": "Local input tokens do not control other ranks' tokens, received expert tokens, per-expert loads, stream overlap or synchronization."},
            "by_client_workload_time_window": {name: finalize_groups(groups) for name, groups in by_workload.items()},
            "cpu_moe_input_shape_histogram": [
                {"operator": n, "input_dims": json.loads(d), "calls": calls}
                for (n, d), calls in sorted(shape_counts.items())],
            "trace_time_metadata": {k: trace.get(k) for k in
                                    ("baseTimeNanoseconds", "displayTimeUnit", "schemaVersion")}}


def summarize_top_k(summary):
    by_kernel, categories = {}, {}
    for kernel in summary["top_kernels_by_summed_duration"]:
        name = kernel["name"]
        calls, total = kernel["calls"], kernel["summed_duration_us"]
        add(by_kernel, name, calls, total, kernel_name=name)
        add(categories, classify(name), calls, total, kernel_name=name)
    known = sum(v["calls"] for v in categories.values())
    total = summary.get("cuda_kernel_events", 0)
    return {"source": summary["path"], "source_kind": "top_kernel_summary",
            "complete_kernel_coverage": False,
            "observed_cpu_events": summary.get("cpu_events", 0),
            "observed_cuda_kernel_events": total, "known_kernel_calls": known,
            "known_kernel_call_coverage": known / total if total else 0,
            "known_summed_gpu_duration_us": sum(v["total"] for v in categories.values()),
            "kernel_interval_union_us": None, "kernel_span_us": None,
            "categories": finalize_groups(categories), "kernels": finalize_groups(by_kernel),
            "cpu_moe_input_shape_histogram": [],
            "coverage_note": "Top-40 kernel names only. Counts/durations for omitted kernels are unavailable; category sums are lower bounds and exact per-call quantiles cannot be recovered."}


def normalize_rank(rank):
    categories = rank["categories"]
    router_calls = sum(categories.get(name, {}).get("calls", 0)
                       for name in ("router_stock", "router_placement"))
    dispatch_calls = categories.get("dispatch_payload", {}).get("calls", 0)
    rank["normalization"] = {"router_calls": router_calls, "dispatch_payload_calls": dispatch_calls,
                             "router_dispatch_call_count_match": router_calls == dispatch_calls and router_calls > 0}
    for category in categories.values():
        category["summed_us_per_router_call"] = category["summed_gpu_duration_us"] / router_calls if router_calls else None
        category["fraction_of_known_summed_kernel_duration"] = (category["summed_gpu_duration_us"] /
            rank["known_summed_gpu_duration_us"] if rank["known_summed_gpu_duration_us"] else None)
    expected_router = "router_stock" if rank["arm"] == "stock" else "router_placement"
    rank["checks"] = {
        "has_cpu_and_cuda": rank["observed_cpu_events"] > 0 and rank["observed_cuda_kernel_events"] > 0,
        "has_expected_router": categories.get(expected_router, {}).get("calls", 0) > 0,
        "has_dispatch_and_combine_payload": all(categories.get(name, {}).get("calls", 0) > 0
            for name in ("dispatch_payload", "combine_payload")),
        "has_expert_gemm": categories.get("expert_gemm", {}).get("calls", 0) > 0,
        "router_and_dispatch_calls_match": router_calls == dispatch_calls and router_calls > 0,
    }
    return rank


def analyze_result(path: Path, roots: list[Path]):
    result = json.loads(path.read_text())
    ranks = []
    for summary in result.get("trace_summary", {}).get("traces", []):
        if summary.get("dp_rank") is None:
            continue
        raw = find_raw(summary["path"], [path.parent, *roots])
        rank = summarize_raw(raw, result.get("shapes", {})) if raw else summarize_top_k(summary)
        rank.update(dp_rank=summary["dp_rank"], arm=result["arm"])
        ranks.append(normalize_rank(rank))
    ranks.sort(key=lambda r: r["dp_rank"])
    category_spread = {}
    for name in CATEGORY_NOTES:
        values = [r["categories"][name]["mean_us_per_call"] for r in ranks if name in r["categories"]]
        totals = [r["categories"][name]["summed_gpu_duration_us"] for r in ranks if name in r["categories"]]
        if values:
            category_spread[name] = {
                "ranks_present": len(values), "min_rank_mean_us_per_call": min(values),
                "max_rank_mean_us_per_call": max(values),
                "max_over_min_rank_mean_us_per_call": max(values) / min(values) if min(values) else None,
                "median_rank_mean_us_per_call": statistics.median(values),
                "median_rank_summed_duration_us": statistics.median(totals)}
    workload = {name: {key: shape[key] for key in
                      ("requests", "concurrency", "prompt_tokens", "completion_tokens", "elapsed_seconds") if key in shape}
                for name, shape in result.get("shapes", {}).items()}
    return {"arm": result["arm"], "result_file": str(path), "source_marked_complete": result.get("complete"),
            "expected_dp_ranks": result.get("trace_summary", {}).get("expected_dp_ranks"),
            "observed_dp_ranks": [r["dp_rank"] for r in ranks],
            "all_rank_mechanism_checks_pass": bool(ranks) and [r["dp_rank"] for r in ranks] == list(range(result.get("trace_summary", {}).get("expected_dp_ranks", len(ranks)))) and all(all(r["checks"].values()) for r in ranks),
            "workloads": workload, "ranks": ranks, "rank_spread": category_spread}


def change(before, after):
    return (after / before - 1) * 100 if before else None


def compare(stock, modified):
    records, shape_evidence, matched_token_records = [], [], []
    for old in stock["ranks"]:
        new = next((r for r in modified["ranks"] if r["dp_rank"] == old["dp_rank"]), None)
        if new is None:
            continue
        old_shapes = [x for x in old["cpu_moe_input_shape_histogram"] if x["operator"] == "vllm::moe_forward"]
        new_shapes = [x for x in new["cpu_moe_input_shape_histogram"] if x["operator"] == "vllm::moe_forward"]
        if old_shapes and new_shapes:
            shape_evidence.append({"dp_rank": old["dp_rank"], "moe_input_shape_histograms_match": old_shapes == new_shapes,
                                   "stock_histogram": old_shapes, "modified_histogram": new_shapes})
        old_tokens = old.get("moe_kernel_attribution", {}).get("by_source_local_input_tokens", {})
        new_tokens = new.get("moe_kernel_attribution", {}).get("by_source_local_input_tokens", {})
        for tokens in sorted(set(old_tokens) & set(new_tokens), key=int):
            pairs = {category: (old_tokens[tokens][category], new_tokens[tokens][category])
                     for category in sorted(set(old_tokens[tokens]) & set(new_tokens[tokens]))}
            if "router_stock" in old_tokens[tokens] and "router_placement" in new_tokens[tokens]:
                pairs["router_implementation"] = (old_tokens[tokens]["router_stock"], new_tokens[tokens]["router_placement"])
            for category, (a, b) in pairs.items():
                qa, qb = a["per_call_distribution"], b["per_call_distribution"]
                matched_token_records.append({"dp_rank": old["dp_rank"], "source_local_input_tokens": int(tokens),
                    "category": category, "stock_calls": a["calls"], "modified_calls": b["calls"],
                    "stock_mean_us_per_call": a["mean_us_per_call"], "modified_mean_us_per_call": b["mean_us_per_call"],
                    "stock_per_call_distribution": qa, "modified_per_call_distribution": qb,
                    "mean_us_per_call_change_pct": change(a["mean_us_per_call"], b["mean_us_per_call"]),
                    "median_us_per_call_change_pct": change(qa["p50_us"], qb["p50_us"]) if qa and qb else None,
                    "p95_us_per_call_change_pct": change(qa["p95_us"], qb["p95_us"]) if qa and qb else None})
        common = set(old["categories"]) & set(new["categories"])
        for category in sorted(common):
            a, b = old["categories"][category], new["categories"][category]
            records.append({"dp_rank": old["dp_rank"], "category": category,
                "stock_calls": a["calls"], "modified_calls": b["calls"],
                "call_count_change_pct": change(a["calls"], b["calls"]),
                "summed_gpu_duration_change_pct": change(a["summed_gpu_duration_us"], b["summed_gpu_duration_us"]),
                "mean_us_per_call_change_pct": change(a["mean_us_per_call"], b["mean_us_per_call"]),
                "summed_us_per_router_call_change_pct": change(a["summed_us_per_router_call"], b["summed_us_per_router_call"])})
        a = old["categories"].get("router_stock")
        b = new["categories"].get("router_placement")
        if a and b:
            records.append({"dp_rank": old["dp_rank"], "category": "router_implementation",
                "stock_calls": a["calls"], "modified_calls": b["calls"],
                "call_count_change_pct": change(a["calls"], b["calls"]),
                "summed_gpu_duration_change_pct": change(a["summed_gpu_duration_us"], b["summed_gpu_duration_us"]),
                "mean_us_per_call_change_pct": change(a["mean_us_per_call"], b["mean_us_per_call"]),
                "summed_us_per_router_call_change_pct": change(a["summed_us_per_router_call"], b["summed_us_per_router_call"])})
    return {"stock_arm": stock["arm"], "modified_arm": modified["arm"],
            "request_token_budgets_match": stock["workloads"].keys() == modified["workloads"].keys() and all(
                all(stock["workloads"][s].get(k) == modified["workloads"][s].get(k)
                    for k in ("requests", "concurrency", "prompt_tokens", "completion_tokens")) for s in stock["workloads"]),
            "token_shapes_controlled": False,
            "warning": "Equal HTTP token budgets do not imply equal engine iterations, per-rank token counts, grouped-expert sizes or overlap. Changes below are diagnostics of instrumented traces, not serving speedups or pure communication savings.",
            "input_shape_comparison_evidence": shape_evidence,
            "matched_source_local_input_token_comparisons": matched_token_records,
            "per_rank_categories": records}


def table(analysis):
    lines = ["GPU duration sums below can overlap. Profile timing is instrumented; this table does not measure serving speedup.", ""]
    key_categories = ["dispatch_notify_wait", "dispatch_payload", "combine_notify_wait", "combine_payload",
                      "expert_gemm", "router_stock", "router_placement", "expert_pack_sort_align",
                      "dispatch_layout_pack", "dense_gemm_candidate", "attention"]
    for profile in analysis["profiles"]:
        lines += [f"{profile['arm']}:", "", "| Category | Rank 0 ms | Rank 1 ms | Rank 2 ms | Rank 3 ms |",
                  "|---|---:|---:|---:|---:|"]
        for name in key_categories:
            if any(name in r["categories"] for r in profile["ranks"]):
                cells = []
                for rank in range(4):
                    item = next((r for r in profile["ranks"] if r["dp_rank"] == rank), None)
                    value = item["categories"].get(name) if item else None
                    cells.append(f"{value['summed_gpu_duration_us'] / 1000:.2f}" if value else "—")
                lines.append("| " + name + " | " + " | ".join(cells) + " |")
        lines.append("")
        for rank in profile["ranks"]:
            lines.append(f"Rank {rank['dp_rank']}: {rank['source_kind']}; covered "
                         f"{rank['known_kernel_call_coverage'] * 100:.1f}% of kernel calls; "
                         f"{rank['normalization']['router_calls']} router calls.")
        lines.append("")
    for comparison in analysis.get("comparisons", []):
        matched = comparison["matched_source_local_input_token_comparisons"]
        if not matched:
            continue
        lines += [f"Exact local input-token comparisons: stock → {comparison['modified_arm']}", "",
                  "Only raw GPU events with a launch correlated to an enclosing CPU MoE scope are included. Equal local T still leaves other ranks' tokens, received expert loads, and synchronization uncontrolled.", "",
                  "| DP | Local T | Category | Calls stock / modified | Median µs stock / modified | p95 µs stock / modified |",
                  "|---:|---:|---|---:|---:|---:|"]
        wanted = {"router_implementation", "dispatch_payload", "dispatch_notify_wait",
                  "combine_payload", "combine_notify_wait", "expert_gemm"}
        for item in matched:
            if item["category"] not in wanted:
                continue
            before, after = item["stock_per_call_distribution"], item["modified_per_call_distribution"]
            lines.append(f"| {item['dp_rank']} | {item['source_local_input_tokens']} | {item['category']} | "
                         f"{item['stock_calls']} / {item['modified_calls']} | "
                         f"{before['p50_us']:.3f} / {after['p50_us']:.3f} | "
                         f"{before['p95_us']:.3f} / {after['p95_us']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def self_test():
    assert classify("fused_moe_kernel") == "expert_gemm"
    assert classify("void cutlass::device_kernel<flash::FlashAttnFwdCombine<...>") == "attention"
    assert classify("void deep_ep::legacy::intranode::cached_notify_combine<4>(...)" ) == "combine_notify_wait"
    assert classify("void deep_ep::legacy::intranode::notify_dispatch<4>(...)" ) == "dispatch_notify_wait"
    assert classify("void deep_ep::legacy::layout::get_dispatch_layout<4>(...)" ) == "dispatch_layout_pack"
    assert classify("void deep_ep::legacy::intranode::dispatch<4>(...)" ) == "dispatch_payload"
    assert classify("void deep_ep::legacy::intranode::combine<4>(...)" ) == "combine_payload"
    assert classify("_combine_sampled_and_draft_tokens_kernel") == "sampling"
    assert classify("_par_select") == "router_placement"
    assert classify("void vllm::moe::topkGating<8>(...)") == "router_stock"
    assert classify("nvjet_sm90_tst_320x128") == "dense_gemm_candidate"
    assert classify("moe_align_block_size_kernel") == "expert_pack_sort_align"
    assert distribution([1, 2, 3])["p50_us"] == 2
    # Exercise asynchronous correlation: the GPU kernel begins after the CPU
    # MoE scope ends, so timestamp containment on GPU time would be incorrect.
    import tempfile
    synthetic = {"baseTimeNanoseconds": 1000000, "traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "vllm::moe_forward", "pid": 10, "tid": 11,
         "ts": 100, "dur": 100, "args": {"Input Dims": [[16, 2048]]}},
        {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx", "pid": 10, "tid": 11,
         "ts": 120, "dur": 2, "args": {"correlation": 42}},
        {"ph": "X", "cat": "kernel", "name": "fused_moe_kernel", "pid": 0, "tid": 1,
         "ts": 250, "dur": 10, "args": {"correlation": 42}}]}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        path.write_text(json.dumps(synthetic))
        summary = summarize_raw(path, {"decode": {"started_unix_ns": 1200000, "finished_unix_ns": 1300000}})
        assert summary["moe_kernel_attribution"]["by_source_local_input_tokens"]["16"]["expert_gemm"]["calls"] == 1
        assert summary["by_client_workload_time_window"]["decode"]["expert_gemm"]["calls"] == 1
    fallback = summarize_top_k({"path": "trace", "cpu_events": 1, "cuda_kernel_events": 2,
        "top_kernels_by_summed_duration": [{"name": "fused_moe_kernel", "calls": 1, "summed_duration_us": 5}]})
    assert fallback["categories"]["expert_gemm"]["mean_us_per_call"] == 5
    assert fallback["known_kernel_call_coverage"] == .5
    print("17 classification/correlation/statistics checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="*")
    parser.add_argument("--trace-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if not args.results:
            return
    if not args.results:
        parser.error("At least one profile result.json is required")
    profiles = [analyze_result(p.resolve(), args.trace_root) for p in args.results]
    baseline = next((p for p in profiles if p["arm"] == "stock"), None)
    analysis = {"category_definitions": CATEGORY_NOTES, "profiles": profiles,
                "comparisons": [compare(baseline, p) for p in profiles if p is not baseline] if baseline else [],
                "interpretation_limits": [
                    "Kernel event durations are summed; concurrent streams and GPUs overlap. Sums are not critical-path time, MoE layer time, or serving latency.",
                    "Notification kernels include waits and bookkeeping; payload kernels also include coordination. Names do not decompose pure communication bytes/bandwidth.",
                    "Profiles combine short decode and long prefill workloads. Different scheduling/batch shapes and expert token distributions confound raw sums and per-call means.",
                    "Generic nvjet/cuBLAS kernels are candidates for dense GEMM; call stacks/correlation are needed for definitive operator attribution.",
                    "When only top-kernel summaries are present, omitted kernel durations remain unknown. No category total should be presented as exhaustive.",
                    "Use repeated uninstrumented HTTP measurements for performance claims and per-shape, fixed-token engine measurements for causal mechanism comparisons."]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    rendered = table(analysis)
    if args.markdown:
        args.markdown.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
