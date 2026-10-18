#!/usr/bin/env python3
"""Offline audit of separate stock/original-balanced paper diagnostic profiles.

  python experiments/analyze_paper_profiles.py EP4/profile_*/result.json \
    EP8/profile_*/result.json --trace-root COPIED_RUN_ROOT \
    --phase-analysis EP4_TIMING/paper_analysis.json \
    --phase-analysis EP8_TIMING/paper_analysis.json --output NEW_LOCAL_AUDIT.json

Reads each complete raw trace, verifies recorded hashes, and writes JSON plus
Markdown. No model, GPU, cluster operations or benchmark-source changes. Typed
routing-ID casts require an int32->int64 [T,8] copy CPU scope inside MoE, then
an External-id or CUDA-launch correlation to a GPU kernel. Kernel names alone
never establish a cast. Exact local T comes from the enclosing CPU MoE scope,
not from a nearby GPU timestamp or the requested HTTP token budget.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

from analyze_profiles import add, classify, distribution, finalize_groups
import paper_reproduce as paper

ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = ("stock", "balanced6p1")
GRAPH_API = re.compile(r"^(?:cuda|cu)GraphLaunch(?:_|$)")
LIMITS = [
    "Profiles are separate instrumented launches. Their HTTP latency and CPU durations include profiler overhead and cannot replace the CUDA-event matrix.",
    "Summed GPU durations overlap across streams and ranks. They are not critical-path latency, clean speedups or Amdahl bounds; ranks are not independent experiment replicates.",
    "Matching local hidden T does not match other ranks' T, received expert-token counts, per-expert loads, stream overlap or rank-arrival waits.",
    "A CPU MoE-op shape count does not prove unique layer names or complete 48-layer steps. The independent CUDA-event probe establishes that coverage.",
    "Dispatch/combine payload kernels can contain copies, reductions and waits. Notify kernels contain coordination and waits; neither reports wire bytes or pure network transfer time.",
    "aten::_to_copy includes allocation/copy_ work. Parent and child CPU durations overlap and must not be added. Typed-copy GPU correlation is reported independently of generic cast-like kernel names.",
    "One stock and one original-balanced profiler launch per EP cell provide mechanism observations, not independent performance replicates or causal attribution.",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def duration_summary(values):
    return {"calls": len(values), "summed_duration_us": sum(values),
            "distribution_us": distribution(values)}


def thread(event):
    return event.get("pid"), event.get("tid")


def integer_type(value):
    value = str(value).lower().replace("torch.", "").replace("c10::", "").strip()
    if value in ("int", "int32", "int32_t", "int32type"):
        return 32
    if value in ("long", "long int", "int64", "int64_t", "longtype"):
        return 64
    return None


def dimensions(event):
    args = event.get("args", {})
    return args.get("Input Dims") or args.get("Input dims") or []


def routing_T(event):
    dims = dimensions(event)
    if dims and isinstance(dims[0], list) and len(dims[0]) == 2 and dims[0][1] == 8:
        return dims[0][0]
    return None


def hidden_T(event):
    dims = dimensions(event)
    if dims and isinstance(dims[0], list) and len(dims[0]) == 2 and dims[0][1] == 2048:
        return dims[0][0]
    return None


class ScopeIndex:
    """Same-thread containment; no GPU/CPU timestamp-proximity inference."""
    def __init__(self, events):
        self.events = collections.defaultdict(list)
        for event in events:
            self.events[thread(event)].append(event)
        for values in self.events.values():
            values.sort(key=lambda event: event["ts"])
        self.starts = {key: [event["ts"] for event in values] for key, values in self.events.items()}

    def enclosing(self, event, full=True):
        values = self.events.get(thread(event), [])
        index = bisect.bisect_right(self.starts.get(thread(event), []), event.get("ts", -math.inf)) - 1
        if index < 0:
            return None
        scope = values[index]
        end = event["ts"] + event.get("dur", 0) if full else event["ts"]
        return scope if scope["ts"] <= event["ts"] and end <= scope["ts"] + scope["dur"] + .01 else None


def summarize_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        trace = json.load(source)
    cpu, kernels, scopes = [], [], []
    by_external, apis = collections.defaultdict(list), collections.defaultdict(list)
    graph_names = collections.Counter()
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        cat, name = str(event.get("cat", "")), event.get("name", "")
        if cat in ("cpu_op", "python_function", "user_annotation"):
            cpu.append(event)
            ext = event.get("args", {}).get("External id")
            if ext is not None:
                by_external[ext].append(event)
            if cat == "cpu_op" and name == "vllm::moe_forward":
                scopes.append(event)
        if "kernel" in cat.lower():
            kernels.append(event)
        if cat in ("cuda_runtime", "cuda_driver"):
            args = event.get("args", {})
            if args.get("correlation") is not None:
                apis[args["correlation"]].append(event)
            if GRAPH_API.search(name):
                graph_names[name] += 1
    scope_index = ScopeIndex(scopes)
    scope_ids = {id(event) for event in scopes}
    copies, parents, shape_mismatches = [], [], []
    for event in cpu:
        name, args, T = event.get("name"), event.get("args", {}), routing_T(event)
        if T is None or name not in ("aten::copy_", "aten::_to_copy"):
            continue
        scope = scope_index.enclosing(event)
        if scope is None:
            continue
        types = args.get("Input type", [])
        if name == "aten::copy_" and len(types) >= 2 and [integer_type(t) for t in types[:2]] == [64, 32]:
            copies.append(event)
            if hidden_T(scope) != T:
                shape_mismatches.append({"copy_T": T, "enclosing_moe_T": hidden_T(scope), "ts": event["ts"]})
        if name == "aten::_to_copy" and types and integer_type(types[0]) == 32:
            parents.append(event)
    copy_index = ScopeIndex(copies)
    copy_external = collections.defaultdict(list)
    for event in copies:
        ext = event.get("args", {}).get("External id")
        if ext is not None:
            copy_external[ext].append(event)

    def cast_parent(kernel):
        args = kernel.get("args", {})
        candidates = copy_external.get(args.get("External id"), [])
        if len(candidates) == 1:
            return candidates[0], "typed_copy_external_id"
        for api in apis.get(args.get("correlation"), []):
            copy = copy_index.enclosing(api)
            if copy is not None:
                return copy, "launch_inside_typed_copy"
        return None, None

    def moe_parent(kernel):
        args = kernel.get("args", {})
        candidates = by_external.get(args.get("External id"), [])
        if len(candidates) == 1:
            event = candidates[0]
            scope = event if id(event) in scope_ids else scope_index.enclosing(event)
            if scope is not None:
                return scope, "cpu_external_id_to_moe_scope"
        for api in apis.get(args.get("correlation"), []):
            scope = scope_index.enclosing(api)
            if scope is not None:
                return scope, "cuda_launch_to_moe_scope"
        return None, None

    categories, by_T, by_kernel = {}, collections.defaultdict(dict), {}
    cast_gpu, cast_methods, attributed_methods = [], collections.Counter(), collections.Counter()
    cast_by_T = collections.defaultdict(lambda: {"cpu_copy": [], "cpu_to_copy_parent": [], "gpu_cast": []})
    copy_gpu_counts, intervals = collections.Counter(), []
    router_names, deep_ep_geometry = collections.Counter(), collections.Counter()
    for event in copies:
        cast_by_T[str(routing_T(event))]["cpu_copy"].append(event["dur"])
    for event in parents:
        cast_by_T[str(routing_T(event))]["cpu_to_copy_parent"].append(event["dur"])
    for kernel in kernels:
        name, duration = kernel.get("name", ""), kernel.get("dur", 0.)
        if not isinstance(duration, (int, float)) or duration < 0 or not math.isfinite(duration):
            raise ValueError("Invalid GPU kernel duration")
        copy, cast_method = cast_parent(kernel)
        category = "routing_id_cast" if copy is not None else classify(name)
        if category in ("dispatch_notify_wait", "dispatch_payload", "combine_notify_wait", "combine_payload", "dispatch_layout_pack"):
            geometry = kernel.get("args", {})
            deep_ep_geometry[category, name, json.dumps(geometry.get("grid")), json.dumps(geometry.get("block"))] += 1
        if copy is not None:
            cast_gpu.append(kernel)
            cast_methods[cast_method] += 1
            copy_gpu_counts[id(copy)] += 1
            cast_by_T[str(routing_T(copy))]["gpu_cast"].append(duration)
        scope, method = moe_parent(kernel)
        if scope is not None:
            attributed_methods[method] += 1
            T = hidden_T(scope)
            if T is not None:
                add(by_T[str(T)], category, 1, duration, [duration], name)
        else:
            attributed_methods["unattributed"] += 1
        add(categories, category, 1, duration, [duration], name)
        add(by_kernel, name, 1, duration, [duration], name)
        if "_par_select" in name or classify(name) == "router_stock":
            router_names[name] += 1
        if isinstance(kernel.get("ts"), (int, float)):
            intervals.append((kernel["ts"], kernel["ts"] + duration))
    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    Ts = collections.Counter(str(hidden_T(scope)) for scope in scopes if hidden_T(scope) is not None)
    buckets = collections.Counter()
    for T, count in Ts.items():
        numeric = int(T)
        buckets[str(1 << max(0, numeric.bit_length() - 1))] += count
    cast_names = collections.Counter(kernel.get("name") for kernel in cast_gpu)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "cpu_events": len(cpu), "cuda_kernel_events": len(kernels), "moe_cpu_calls": len(scopes),
        "moe_actual_hidden_T_counts": dict(Ts), "moe_power_of_two_bucket_counts": dict(buckets),
        "moe_calls_without_hidden_T": len(scopes) - sum(Ts.values()),
        "router_kernel_counts_by_exact_name": dict(router_names),
        "original_par_select_gpu_calls": router_names.get("_par_select", 0),
        "unexpected_placement_kernel_calls": sum(count for name, count in router_names.items()
            if "_par_select" in name and name != "_par_select"),
        "cuda_graph_launch_api_counts": dict(graph_names),
        "deep_ep_launch_geometry_counts": [{"category": category, "name": name, "grid": json.loads(grid),
            "block": json.loads(block), "calls": count}
            for (category, name, grid, block), count in sorted(deep_ep_geometry.items())],
        "categories": finalize_groups(categories),
        "kernels": finalize_groups(by_kernel),
        "by_exact_local_hidden_T": {T: finalize_groups(groups) for T, groups in by_T.items()},
        "gpu_to_moe_attribution_counts": dict(attributed_methods),
        "gpu_interval_union_us": sum(hi - lo for lo, hi in merged),
        "gpu_kernel_span_us": max(end for _, end in intervals) - intervals[0][0] if intervals else None,
        "summed_gpu_kernel_duration_us": sum(kernel.get("dur", 0.) for kernel in kernels),
        "routing_id_cast": {"typed_int32_to_int64_T8_cpu_copy": duration_summary([e["dur"] for e in copies]),
            "int32_T8_to_copy_cpu_parent_candidate": duration_summary([e["dur"] for e in parents]),
            "correlated_gpu_cast": duration_summary([e["dur"] for e in cast_gpu]),
            "gpu_cast_kernel_names": dict(cast_names), "gpu_correlation_methods": dict(cast_methods),
            "typed_copy_scopes_with_gpu_kernel": len(copy_gpu_counts),
            "typed_copy_scopes_without_gpu_kernel": len(copies) - len(copy_gpu_counts),
            "cpu_copy_calls_per_moe_forward": len(copies) / len(scopes) if scopes else None,
            "gpu_cast_kernels_per_moe_forward": len(cast_gpu) / len(scopes) if scopes else None,
            "copy_T_vs_enclosing_moe_T_mismatches": shape_mismatches,
            "by_actual_source_T": {T: {kind: duration_summary(values) for kind, values in groups.items()}
                                   for T, groups in cast_by_T.items()},
            "proof": "Typed CPU copy [T,8] destinationint64/sourceint32 inside same-thread MoE; GPU External-id or launch containment. Parent_to_copy is only a candidate because its output type is not recorded."},
        "trace_time_metadata": {key: trace.get(key) for key in ("baseTimeNanoseconds", "schemaVersion", "displayTimeUnit")}}


def locate_trace(record, roots):
    """Resolve copied paths by identity; ambiguous differing files never win by order."""
    expected = record.get("sha256")
    if not expected:
        raise ValueError("Raw trace has no recorded SHA256; summary-only evidence is insufficient")
    original = Path(record["path"])
    candidates = [original] if original.is_file() else []
    for root in roots:
        if root.is_file() and root.name == original.name:
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(root.rglob(original.name))
    candidates = sorted({path.resolve() for path in candidates})
    matches = [path for path in candidates if path.stat().st_size == record.get("bytes") and sha256(path) == expected]
    if not matches:
        raise FileNotFoundError("No SHA/size-matching raw trace for " + record["path"])
    return matches[0]


def validate_rank(row, arm):
    errors = []
    if not row["cpu_events"] or not row["cuda_kernel_events"]:
        errors.append("Missing CPU+CUDA coverage")
    if row["moe_actual_hidden_T_counts"].get("16384", 0) < 48:
        errors.append("Fewer than48 positive actual-T16384 MoE CPU scopes")
    if row["categories"].get("expert_gemm", {}).get("calls", 0) < 48:
        errors.append("Missing positive expert-GEMM coverage")
    if row["cuda_graph_launch_api_counts"]:
        errors.append("Unexpected graph launch in HT/NONE profile")
    if row["unexpected_placement_kernel_calls"]:
        errors.append("Experimental placement kernel present")
    stock_calls = row["categories"].get("router_stock", {}).get("calls", 0)
    original_calls = row["original_par_select_gpu_calls"]
    if arm == "stock" and (stock_calls < 48 or original_calls):
        errors.append("Native stock router identity not proven")
    if arm == "balanced6p1" and (original_calls < 48 or stock_calls):
        errors.append("Original balanced selector identity not proven")
    if row["routing_id_cast"]["copy_T_vs_enclosing_moe_T_mismatches"]:
        errors.append("Typed routing-copy T differs from enclosing MoE hidden T")
    return errors


def inspect_result(path, trace_roots):
    result = read(path)
    if result.get("complete") is not True or result.get("measurement_type") != "profile_only":
        raise ValueError("Expected a complete, separate profiler-only result: " + str(path))
    arm = result.get("arm")
    if arm not in ARM_NAMES:
        raise ValueError("Unexpected profile arm: " + str(arm))
    manifest_path = path.parent.parent / "manifest.json"
    manifest = read(manifest_path)
    if result.get("identity_sha256") != sha256(manifest_path):
        raise ValueError("Profile manifest SHA differs from the recorded identity")
    identity = manifest["identity"]
    paper.require_original_source(identity)
    config = identity["configuration"]
    if not (config["dp"] in (4, 8) and config["ep"] == config["dp"] and config["tp"] == 1
            and config["graphs"] == "NONE" and config["backend"] == "deepep_high_throughput"
            and config["dtype"] == "bfloat16" and config.get("num_sms_source_default") == 20):
        raise ValueError("Profile engine configuration differs from the declared paper protocol")
    checkpoint = identity["checkpoint"]
    proof = checkpoint["proof"]
    if proof.get("model_id") != paper.MODEL_ID or proof.get("canonical_identity_verified") is not True:
        raise ValueError("Canonical Instruct-2507 checkpoint identity not verified")
    server_path = path.parent / "server.json"
    server = read(server_path)
    environment = server.get("controlled_environment", {})
    if (server.get("cuda_event_probe_active") is not False
            or environment.get("PLACEMENT_PROBE") is not None
            or environment.get("PLACEMENT_EXPERIMENTAL_SELECTOR") is not None
            or server.get("paper_probe_environment") != {}):
        raise ValueError("Profile overlaps a CUDA-event probe or experimental selector")
    if environment.get("PLACEMENT_KEEP") != ("6" if arm == "balanced6p1" else None):
        raise ValueError("Profile keep environment does not match its arm")
    if environment.get("PLACEMENT_LOAD_PENALTY") != ("1" if arm == "balanced6p1" else None):
        raise ValueError("Profile load-penalty environment does not match its arm")
    if (environment.get("OMP_NUM_THREADS") != "1" or environment.get("NVSHMEM_QP_DEPTH") != "4096"
            or environment.get("VLLM_PLUGINS") != "placement_aware_router"
            or server.get("source_sha256") != identity["source_sha256"]):
        raise ValueError("Profile server controls/source identity differ from its manifest")
    command = list(server["command"])
    if "--profiler-config" not in command:
        raise ValueError("Profile server command has no profiler config")
    position = command.index("--profiler-config")
    profiler_config = json.loads(command[position + 1])
    del command[position:position + 2]
    if (command != paper.server_command(SimpleNamespace(**config))
            or profiler_config.get("profiler") != "torch"
            or profiler_config.get("torch_profiler_record_shapes") is not True):
        raise ValueError("Profile server command differs from exact paper/profiler configuration")
    log_path = path.parent / "server.log"
    log = log_path.read_text(errors="replace")
    if re.search(r"\[paper-probe\]|\[moet\]|experimental selector=", log):
        raise ValueError("Profile logs show a timing probe or experimental selector")
    runtime_log_evidence = paper.check_log(SimpleNamespace(dp=config["dp"]),
        "base" if arm == "stock" else arm, log_path)
    warmups, profiled = result.get("warmups", []), result.get("profile_workload") or {}
    if len(warmups) != 3:
        raise ValueError("Profile did not preserve three full warmup rounds")
    for row in [*warmups, profiled]:
        count = 128 if row is profiled else 256
        if (row.get("requests") != count or row.get("concurrency") != 128
                or row.get("prompt_tokens") != 4096 * count or row.get("completion_tokens") != count):
            raise ValueError("Profile HTTP shape/token totals differ from fixed warmup/profile protocol")
    rows, auxiliary, errors, seen = [], [], [], set()
    for record in result["trace_summary"]["traces"]:
        rank = record.get("dp_rank")
        raw = locate_trace(record, [path.parent, *trace_roots])
        if rank is None:
            extra = summarize_trace(raw)
            auxiliary.append({key: extra[key] for key in ("path", "sha256", "bytes", "cpu_events",
                "cuda_kernel_events", "moe_cpu_calls", "cuda_graph_launch_api_counts")})
            if extra["moe_cpu_calls"]:
                errors.append("Unranked auxiliary trace contains model MoE scopes")
            continue
        if rank not in range(config["dp"]) or rank in seen:
            raise ValueError("Unexpected/duplicate DP rank in raw profile trace list")
        filename_rank = re.search(r"(?:^|_)dp(\d+)(?:_|\.)", raw.name)
        if not filename_rank or int(filename_rank.group(1)) != rank:
            raise ValueError("Recorded DP rank differs from raw trace filename identity")
        seen.add(rank)
        row = summarize_trace(raw)
        row.update(dp_rank=rank, validation_errors=validate_rank(row, arm))
        errors += [f"DP{rank}: {error}" for error in row["validation_errors"]]
        rows.append(row)
        print(f"[audit] EP{config['dp']} {arm} DP{rank}: MoE={row['moe_cpu_calls']}, "
              f"original_router={row['original_par_select_gpu_calls']}, "
              f"typed_cast={row['routing_id_cast']['typed_int32_to_int64_T8_cpu_copy']['calls']}, "
              f"GPU_cast={row['routing_id_cast']['correlated_gpu_cast']['calls']}", flush=True)
    if seen != set(range(config["dp"])):
        errors.append("Missing physical DP ranks")
    return {"result_path": str(path), "result_sha256": sha256(path), "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "server_sha256": sha256(server_path),
        "server_log_sha256": sha256(log_path), "runtime_log_evidence": runtime_log_evidence,
        "arm": arm, "ep": config["dp"], "configuration": config,
        "canonical_checkpoint": {"model_id": proof["model_id"], "revision": proof.get("revision"),
            "proof_sha256": checkpoint.get("proof_sha256"),
            "historical_paper_checkpoint_verified": checkpoint.get("paper_checkpoint_identity_verified") is True},
        "production_source_sha256": {key: identity["source_sha256"][key] for key in paper.ORIGINAL_SOURCE},
        "profile_helper_source_sha256": identity["source_sha256"].get("experiments/paper_profile.py"),
        "warmup_requests": [row.get("requests") for row in result.get("warmups", [])],
        "jit_warnings_preserved": {"warmups": [row.get("jit_events_observed_in_log_slice", []) for row in warmups],
            "profile_workload": profiled.get("jit_events_observed_in_log_slice", [])},
        "profile_workload": {key: result.get("profile_workload", {}).get(key) for key in
            ("configuration", "requests", "concurrency", "started_unix_ns", "finished_unix_ns", "elapsed_seconds")},
        "valid": not errors, "validation_errors": errors, "auxiliary_non_DP_traces": auxiliary,
        "ranks": sorted(rows, key=lambda row: row["dp_rank"])}


def matched_local_T_comparisons(profiles):
    indexed, output = {}, []
    for profile in profiles:
        key = profile["ep"], profile["arm"]
        if key in indexed:
            raise ValueError("Multiple profiler launches for the same EP/arm are not independent replicates")
        indexed[key] = profile
    for ep in sorted({key[0] for key in indexed}):
        if (ep, "stock") not in indexed or (ep, "balanced6p1") not in indexed:
            continue
        stock, balanced = indexed[ep, "stock"], indexed[ep, "balanced6p1"]
        if (stock["canonical_checkpoint"] != balanced["canonical_checkpoint"]
                or stock["configuration"] != balanced["configuration"]
                or stock["production_source_sha256"] != balanced["production_source_sha256"]):
            raise ValueError("Stock and balanced profiles have different checkpoint/config/source identities")
        by_rank = {row["dp_rank"]: row for row in balanced["ranks"]}
        for base in stock["ranks"]:
            other = by_rank.get(base["dp_rank"])
            if other is None:
                continue
            for T in sorted(set(base["by_exact_local_hidden_T"]) & set(other["by_exact_local_hidden_T"]), key=int):
                lhs, rhs = base["by_exact_local_hidden_T"][T], other["by_exact_local_hidden_T"][T]
                categories = set(lhs) | set(rhs)
                categories -= {"router_stock", "router_placement"}
                for category in sorted(categories | {"router"}):
                    left = lhs.get("router_stock" if category == "router" else category)
                    right = rhs.get("router_placement" if category == "router" else category)
                    if not left or not right:
                        continue
                    p50 = left["per_call_distribution"]["p50_us"]
                    output.append({"ep": ep, "dp_rank": base["dp_rank"], "actual_local_T": int(T),
                        "category": category, "stock": left, "balanced6p1": right,
                        "balanced_vs_stock_p50_duration_change_pct":
                            100 * (right["per_call_distribution"]["p50_us"] / p50 - 1) if p50 else None,
                        "scope": "One instrumented launch per arm, matched local-T only; not clean latency or causal speedup"})
    return output


def phase_findings(paths, profiles):
    output = []
    for path in paths:
        data = read(path)
        scope = data["scope"]
        ep = scope["config"]["dp"]
        checkpoint = scope["checkpoint"]
        matching = [row for row in profiles if row["ep"] == ep]
        same = bool(matching) and all(row["canonical_checkpoint"]["proof_sha256"] == checkpoint.get("proof_sha256")
            and row["production_source_sha256"] == {key: scope["source_sha256"].get(key) for key in paper.ORIGINAL_SOURCE}
            for row in matching)
        output.append({"path": str(path), "sha256": sha256(path), "ep": ep,
            "matrix_complete": data.get("complete") is True, "counts": data.get("counts"),
            "same_checkpoint_proof_and_production_source": same,
            "balanced6p1_phase_comparisons": [row for row in data.get("comparisons", []) if row.get("arm") == "balanced6p1"],
            "noise_floors": data.get("noise_floors"),
            "limits": "Independent CUDA-event matrix summaries retained unchanged; profiler timings are never pooled or used to revise/filter the matrix. Compare mechanisms only when checkpoint/source identity agrees."})
    return output


def markdown(data):
    lines = ["Paper profile mechanism audit", "", f"All supplied profiles valid: {data['valid']}", "",
        "Profiles contain instrumentation overhead. GPU sums and per-call ratios below are diagnostic; they are not clean-serving or critical-path speedups.", "",
        "| EP | Arm | Rank | MoE scopes | Actual T=16384 scopes | Original _par_select | Typed ID copies | Correlated cast kernels | Copy/MoE | GPU sum ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for profile in data["profiles"]:
        for row in profile["ranks"]:
            cast = row["routing_id_cast"]
            fraction = cast["cpu_copy_calls_per_moe_forward"]
            fraction_text = f"{fraction:.3f}" if fraction is not None else "unavailable"
            lines.append(f"|{profile['ep']}|{profile['arm']}|{row['dp_rank']}|{row['moe_cpu_calls']}|"
                f"{row['moe_actual_hidden_T_counts'].get('16384', 0)}|{row['original_par_select_gpu_calls']}|"
                f"{cast['typed_int32_to_int64_T8_cpu_copy']['calls']}|{cast['correlated_gpu_cast']['calls']}|"
                f"{fraction_text}|{row['summed_gpu_kernel_duration_us']/1000:.3f}|")
    lines += ["", "Typed copies require int32→int64 [T,8] inside a CPU MoE scope; cast GPU kernels require correlation to those copies. One copy per MoE forward corresponds to 48 copies for a 48-layer forward, subject to the independent complete-step coverage proof.", "",
        "| EP | Arm | Rank | Actual hidden-T counts | Power-of-two bucket counts |",
        "|---:|---|---:|---|---|"]
    for profile in data["profiles"]:
        for row in profile["ranks"]:
            lines.append(f"|{profile['ep']}|{profile['arm']}|{row['dp_rank']}|{row['moe_actual_hidden_T_counts']}|{row['moe_power_of_two_bucket_counts']}|")
    lines += ["", "Exact local-T=16384 phase kernels (GPU per-call medians; counts and other T strata are retained in JSON):", "",
        "| EP | Rank | Category | Stock calls | Balanced calls | Stock p50 µs | Balanced p50 µs |",
        "|---:|---:|---|---:|---:|---:|---:|"]
    for row in data["matched_local_T_comparisons"]:
        if row["actual_local_T"] != 16384 or row["category"] not in (
                "router", "dispatch_notify_wait", "dispatch_payload", "combine_notify_wait", "combine_payload", "expert_gemm"):
            continue
        lhs, rhs = row["stock"], row["balanced6p1"]
        lines.append(f"|{row['ep']}|{row['dp_rank']}|{row['category']}|{lhs['calls']}|{rhs['calls']}|"
                     f"{lhs['per_call_distribution']['p50_us']:.3f}|{rhs['per_call_distribution']['p50_us']:.3f}|")
    for finding in data["independent_cuda_event_findings"]:
        lines += ["", f"Independent EP{finding['ep']} CUDA-event matrix: complete={finding['matrix_complete']}, same checkpoint/source={finding['same_checkpoint_proof_and_production_source']}.", ""]
        for row in finding["balanced6p1_phase_comparisons"]:
            values = row["paper_arithmetic_speedup_pct"]
            lines.append(f"- {row['estimator']}, {row['phase']}: mean gain {values['mean']:+.3f}%, n={values['n']}, CI95={values['ci95']}; own floor={row.get('floor_pct')}%." if values.get("mean") is not None else
                         f"- {row['estimator']}, {row['phase']}: unavailable.")
    errors = [f"EP{p['ep']} {p['arm']}: {error}" for p in data["profiles"] for error in p["validation_errors"]]
    if errors:
        lines += ["", "Validation failures:", ""] + ["- " + error for error in errors]
    lines += ["", "Interpretation limits:", ""] + ["- " + value for value in LIMITS]
    lines += ["", "Raw/result/manifest/source hashes and full per-rank categories are retained in the JSON audit."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, nargs="+")
    p.add_argument("--trace-root", type=Path, action="append", default=[])
    p.add_argument("--phase-analysis", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.suffix != ".json":
        p.error("--output must be a new local .json analysis file")
    if args.output.exists() and read(args.output).get("analysis_kind") != "offline_original_paper_profile_mechanisms":
        p.error("Refusing to replace existing source evidence with an analysis")
    if args.output.with_suffix(".md").exists() and not args.output.exists():
        p.error("Refusing to replace an existing unrelated Markdown artifact")
    outputs = {args.output.resolve(), args.output.with_suffix(".md").resolve()}
    inputs = {path.resolve() for path in [*args.results, *args.phase_analysis]}
    if outputs & inputs:
        p.error("Audit output must be a new analysis path, separate from source evidence")
    profiles = [inspect_result(path, args.trace_root) for path in args.results]
    data = {"schema_version": 1, "analysis_kind": "offline_original_paper_profile_mechanisms",
        "created_unix": time.time(), "valid": all(row["valid"] for row in profiles),
        "analysis_source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(__file__),
            "experiments/analyze_profiles.py": sha256(ROOT / "experiments/analyze_profiles.py"),
            "experiments/paper_reproduce.py": sha256(ROOT / "experiments/paper_reproduce.py")},
        "profiles": profiles, "matched_local_T_comparisons": matched_local_T_comparisons(profiles),
        "independent_cuda_event_findings": phase_findings(args.phase_analysis, profiles), "limits": LIMITS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    markdown_path = args.output.with_suffix(".md")
    markdown_temp = markdown_path.with_suffix(".md.tmp")
    markdown_temp.write_text(markdown(data))
    markdown_temp.replace(markdown_path)
    return 0 if data["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
