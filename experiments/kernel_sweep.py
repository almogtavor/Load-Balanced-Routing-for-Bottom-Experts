#!/usr/bin/env python3
"""Microbenchmark existing placement-router kernel launch configurations.

  python experiments/kernel_sweep.py --device cuda:0 --output /results/kernel_sweep.json

No production kernel changes. Tests num_warps=1/2/4 at EP4/K8/KEEP6/penalty1,
E128/256, T1/512/16384, float32/bfloat16. Outputs are preallocated. CUDA-event
measurements time graph-captured blocks so CPU launch gaps do not dominate tiny
kernels. Reported p95 is across block means, not individual-kernel p95. This omits
allocation/casting/engine/communication costs and cannot establish serving gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def greedy_reference(logits, torch, k=8, keep=6, ep=4, penalty=1.0):
    """Float32 oracle with deterministic lowest-index tie breaking, like tl.argmax.

    torch.topk does not guarantee the same tie order. The existing rewrite_torch
    is checked separately for continuous float32 inputs; this oracle covers ties
    in bf16 and deliberately tied inputs without accepting arbitrary selections.
    """
    z = logits.float()
    tokens, experts = z.shape
    rows = torch.arange(tokens, device=z.device)
    ranks = torch.arange(experts, device=z.device) * ep // experts
    taken = torch.zeros_like(z, dtype=torch.bool)
    load = torch.zeros((tokens, ep), device=z.device, dtype=torch.float32)
    chosen = []
    for slot in range(k):
        eligible = ~taken
        if slot >= keep:
            eligible = eligible & (load[:, ranks] > 0)
            scores = z - penalty * load[:, ranks]
        else:
            scores = z
        pick = scores.masked_fill(~eligible, -float("inf")).argmax(-1)
        chosen.append(pick)
        taken[rows, pick] = True
        load[rows, ranks[pick]] += 1
    ids = torch.stack(chosen, dim=1)
    weights = z.gather(1, ids).softmax(-1)
    return weights, ids.to(torch.int32)


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def assert_outputs(torch, weights, ids, ref_weights, ref_ids, context):
    if not torch.equal(ids, ref_ids):
        bad = int((ids != ref_ids).sum().item())
        raise AssertionError(f"{context}: {bad} mismatched expert IDs")
    torch.testing.assert_close(weights, ref_weights, atol=2e-6, rtol=2e-5,
                               msg=f"{context}: weights differ")


def run(args):
    import torch
    import triton
    import placement_router.kernel as router

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("Pass an explicit CUDA device, such as --device cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    props = torch.cuda.get_device_properties(device)
    result = {
        "complete": False, "device": str(device), "gpu": props.name,
        "gpu_total_memory": props.total_memory,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
        "kernel_file": str(Path(router.__file__).resolve()),
        "kernel_sha256": hashlib.sha256(Path(router.__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "warmup_calls": args.warmup, "calls_per_block": args.calls_per_block,
        "blocks": args.blocks, "ep": 4, "k": 8, "keep": 6, "load_penalty": 1.0,
        "method": "CUDA events around graph replay of preallocated direct-kernel launches; reported microseconds/call are block elapsed / calls_per_block. Percentiles summarize block means, not individual calls.",
        "limits": "Isolated microkernel timing excludes output allocations/casts, Python launch gaps, router integration, expert GEMM and all communication. No end-to-end serving speedup is implied.",
        "correctness": [], "measurements": []}
    atomic_json(args.output, result)

    def outputs(logits):
        return (torch.empty((logits.shape[0], 8), dtype=torch.float32, device=device),
                torch.empty((logits.shape[0], 8), dtype=torch.int32, device=device))

    def launch(logits, weights, ids, warps):
        experts = logits.shape[1]
        router._par_select[(logits.shape[0],)](
            logits, ids, weights, ids, weights, logits.stride(0), experts,
            K=8, KEEP=6, N_RANKS=4, BLOCK_E=triton.next_power_of_2(experts),
            HAS_PAD=False, HAS_SCALE=False, LOAD_PENALTY=1.0, num_warps=warps)

    with torch.cuda.device(device), torch.cuda.stream(stream):
        # Check ties explicitly before timing. These small cases make complete
        # expert-by-expert oracle comparison cheap and preserve strict tie rules.
        for experts in args.experts:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                for label in ("continuous", "all_equal", "repeating_ties", "quantized_ties"):
                    if label == "continuous":
                        logits = torch.randn((17, experts), generator=generator, device=device, dtype=torch.float32).to(dtype)
                    elif label == "all_equal":
                        logits = torch.zeros((17, experts), device=device, dtype=dtype)
                    elif label == "repeating_ties":
                        logits = (torch.arange(experts, device=device).remainder(7)
                                  .expand(17, -1).contiguous().to(dtype))
                    else:
                        logits = torch.randn((17, experts), generator=generator, device=device).mul(2).round().to(dtype)
                    expected_weights, expected_ids = greedy_reference(logits, torch)
                    stock_weights, stock_ids = router.select(logits, 8, 6, 4, load_penalty=1.0)
                    assert_outputs(torch, stock_weights, stock_ids, expected_weights, expected_ids,
                                   f"production select E{experts}/{dtype_name}/{label}")
                    if label == "continuous" and dtype == torch.float32:
                        ref_ids = router.rewrite_torch(logits, logits.topk(8, -1).indices, 6, 4, 1.0)
                        ref_weights = router.gate_weights(logits, ref_ids, renormalize=True)
                        assert_outputs(torch, stock_weights, stock_ids, ref_weights, ref_ids.to(torch.int32),
                                       "existing rewrite_torch continuous fp32 reference")
                    for warps in args.warps:
                        weights, ids = outputs(logits)
                        launch(logits, weights, ids, warps)
                        assert_outputs(torch, weights, ids, expected_weights, expected_ids,
                                       f"warps{warps} E{experts}/{dtype_name}/{label}")
                    result["correctness"].append({"experts": experts, "dtype": dtype_name,
                                                  "case": label, "rows": 17, "warps": args.warps,
                                                  "ids_exact_match": True, "weights_close": True})
        atomic_json(args.output, result)
        for experts in args.experts:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                for tokens in args.tokens:
                    logits = torch.randn((tokens, experts), generator=generator, device=device,
                                         dtype=torch.float32).to(dtype)
                    baseline_weights, baseline_ids = router.select(logits, 8, 6, 4, load_penalty=1.0)
                    group = []
                    # Rotate deterministic execution order to avoid always timing
                    # num_warps=4 last as GPU clocks/temperature evolve.
                    warps_order = args.warps.copy()
                    import random
                    random.Random(args.seed + tokens + experts).shuffle(warps_order)
                    for warps in warps_order:
                        weights, ids = outputs(logits)
                        for _ in range(args.warmup):
                            launch(logits, weights, ids, warps)
                        stream.synchronize()
                        assert_outputs(torch, weights, ids, baseline_weights, baseline_ids,
                                       f"timed full tensor E{experts}/T{tokens}/{dtype_name}/warps{warps}")
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            for _ in range(args.calls_per_block):
                                launch(logits, weights, ids, warps)
                        for _ in range(3):
                            graph.replay()
                        stream.synchronize()
                        pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                                 for _ in range(args.blocks)]
                        for start, end in pairs:
                            start.record(stream)
                            graph.replay()
                            end.record(stream)
                        stream.synchronize()
                        samples = [start.elapsed_time(end) * 1000 / args.calls_per_block
                                   for start, end in pairs]
                        median = statistics.median(samples)
                        item = {"experts": experts, "tokens": tokens, "dtype": dtype_name,
                                "num_warps": warps, "microseconds_per_call": samples,
                                "median_us_per_call": median, "p95_block_mean_us_per_call": quantile(samples, .95),
                                "min_block_mean_us_per_call": min(samples),
                                "max_block_mean_us_per_call": max(samples),
                                "full_tensor_ids_exact_match_production": True,
                                "full_tensor_weights_close_production": True}
                        group.append(item)
                        del graph
                    baseline = next(item for item in group if item["num_warps"] == 4)
                    for item in group:
                        item["speed_ratio_vs_warps4"] = baseline["median_us_per_call"] / item["median_us_per_call"]
                        item["median_duration_change_pct_vs_warps4"] = (item["median_us_per_call"] / baseline["median_us_per_call"] - 1) * 100
                        print(f"E={experts} T={tokens} {dtype_name} warps={item['num_warps']} "
                              f"median={item['median_us_per_call']:.3f}us "
                              f"p95(blockmean)={item['p95_block_mean_us_per_call']:.3f}us "
                              f"ratio_vs4={item['speed_ratio_vs_warps4']:.3f}x", flush=True)
                    result["measurements"].extend(sorted(group, key=lambda item: item["num_warps"]))
                    atomic_json(args.output, result)
    result["complete"] = True
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experts", nargs="+", type=int, choices=(128, 256), default=[128, 256])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 512, 16384])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "bfloat16"), default=["float32", "bfloat16"])
    parser.add_argument("--warps", nargs="+", type=int, choices=(1, 2, 4), default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--calls-per-block", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if 4 not in args.warps:
        parser.error("--warps must include 4 for baseline ratios")
    if min([args.warmup, args.calls_per_block, args.blocks, *args.tokens]) < 1:
        parser.error("Warmup, blocks, calls-per-block, and tokens must be positive")
    run(args)


if __name__ == "__main__":
    main()
