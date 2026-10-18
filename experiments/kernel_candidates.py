#!/usr/bin/env python3
"""Validate/time opt-in routing candidates against the unchanged production kernel.

  python experiments/kernel_candidates.py --device cuda:0 --output candidates.json

Start with --correctness smoke --candidates baseline warp1 rows4 rows8 rows16.
Use full correctness before serving a candidate. Timings are CUDA-event block means
of graph-captured, preallocated direct launches; they cannot establish E2E gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import traceback

from kernel_sweep import atomic_json, assert_outputs, greedy_reference, quantile

DEFAULT_CANDIDATES = ["baseline", "warp1", "rows4", "rows8", "rows16", "rows4w8",
                      "rows8w8", "rows16w8", "rows4_loop", "rows4_tl"]


def configuration(name):
    """Simple metadata for a named selector; validation stays in the module."""
    if name in ("baseline", "warp1"):
        return {"num_warps": 4 if name == "baseline" else 1, "rows_per_program": 1,
                "backend": "production", "unroll": False}
    import re
    match = re.fullmatch(r"rows(4|8|16)(?:w(4|8))?(_loop)?(_tl)?", name)
    if not match:
        raise ValueError(f"Invalid selector: {name}")
    return {"rows_per_program": int(match[1]), "num_warps": int(match[2] or 4),
            "unroll": not bool(match[3]), "backend": "triton" if match[4] else "gluon"}


def run(args):
    import torch
    import triton
    import placement_router.kernel as production
    import placement_router.experimental as experimental

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None:
        raise RuntimeError("Pass an available, explicit CUDA device such as cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    props = torch.cuda.get_device_properties(device)
    configs = {name: configuration(name) for name in args.candidates}
    result = {
        "complete": False, "all_correct": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device), "gpu": props.name, "gpu_total_memory": props.total_memory,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
        "import_time_dispatch_selector": production._EXPERIMENTAL_SELECTOR,
        "baseline_reference": "direct unchanged production._par_select with num_warps=4; select() dispatch is bypassed",
        "source_sha256": {str(Path(mod.__file__).resolve()): hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                          for mod in (production, experimental)},
        "candidates": configs, "seed": args.seed, "correctness_level": args.correctness,
        "warmup_calls": args.warmup, "calls_per_block": args.calls_per_block, "blocks": args.blocks,
        "method": "CUDA events around graph replay of preallocated direct-kernel launches; elapsed/calls_per_block. p95 summarizes block means, not individual calls. Candidate order is shuffled per tensor using a recorded seed.",
        "limits": "Excludes allocations/casts, CPU launch gaps, model execution, expert GEMM and communication. Isolated kernel gains do not establish serving gains.",
        "correctness": [], "failures": [], "measurements": []}
    atomic_json(args.output, result)
    failed = set()

    def outputs(logits, k=8):
        return (torch.empty((logits.shape[0], k), dtype=torch.float32, device=logits.device),
                torch.empty((logits.shape[0], k), dtype=torch.int32, device=logits.device))

    def launch(name, logits, weights, ids, keep=6, ep=4, penalty=1.0, padding=None, scale=None):
        config = configs[name] if name in configs else configuration(name)
        if config["backend"] == "production":
            production._par_select[(logits.shape[0],)](
                logits, ids, weights, ids if padding is None else padding,
                weights if scale is None else scale, logits.stride(0), logits.shape[1],
                K=8, KEEP=keep, N_RANKS=ep, BLOCK_E=triton.next_power_of_2(logits.shape[1]),
                HAS_PAD=padding is not None, HAS_SCALE=scale is not None,
                LOAD_PENALTY=penalty, num_warps=config["num_warps"])
        else:
            experimental.launch_preallocated(logits, weights, ids, 8, keep, ep,
                                              padding, scale, penalty, **config)

    def baseline(logits, keep=6, ep=4, penalty=1.0, padding=None, scale=None):
        weights, ids = outputs(logits)
        launch("baseline", logits, weights, ids, keep, ep, penalty, padding, scale)
        return weights, ids

    def failure(name, stage, context, error):
        failed.add(name)
        item = {"candidate": name, "stage": stage, "context": context,
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc()}
        result["failures"].append(item)
        atomic_json(args.output, result)
        print(f"FAILED {name} {stage} {context}: {error}", flush=True)

    with torch.cuda.device(device), torch.cuda.stream(stream):
        # All-equal and quantized cases force exact tie behavior. Partial row
        # blocks, padding, scales and non-power-of-two E exercise masked lanes.
        expert_sizes = list(dict.fromkeys(args.experts + ([96] if args.correctness == "full" else [])))
        for experts in expert_sizes:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                arms = [(6, 4, 1.0)]
                if args.correctness == "full":
                    arms += [(8, 4, 0.0), (6, 4, 0.0), (4, 4, 0.5), (6, 8, 1.0)]
                for keep, ep, penalty in arms:
                    for label in ("continuous", "all_equal", "repeating_ties", "quantized_ties"):
                        if label == "continuous":
                            logits = torch.randn((17, experts), generator=generator, device=device).to(dtype)
                        elif label == "all_equal":
                            logits = torch.zeros((17, experts), device=device, dtype=dtype)
                        elif label == "repeating_ties":
                            logits = torch.arange(experts, device=device).remainder(7).expand(17, -1).contiguous().to(dtype)
                        else:
                            logits = torch.randn((17, experts), generator=generator, device=device).mul(2).round().to(dtype)
                        refw, refids = greedy_reference(logits, torch, keep=keep, ep=ep, penalty=penalty)
                        bw, bi = baseline(logits, keep, ep, penalty)
                        context = f"E{experts}/{dtype_name}/{label}/keep{keep}/EP{ep}/pen{penalty}"
                        assert_outputs(torch, bw, bi, refw, refids, "baseline " + context)
                        for name in args.candidates:
                            if name in failed:
                                continue
                            try:
                                weights, ids = outputs(logits)
                                launch(name, logits, weights, ids, keep, ep, penalty)
                                assert_outputs(torch, weights, ids, refw, refids, name + " " + context)
                                result["correctness"].append({"candidate": name, "case": context,
                                    "rows": 17, "ids_exact_oracle": True, "weights_close_oracle": True,
                                    "weights_bitwise_production": torch.equal(weights, bw),
                                    "max_abs_weight_error_production": float((weights - bw).abs().max().item())})
                            except Exception as error:
                                failure(name, "correctness", context, error)
                        atomic_json(args.output, result)
                # A noncontiguous view forces public wrappers to copy their input;
                # direct launches above also test row-stride addressing.
                backing = torch.randn((513, experts * 2), generator=generator, device=device)
                logits = backing[:, ::2].to(dtype)
                if logits.stride(1) == 1:
                    # dtype conversion may make a contiguous copy.
                    logits = torch.empty((experts, 513), device=device, dtype=dtype).t()
                    logits.copy_(backing[:, ::2])
                padding = torch.arange(513, device=device).remainder(7) == 0
                scale = torch.linspace(0.0, 2.0, experts, device=device)
                bw, bi = baseline(logits.contiguous(), padding=padding, scale=scale)
                for name in args.candidates:
                    if name in failed:
                        continue
                    try:
                        if name == "baseline":
                            continue
                        selector = experimental.configured_selector(name)
                        weights, ids = selector(logits, 8, 6, 4, is_padding=padding, scale=scale, load_penalty=1.0)
                        assert_outputs(torch, weights, ids, bw, bi, name + " padding/scale/noncontiguous")
                        uw, ui = selector(logits, 8, 6, 4, is_padding=padding, scale=scale,
                                          indices_dtype=torch.uint32, load_penalty=1.0)
                        assert_outputs(torch, uw, ui, bw, bi.to(torch.uint32), name + " unsigned sentinel")
                        result["correctness"].append({"candidate": name, "experts": experts,
                            "dtype": dtype_name, "case": "padding/scales/noncontiguous/uint32",
                            "rows": 513, "ids_exact_production": True, "weights_close_production": True,
                            "max_abs_weight_error_production": float((weights - bw).abs().max().item())})
                    except Exception as error:
                        failure(name, "wrapper_correctness", f"E{experts}/{dtype_name}", error)
                atomic_json(args.output, result)
        # Guard against launching on the wrong GPU in HF device_map='auto'.
        # vLLM normally has current device already set correctly.
        if args.correctness == "full" and torch.cuda.device_count() > 1:
            other = (device.index + 1) % torch.cuda.device_count()
            logits = torch.randn((17, args.experts[0]), generator=generator, device=device)
            bw, bi = baseline(logits)
            for name in args.candidates:
                if name in failed or name == "baseline":
                    continue
                try:
                    selector = experimental.configured_selector(name)
                    with torch.cuda.device(other):
                        weights, ids = selector(logits, 8, 6, 4, load_penalty=1.0)
                        assert torch.cuda.current_device() == other, "selector failed to restore caller's device"
                    assert_outputs(torch, weights, ids, bw, bi, name + " device guard")
                    result["correctness"].append({"candidate": name, "case": "wrong_current_device",
                                                  "ids_exact_production": True, "weights_close_production": True})
                except Exception as error:
                    failure(name, "device_guard", "current CUDA device differs from input", error)
        atomic_json(args.output, result)
        for experts in args.experts:
            for dtype_name in args.dtypes:
                dtype = getattr(torch, dtype_name)
                for tokens in args.tokens:
                    logits = torch.randn((tokens, experts), generator=generator, device=device).to(dtype)
                    bw, bi = baseline(logits)
                    group = []
                    order = [name for name in args.candidates if name not in failed]
                    random.Random(args.seed + tokens + experts).shuffle(order)
                    for name in order:
                        try:
                            weights, ids = outputs(logits)
                            for _ in range(args.warmup):
                                launch(name, logits, weights, ids)
                            stream.synchronize()
                            assert_outputs(torch, weights, ids, bw, bi, f"full tensor {name}/E{experts}/T{tokens}/{dtype_name}")
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                for _ in range(args.calls_per_block):
                                    launch(name, logits, weights, ids)
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
                            samples = [start.elapsed_time(end) * 1000 / args.calls_per_block for start, end in pairs]
                            item = {"candidate": name, "experts": experts, "tokens": tokens, "dtype": dtype_name,
                                    "microseconds_per_call": samples, "median_us_per_call": statistics.median(samples),
                                    "p95_block_mean_us_per_call": quantile(samples, .95),
                                    "full_tensor_ids_exact_production": True, "full_tensor_weights_close_production": True,
                                    "max_abs_weight_error_production": float((weights - bw).abs().max().item()),
                                    "execution_order": order}
                            group.append(item)
                            del graph
                        except Exception as error:
                            failure(name, "timing", f"E{experts}/T{tokens}/{dtype_name}", error)
                    baseline_time = next(item["median_us_per_call"] for item in group if item["candidate"] == "baseline")
                    for item in group:
                        item["speed_ratio_vs_baseline"] = baseline_time / item["median_us_per_call"]
                        item["median_duration_change_pct_vs_baseline"] = (item["median_us_per_call"] / baseline_time - 1) * 100
                        print(f"E={experts} T={tokens} {dtype_name} {item['candidate']} "
                              f"median={item['median_us_per_call']:.3f}us "
                              f"ratio_vs_baseline={item['speed_ratio_vs_baseline']:.3f}x", flush=True)
                    result["measurements"].extend(group)
                    atomic_json(args.output, result)
    result["all_correct"] = not result["failures"]
    result["complete"] = True
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--correctness", choices=("smoke", "full"), default="full")
    parser.add_argument("--experts", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 512, 16384])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "bfloat16"), default=["float32", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--calls-per-block", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if "baseline" not in args.candidates:
        parser.error("Include baseline for strict production comparisons and ratios")
    if min([args.warmup, args.calls_per_block, args.blocks, *args.tokens]) < 1:
        parser.error("Warmup, blocks, calls per block, and tokens must be positive")
    for name in args.candidates:
        configuration(name)
    result = run(args)
    if result["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
