#!/usr/bin/env python3
"""Validate opt-in direct ID stores and compare captured public wrapper GPU work.

 python experiments/kernel_direct.py --device cuda:0 --output direct.json

Exact expert IDs are required for every dtype, including uint32 padding sentinel.
Timings capture complete selector calls, including old int32->int64/uint32 casts,
with one retained output pair per captured call. They exclude host validation,
allocation bookkeeping, and engine work. They do not establish serving gains.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random
import statistics
import time
import traceback

from kernel_sweep import atomic_json, assert_outputs, greedy_reference, quantile


def run(args):
    import torch
    import triton
    import placement_router.kernel as production
    import placement_router.experimental as experimental

    device = torch.device(args.device)
    if not torch.cuda.is_available() or device.type != "cuda" or device.index is None:
        raise RuntimeError("Pass an available explicit CUDA device such as cuda:0")
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream(device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    props = torch.cuda.get_device_properties(device)
    result = {
        "complete": False, "all_correct": False,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device), "gpu": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "unavailable")),
        "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
        "source_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (production.__file__, experimental.__file__, __file__)},
        "candidates": args.candidates, "output_dtypes": args.output_dtypes,
        "correctness_level": args.correctness, "seed": args.seed,
        "warmup_calls": args.warmup, "calls_per_block": args.calls_per_block, "blocks": args.blocks,
        "baseline_reference": "unchanged production._par_select fourwarp, int32 output followed by requested dtype cast; select() dispatch bypassed",
        "method": "CUDA events around graph replay of captured public selector calls. One output pair is retained per captured call. Includes GPU cast kernels for existing selectors. Elapsed/calls_per_block; p95 is across block means.",
        "limits": "Excludes host validation, allocation bookkeeping, CPU launch gaps and all engine/communication/expert work. Graph capture retains multiple output buffers, unlike same-output preallocated kernel_sweep. No E2E gain is implied.",
        "correctness": [], "failures": [], "measurements": []}
    atomic_json(args.output, result)
    failed = set()

    def baseline(logits, k, keep, ep, is_padding=None, scale=None,
                 indices_dtype=None, load_penalty=0.0):
        logits = logits.contiguous()
        tokens, experts = logits.shape
        ids = torch.empty((tokens, k), dtype=torch.int32, device=logits.device)
        weights = torch.empty((tokens, k), dtype=torch.float32, device=logits.device)
        production._par_select[(tokens,)](
            logits, ids, weights, ids if is_padding is None else is_padding,
            weights if scale is None else scale, logits.stride(0), experts,
            K=k, KEEP=keep, N_RANKS=ep, BLOCK_E=triton.next_power_of_2(experts),
            HAS_PAD=is_padding is not None, HAS_SCALE=scale is not None,
            LOAD_PENALTY=load_penalty, num_warps=4)
        return weights, ids.to(indices_dtype or torch.int32)

    selectors = {name: baseline if name == "baseline" else experimental.configured_selector(name)
                 for name in args.candidates}

    def failure(name, stage, context, error):
        failed.add(name)
        result["failures"].append({"candidate": name, "stage": stage, "context": context,
                                    "error_type": type(error).__name__, "error": str(error),
                                    "traceback": traceback.format_exc()})
        atomic_json(args.output, result)
        print(f"FAILED {name} {stage} {context}: {error}", flush=True)

    def compare(name, logits, dtype_name, keep, ep, penalty, context, padding=None, scale=None):
        dtype = getattr(torch, dtype_name)
        expected_w, expected_ids = greedy_reference(logits, torch, keep=keep, ep=ep, penalty=penalty)
        if scale is not None:
            expected_w *= scale[expected_ids.long()].float()
        if padding is not None:
            expected_ids = expected_ids.masked_fill(padding[:, None], -1)
        expected_ids = expected_ids.to(dtype)
        bw, bi = baseline(logits, 8, keep, ep, padding, scale, dtype, penalty)
        assert_outputs(torch, bw, bi, expected_w, expected_ids, "baseline " + context)
        weights, ids = selectors[name](logits, 8, keep, ep, padding, scale, dtype, penalty)
        assert ids.dtype == dtype, f"requested {dtype}, got {ids.dtype}"
        assert_outputs(torch, weights, ids, expected_w, expected_ids, name + " " + context)
        return {"candidate": name, "case": context, "output_dtype": dtype_name,
                "rows": logits.shape[0], "ids_exact_oracle": True, "weights_close_oracle": True,
                "max_abs_weight_error_production": float((weights - bw).abs().max().item())}

    with torch.cuda.device(device), torch.cuda.stream(stream):
        expert_sizes = list(dict.fromkeys(args.experts + ([96] if args.correctness == "full" else [])))
        for experts in expert_sizes:
            for input_dtype in args.dtypes:
                dtype = getattr(torch, input_dtype)
                arms = [(6, 4, 1.0)]
                if args.correctness == "full":
                    arms += [(8, 4, 0.0), (6, 4, 0.0), (4, 8, 0.5)]
                for keep, ep, penalty in arms:
                    for label in ("continuous", "all_equal", "repeating_ties", "quantized_ties"):
                        if label == "continuous":
                            z = torch.randn((17, experts), generator=generator, device=device)
                        elif label == "all_equal":
                            z = torch.zeros((17, experts), device=device)
                        elif label == "repeating_ties":
                            z = torch.arange(experts, device=device).remainder(7).expand(17, -1).contiguous()
                        else:
                            z = torch.randn((17, experts), generator=generator, device=device).mul(2).round()
                        logits = z.to(dtype)
                        padding = torch.arange(17, device=device).remainder(5) == 0
                        scale = torch.linspace(0.0, 2.0, experts, device=device)
                        for output_dtype in args.output_dtypes:
                            context = f"E{experts}/{input_dtype}/{label}/keep{keep}/EP{ep}/pen{penalty}/padding+scale"
                            for name in args.candidates:
                                if name in failed:
                                    continue
                                try:
                                    result["correctness"].append(compare(name, logits, output_dtype,
                                        keep, ep, penalty, context, padding, scale))
                                except Exception as error:
                                    failure(name, "correctness", context + "/" + output_dtype, error)
                            atomic_json(args.output, result)
                # Noncontiguous public input, partial row block, and every ID dtype.
                logits = torch.empty((experts, 513), device=device, dtype=dtype).t()
                logits.copy_(torch.randn((513, experts), generator=generator, device=device))
                padding = torch.arange(513, device=device).remainder(7) == 0
                scale = torch.linspace(0.0, 2.0, experts, device=device)
                for output_dtype in args.output_dtypes:
                    context = f"E{experts}/{input_dtype}/noncontiguous/padding+scale"
                    for name in args.candidates:
                        if name in failed:
                            continue
                        try:
                            result["correctness"].append(compare(name, logits, output_dtype,
                                6, 4, 1.0, context, padding, scale))
                        except Exception as error:
                            failure(name, "wrapper_correctness", context + "/" + output_dtype, error)
                atomic_json(args.output, result)
        if torch.cuda.device_count() > 1:
            other = (device.index + 1) % torch.cuda.device_count()
            logits = torch.randn((17, args.experts[0]), generator=generator, device=device)
            padding = torch.arange(17, device=device).remainder(3) == 0
            scale = torch.linspace(0.5, 1.5, args.experts[0], device=device)
            for output_dtype in args.output_dtypes:
                dtype = getattr(torch, output_dtype)
                bw, bi = baseline(logits, 8, 6, 4, padding, scale, dtype, 1.0)
                for name in args.candidates:
                    if name in failed or name == "baseline":
                        continue
                    try:
                        with torch.cuda.device(other):
                            weights, ids = selectors[name](logits, 8, 6, 4, padding, scale, dtype, 1.0)
                            assert torch.cuda.current_device() == other, "caller's device was not restored"
                        assert ids.dtype == dtype
                        assert_outputs(torch, weights, ids, bw, bi, name + " wrong-current-device")
                        result["correctness"].append({"candidate": name, "case": "wrong_current_device/padding+scale",
                                                      "output_dtype": output_dtype,
                                                      "ids_exact_production": True, "weights_close_production": True})
                    except Exception as error:
                        failure(name, "device_guard", output_dtype, error)
        atomic_json(args.output, result)
        for experts in args.experts:
            for input_dtype in args.dtypes:
                for tokens in args.tokens:
                    logits = torch.randn((tokens, experts), generator=generator, device=device).to(getattr(torch, input_dtype))
                    for output_dtype in args.output_dtypes:
                        dtype = getattr(torch, output_dtype)
                        bw, bi = baseline(logits, 8, 6, 4, indices_dtype=dtype, load_penalty=1.0)
                        group = []
                        order = [name for name in args.candidates if name not in failed]
                        random.Random(args.seed + tokens + experts).shuffle(order)
                        for name in order:
                            try:
                                selector = selectors[name]
                                for _ in range(args.warmup):
                                    warm = selector(logits, 8, 6, 4, indices_dtype=dtype, load_penalty=1.0)
                                stream.synchronize()
                                assert_outputs(torch, *warm, bw, bi, name + " timed tensor")
                                del warm
                                captured_outputs = []
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph, stream=stream):
                                    for _ in range(args.calls_per_block):
                                        captured_outputs.append(selector(logits, 8, 6, 4,
                                                                         indices_dtype=dtype, load_penalty=1.0))
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
                                assert_outputs(torch, *captured_outputs[-1], bw, bi, name + " captured output")
                                item = {"candidate": name, "experts": experts, "tokens": tokens,
                                        "input_dtype": input_dtype, "output_dtype": output_dtype,
                                        "microseconds_per_call": samples, "median_us_per_call": statistics.median(samples),
                                        "p95_block_mean_us_per_call": quantile(samples, .95),
                                        "ids_exact_production": True, "weights_close_production": True,
                                        "execution_order": order}
                                group.append(item)
                                del graph, captured_outputs
                            except Exception as error:
                                failure(name, "timing", f"E{experts}/T{tokens}/{input_dtype}/{output_dtype}", error)
                        base_time = next(item["median_us_per_call"] for item in group if item["candidate"] == "baseline")
                        by_name = {item["candidate"]: item for item in group}
                        for item in group:
                            item["speed_ratio_vs_baseline"] = base_time / item["median_us_per_call"]
                            old = item["candidate"].removesuffix("_direct")
                            if item["candidate"].endswith("_direct") and old in by_name:
                                old_time = by_name[old]["median_us_per_call"]
                                item["speed_ratio_vs_casting_variant"] = old_time / item["median_us_per_call"]
                            print(f"E={experts} T={tokens} {input_dtype}->{output_dtype} {item['candidate']} "
                                  f"median={item['median_us_per_call']:.3f}us", flush=True)
                        result["measurements"].extend(group)
                        atomic_json(args.output, result)
    result["complete"] = True
    result["all_correct"] = not result["failures"]
    result["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["baseline", "rows4", "rows4_direct", "warp1_direct"])
    parser.add_argument("--correctness", choices=("smoke", "full"), default="full")
    parser.add_argument("--experts", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "bfloat16"), default=["float32", "bfloat16"])
    parser.add_argument("--output-dtypes", nargs="+", choices=("int32", "int64", "uint32"), default=["int32", "int64", "uint32"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 512, 16384])
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--calls-per-block", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if "baseline" not in args.candidates:
        parser.error("Include baseline for strict comparisons")
    if min([args.warmup, args.calls_per_block, args.blocks, *args.tokens]) < 1:
        parser.error("Warmup, calls per block, blocks and tokens must be positive")
    result = run(args)
    if result["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
