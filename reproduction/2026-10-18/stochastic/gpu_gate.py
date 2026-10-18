#!/usr/bin/env python3
"""Mandatory CUDA selector gate; no model/cluster operation or CPU fallback.

Writes a failed proof and exits nonzero on any contract failure. GPU programs
are compared with independent Torch sorting/hash/softmax mechanics. Weighted
keys use a frozen2e-5 tolerance for CUDA-log arithmetic; any changed tail must
have bounded reference-score regret and is explicitly counted, never hidden.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from stochastic_tail import reference
from stochastic_tail.plugin import private_hashes


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def need(value, message):
    if not value:
        raise RuntimeError(message)


def one(torch, select, native_topk, logits, *, keep, ranks, mode, seed, pad,
        index_dtype, call=13, layer=47, physical_rank=0):
    t, e = logits.shape
    kwargs = dict(mode=mode, seed=seed, layer=layer, call=call, range_limit=32,
                  temperature=1.0, indices_dtype=index_dtype, physical_rank=physical_rank)
    w, ids, debug = select(logits, 8, keep, ranks, is_padding=pad, debug=True, **kwargs)
    rw, ri, rd = reference.select_torch(logits, keep, ranks, padding=pad, return_debug=True, **kwargs)
    real = torch.ones(t, dtype=torch.bool, device=logits.device) if pad is None else ~pad
    signed = ids.to(torch.int64)
    if index_dtype == torch.uint32:
        signed = ids.view(torch.int32).long()
    ref_signed = ri.view(torch.int32).long() if index_dtype == torch.uint32 else ri.long()
    need(ids.dtype == index_dtype and w.dtype == torch.float32, "Backend index/weight dtype mismatch")
    need(torch.equal(debug["words"], rd["words"]), "Unsigned32 RNG words differ")
    need(torch.equal(debug["candidate"], rd["candidate"]), "Allowed top32/range/tie pool differs")
    need(torch.equal(signed[real, :keep], ref_signed[real, :keep]), "Exact preserved head differs")
    native_w = torch.empty(t, 8, device=logits.device, dtype=torch.float32)
    native_ids = torch.empty(t, 8, device=logits.device, dtype=torch.int32)
    token_expert = torch.empty_like(native_ids)
    native_topk(native_w, native_ids, token_expert, logits.float().contiguous(), True)
    native_by_id = native_ids.long().sort(-1).values
    native_score_order = logits.float().gather(1, native_by_id).argsort(dim=-1, descending=True, stable=True)
    native_head = native_by_id.gather(1, native_score_order)[:, :keep]
    need(torch.equal(signed[real, :keep], native_head[real]), "Preserved head differs from score-ranked actual vLLM native top-k")
    need(bool(((signed[real] >= 0) & (signed[real] < e)).all()), "Real expert ID out of range")
    need(bool((signed[real].sort(-1).values.diff(dim=-1) > 0).all()), "Repeated expert")
    if pad is not None:
        need(bool((signed[pad] == -1).all()) and bool((w[pad] == 0).all()), "Padding sentinel/zero weight differs")
    selected = signed[real]
    expected_weights = logits.float()[real].gather(1, selected).softmax(-1)
    need(torch.allclose(w[real], expected_weights, atol=2e-6, rtol=2e-6), "Original unperturbed gating weights not restored")
    need(torch.allclose(w[real].sum(-1), torch.ones(int(real.sum()), device=logits.device), atol=2e-6), "Weights do not renormalize")
    head_rank = selected[:, :keep] * ranks // e
    ranks_selected = selected * ranks // e
    need(bool((ranks_selected[:, :, None] == head_rank[:, None, :]).any(-1).all()), "Fanout expanded outside preserved head's ranks")
    near_tie_rows = 0
    max_key_error = 0.0
    if mode == "weighted_hot":
        need(bool(torch.isfinite(debug["scores"]).all()), "Nonfinite Gumbel key")
        max_key_error = float((debug["scores"] - rd["scores"]).abs().max()) if t else 0.0
        need(max_key_error <= 2e-5, "GPU Gumbel key disagrees beyond frozen floating tolerance")
        if keep < 8 and bool(real.any()):
            actual_tail = selected[:, keep:]
            need(bool(rd["candidate"][real].gather(1, actual_tail).all()), "Weighted tail outside frozen candidate range")
            ref_scores = rd["scores"][real].masked_fill(~rd["candidate"][real], -float("inf"))
            cutoff = ref_scores.topk(8 - keep, -1).values[:, -1]
            worst_selected = ref_scores.gather(1, actual_tail).min(-1).values
            need(bool((cutoff - worst_selected <= 4e-5).all()), "Selected weighted tail has excessive reference-score regret")
            near_tie_rows = int((selected[:, keep:].sort(-1).values != ref_signed[real, keep:].sort(-1).values).any(-1).sum())
    else:
        need(torch.equal(signed, ref_signed), "Uniform integer-key tail differs from exact Torch reference")
    again_w, again_ids = select(logits, 8, keep, ranks, is_padding=pad, debug=False, **kwargs)
    need(torch.equal(ids, again_ids) and torch.equal(w, again_w), "Same seed/key or debug-off production program changed outputs")
    return {"T": t, "keep": keep, "ranks": ranks, "mode": mode, "seed": seed,
            "logits_dtype": str(logits.dtype), "indices_dtype": str(index_dtype),
            "physical_rng_rank": physical_rank, "call": call, "layer": layer,
            "weighted_near_tie_tail_rows": near_tie_rows, "max_weighted_key_error": max_key_error}


def run(args):
    output = args.output.resolve()
    need(not output.exists(), "Preserve previous GPU gate; choose a new output")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema": "stochastic-tail-kernel-gate-v1", "complete": False, "cuda_required": True,
             "started_unix_ns": time.time_ns(), "private_source_sha256": private_hashes(),
             "gate_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "cases": [], "checks": {}, "weighted_key_tolerance": 2e-5,
             "weighted_tail_reference_regret_tolerance": 4e-5,
             "interpretation": "CUDA correctness gate only; no throughput or quality evidence"}
    save(output, state)
    try:
        import torch
        from stochastic_tail.kernel import select
        from placement_router.vendor.fused_topk_router import vllm_topk_softmax
        need(torch.cuda.is_available(), "CUDA unavailable; no CPU fallback may pass this gate")
        state.update(gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__))
        generator = torch.Generator(device="cpu").manual_seed(417)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for ranks in (4, 8):
                for keep in (4, 6):
                    for seed in (0, 1, 2):
                        z = torch.randn(17, 128, generator=generator).to(device="cuda", dtype=dtype)
                        pad = torch.arange(17, device="cuda") % 3 == 0
                        for mode in reference.MODES:
                            state["cases"].append(one(torch, select, vllm_topk_softmax, z, keep=keep, ranks=ranks,
                                mode=mode, seed=seed, pad=pad, index_dtype=torch.int64, physical_rank=ranks - 1))
                    save(output, state)
        for tokens in (0, 1, 2, 15, 16, 33, 257, 2051):
            z = torch.randn(tokens, 256, generator=generator).to("cuda")[:, ::2]
            for mode in reference.MODES:
                for dtype in (torch.int32, torch.int64, torch.uint32):
                    # T0 cannot call the upstream CUDA top-k op; empty outputs
                    # are checked directly and no GPU launch is required.
                    if not tokens:
                        w, ids = select(z, 8, 4, 4, mode=mode, seed=0, layer=0, call=0, indices_dtype=dtype)
                        need(w.shape == ids.shape == (0, 8) and ids.dtype == dtype, "Empty shape/dtype mismatch")
                    else:
                        pad = torch.arange(tokens, device="cuda") % 3 == 0
                        state["cases"].append(one(torch, select, vllm_topk_softmax, z, keep=4, ranks=4,
                            mode=mode, seed=0, pad=pad, index_dtype=dtype, call=0, layer=0))
            save(output, state)
        for keep in (4, 6, 8):
            for mode in reference.MODES:
                z = torch.zeros(33, 128, device="cuda")
                state["cases"].append(one(torch, select, vllm_topk_softmax, z, keep=keep, ranks=4,
                                         mode=mode, seed=0, pad=None, index_dtype=torch.int64))
                pad = torch.arange(33, device="cuda") % 3 == 0
                baseline_w, baseline_ids, d = select(z, 8, keep, 4, mode=mode, seed=0, layer=7, call=4, is_padding=pad, debug=True)
                changed = z.clone()
                changed[pad] = torch.randn(int(pad.sum()), 128, generator=generator).to("cuda") * 100
                changed_w, changed_ids = select(changed, 8, keep, 4, mode=mode, seed=0, layer=7, call=4, is_padding=pad)
                need(torch.equal(baseline_ids[~pad], changed_ids[~pad]) and torch.equal(baseline_w[~pad], changed_w[~pad]),
                     "Changing padded logits moved real-row routing")
                _, _, other = select(z, 8, keep, 4, mode=mode, seed=0, layer=7, call=5, is_padding=pad, debug=True)
                need(not torch.equal(d["words"], other["words"]), "Forward-call key not used")
        torch.cuda.synchronize()
        state["checks"] = {name: True for name in ["native_head", "distinct_experts", "dtype_and_sentinel", "original_weights",
            "ragged_and_padding", "rng_words_and_repeat", "fanout", "ties_and_range"]}
        state["complete"] = True
    except BaseException:
        state["error"] = traceback.format_exc()
        raise
    finally:
        state["finished_unix_ns"] = time.time_ns()
        save(output, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
