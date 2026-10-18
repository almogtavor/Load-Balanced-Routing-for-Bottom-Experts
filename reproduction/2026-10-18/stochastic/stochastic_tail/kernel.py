"""Isolated one-program-per-token stochastic selector; no public source edits.

This deliberately retains the public int32-output-then-backend-cast path. Direct
int64 writes are a separate optimization and are not mixed into this experiment.
"""
from __future__ import annotations

from .reference import MASK, parameters, rng_key

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _mix32(value):
        value = value.to(tl.uint32)
        value = value ^ (value >> 16)
        value = value * tl.full([], 0x7FEB352D, tl.uint32)
        value = value ^ (value >> 15)
        value = value * tl.full([], 0x846CA68B, tl.uint32)
        return value ^ (value >> 16)

    @triton.jit
    def _stochastic_select(logits, ids_out, weights_out, padding, words_out, scores_out, pool_out,
                           stride, KEY, E: tl.constexpr, K: tl.constexpr, KEEP: tl.constexpr,
                           RANKS: tl.constexpr, MODE: tl.constexpr, RANGE: tl.constexpr,
                           TEMPERATURE: tl.constexpr, HAS_PAD: tl.constexpr, DEBUG: tl.constexpr,
                           BLOCK: tl.constexpr):
        row = tl.program_id(0)
        expert = tl.arange(0, BLOCK)
        live = expert < E
        z = tl.load(logits + row * stride + expert, mask=live, other=-float("inf")).to(tl.float32)
        rank = tl.where(live, expert * RANKS // E, 0)
        taken = tl.full([BLOCK], False, tl.int1)
        column = tl.full([BLOCK], 0, tl.int32)
        hot = tl.full([], 0, tl.int32)
        for i in range(KEEP):
            choice = tl.argmax(tl.where(live & ~taken, z, -float("inf")), axis=0, tie_break_left=True)
            hit = expert == choice
            taken = taken | hit
            column = tl.where(hit, i, column)
            hot = hot | (1 << (choice * RANKS // E))
        allowed = live & (((hot >> rank) & 1) == 1)
        candidate = allowed & ~taken
        if MODE != 0:  # weighted_hot and matched uniform_range
            sorted_allowed = tl.sort(tl.where(allowed, z, -float("inf")), descending=True)
            cutoff = tl.sum(tl.where(expert == RANGE - 1, sorted_allowed, 0.0), axis=0)
            greater = allowed & (z > cutoff)
            equal = allowed & (z == cutoff)
            remaining = RANGE - tl.sum(greater.to(tl.int32), axis=0)
            # Stable allowed-rank boundary: lower expert ID wins an exact tie.
            equal_position = tl.cumsum(equal.to(tl.int32), axis=0)
            in_range = greater | (equal & (equal_position <= remaining))
            candidate = candidate & in_range
        counter = (row * E + expert).to(tl.uint32)
        words = _mix32(counter ^ KEY.to(tl.uint32))
        if MODE == 1:
            u = ((words >> 9).to(tl.float32) + 0.5) * (2.0 ** -23)
            scores = z / TEMPERATURE - tl.log(-tl.log(u))
        else:
            scores = words.to(tl.int64)
        if DEBUG:
            tl.store(words_out + row * E + expert, words.to(tl.int64), mask=live)
            tl.store(scores_out + row * E + expert, scores.to(tl.float32), mask=live)
            tl.store(pool_out + row * E + expert, candidate, mask=live)
        for i in range(K - KEEP):
            if MODE == 1:
                priority = tl.where(candidate & ~taken, scores, -float("inf"))
            else:
                priority = tl.where(candidate & ~taken, scores, -1)
            choice = tl.argmax(priority, axis=0, tie_break_left=True)
            hit = expert == choice
            taken = taken | hit
            column = tl.where(hit, KEEP + i, column)
        selected_z = tl.where(taken, z, -float("inf"))
        numerator = tl.exp(selected_z - tl.max(selected_z, axis=0))
        weights = numerator / tl.sum(numerator, axis=0)
        pad = tl.full([], False, tl.int1)
        if HAS_PAD:
            pad = tl.load(padding + row) != 0
        output_ids = tl.where(pad, -1, expert)
        output_weights = tl.where(pad, 0.0, weights)
        tl.store(ids_out + row * K + column, output_ids, mask=taken)
        tl.store(weights_out + row * K + column, output_weights, mask=taken)


def select(logits, k, keep, n_ranks, *, mode, seed, layer, call, range_limit=32,
           temperature=1.0, is_padding=None, indices_dtype=None, debug=False, physical_rank=0):
    import torch
    if triton is None or not logits.is_cuda:
        raise RuntimeError("Experimental production selector requires CUDA+Triton; use explicit reference for CPU")
    if logits.ndim != 2 or logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Expected finite floating CUDA logits[T,128]")
    t, e = logits.shape
    parameters(e, k, keep, n_ranks, mode, range_limit, temperature)
    key = rng_key(seed, layer, call, physical_rank)
    if physical_rank >= n_ranks:
        raise ValueError("Physical RNG rank outside EP world")
    if t * e > MASK:
        raise ValueError("Local row/expert RNG counter overflow")
    if is_padding is not None and (is_padding.shape != (t,) or is_padding.dtype != torch.bool
                                   or is_padding.device != logits.device):
        raise ValueError("Padding mask must be bool[T] on same CUDA device")
    if indices_dtype not in (None, torch.int32, torch.int64, torch.uint32):
        raise ValueError("Only the tested backend index dtypes are supported")
    logits = logits.contiguous()
    ids = torch.empty(t, k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(t, k, dtype=torch.float32, device=logits.device)
    words = torch.empty(t, e, dtype=torch.int64, device=logits.device) if debug else ids
    scores = torch.empty(t, e, dtype=torch.float32, device=logits.device) if debug else weights
    pool = torch.empty(t, e, dtype=torch.bool, device=logits.device) if debug else ids
    if t:
        _stochastic_select[(t,)](logits, ids, weights, ids if is_padding is None else is_padding,
            words, scores, pool, logits.stride(0), key, E=e, K=k, KEEP=keep,
            RANKS=n_ranks, MODE={"uniform_hot": 0, "weighted_hot": 1, "uniform_range": 2}[mode],
            RANGE=range_limit, TEMPERATURE=float(temperature), HAS_PAD=is_padding is not None,
            DEBUG=bool(debug), BLOCK=triton.next_power_of_2(e), num_warps=4, enable_fp_fusion=False)
    output = weights, ids.to(indices_dtype or torch.int32)
    return (*output, {"words": words, "scores": scores, "candidate": pool}) if debug else output
