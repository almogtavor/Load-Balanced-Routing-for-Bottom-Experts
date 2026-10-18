"""Opt-in routing launch/layout experiments; production defaults stay in kernel.py.

Public selectors preserve the production return contract and CPU reference fallback.
CUDA correctness and speed must be established with experiments/kernel_candidates.py
before any candidate is treated as a serving improvement.
"""
from __future__ import annotations

from functools import lru_cache, partial
import math
import re

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    gluon = None

_PRODUCTION = None


def _production():
    global _PRODUCTION
    if _PRODUCTION is None:
        from . import kernel
        _PRODUCTION = kernel
    return _PRODUCTION


def _require_legacy_cuda_kernel():
    """These prototypes predate the tile-sharing kernel and its launch ABI."""
    if hasattr(_production(), "LOAD_TILE"):
        raise NotImplementedError(
            "Experimental CUDA selectors target the older per-token kernel. "
            "Use PLACEMENT_EXPERIMENTAL_SELECTOR=baseline with the tile-sharing "
            "router; port and validate candidates against that reference first."
        )


@lru_cache(maxsize=128)
def _validate_config(experts, k, keep, n_ranks, load_penalty):
    """Validate immutable routing configuration once, not once per MoE layer."""
    if not 1 <= keep <= k <= experts:
        raise ValueError("require 1 <= keep <= k <= experts")
    if not 1 <= n_ranks <= 32 or experts % n_ranks:
        raise ValueError("require 1..32 equally sized contiguous ranks")
    if experts // n_ranks < k:
        raise ValueError("each rank must hold at least k experts")
    if not math.isfinite(load_penalty) or load_penalty < 0:
        raise ValueError("load penalty must be finite and nonnegative")


def _validate(logits, k, keep, n_ranks, is_padding, scale, load_penalty):
    if logits.ndim != 2:
        raise ValueError("router logits must have shape [tokens, experts]")
    tokens, experts = logits.shape
    _validate_config(experts, k, keep, n_ranks, load_penalty)
    if is_padding is not None and (is_padding.shape != (tokens,)
                                   or is_padding.device != logits.device):
        raise ValueError("padding must be [tokens] on the logits device")
    if scale is not None and (scale.shape != (experts,)
                              or scale.device != logits.device):
        raise ValueError("scale must be [experts] on the logits device")
    return tokens, experts


def _outputs(logits, k, tokens, indices_dtype=torch.int32):
    shape = (tokens, k)
    return (torch.empty(shape, dtype=torch.float32, device=logits.device),
            torch.empty(shape, dtype=indices_dtype, device=logits.device))


def _direct_dtype(indices_dtype):
    dtype = indices_dtype or torch.int32
    if dtype not in (torch.int32, torch.int64, torch.uint32):
        raise ValueError("direct ID output supports int32, int64, or uint32")
    return dtype


def _onewarp_cuda(logits, k, keep, n_ranks, is_padding, scale,
                  indices_dtype, load_penalty, tokens, experts, direct_indices=False):
    _require_legacy_cuda_kernel()
    logits = logits.contiguous()
    weights, ids = _outputs(logits, k, tokens,
                            indices_dtype if direct_indices else torch.int32)
    if tokens:
        _production()._par_select[(tokens,)](
            logits, ids, weights, ids if is_padding is None else is_padding,
            weights if scale is None else scale, logits.stride(0), experts,
            K=k, KEEP=keep, N_RANKS=n_ranks,
            BLOCK_E=triton.next_power_of_2(experts),
            HAS_PAD=is_padding is not None, HAS_SCALE=scale is not None,
            LOAD_PENALTY=float(load_penalty), num_warps=1)
    return weights, ids if direct_indices else ids.to(indices_dtype or torch.int32)


def select_onewarp(logits, k, keep, n_ranks, is_padding=None, scale=None,
                   indices_dtype=None, load_penalty=0.0, *, direct_indices=False):
    """Exactly the existing production kernel, launched with one warp per row."""
    tokens, experts = _validate(logits, k, keep, n_ranks, is_padding, scale, load_penalty)
    if direct_indices:
        indices_dtype = _direct_dtype(indices_dtype)
    if triton is None or not logits.is_cuda:
        return _production().select(logits, k, keep, n_ranks, is_padding, scale,
                                    indices_dtype, load_penalty)
    if torch.cuda.current_device() != logits.device.index:
        with torch.cuda.device(logits.device):
            return _onewarp_cuda(logits, k, keep, n_ranks, is_padding, scale,
                                  indices_dtype, load_penalty, tokens, experts, direct_indices)
    return _onewarp_cuda(logits, k, keep, n_ranks, is_padding, scale,
                          indices_dtype, load_penalty, tokens, experts, direct_indices)


if gluon is not None:

    @gluon.jit
    def _argmax_pair(v1, i1, v2, i2):
        # Same comparator as tl.argmax, including the lowest-index tie rule.
        take_left = (v1 > v2) | ((v1 == v2) & (i1 < i2))
        return gl.where(take_left, v1, v2), gl.where(take_left, i1, i2)

    @gluon.jit
    def _row_step(z, offs, rank, live, taken, slot, hot, pen,
                  step, HEAD: gl.constexpr, N_RANKS: gl.constexpr,
                  E: gl.constexpr, LOAD_PENALTY: gl.constexpr):
        if HEAD:
            eligible = live & ~taken
            scores = z
        else:
            eligible = ((((hot[:, None] >> rank[None, :]) & 1) == 1)
                        & live & ~taken)
            scores = z - pen
        values = gl.where(eligible, scores, -float("inf"))
        indices, _ = gl.broadcast(offs[None, :], values)
        _, e = gl.reduce((values, indices), 1, _argmax_pair)
        selected = offs[None, :] == e[:, None]
        taken = taken | selected
        slot = gl.where(selected, step, slot)
        re = e * N_RANKS // E
        hot = hot | (1 << re)
        if LOAD_PENALTY != 0.0:
            pen = pen + gl.where(rank[None, :] == re[:, None], LOAD_PENALTY, 0.0)
        return taken, slot, hot, pen

    @gluon.jit
    def _par_select_rows_gl(logits_ptr, out_ptr, w_ptr, pad_ptr, scale_ptr,
                            stride_lm, T, E: gl.constexpr,
                            K: gl.constexpr, KEEP: gl.constexpr,
                            N_RANKS: gl.constexpr, BLOCK_E: gl.constexpr,
                            ROWS: gl.constexpr, WARPS: gl.constexpr,
                            HAS_PAD: gl.constexpr, HAS_SCALE: gl.constexpr,
                            LOAD_PENALTY: gl.constexpr, UNROLL: gl.constexpr):
        # Exactly one warp owns each row. A warp handles additional rows when
        # ROWS > WARPS, rather than reducing a row across multiple warps.
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 32], [WARPS, 1], [1, 0])
        row_layout: gl.constexpr = gl.SliceLayout(1, layout)
        col_layout: gl.constexpr = gl.SliceLayout(0, layout)
        rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=row_layout)
        offs = gl.arange(0, BLOCK_E, layout=col_layout)
        live = offs[None, :] < E
        valid = rows[:, None] < T
        z = gl.load(logits_ptr + rows[:, None] * stride_lm + offs[None, :],
                    mask=valid & live, other=-float("inf")).to(gl.float32)
        rank = gl.where(offs < E, offs * N_RANKS // E, 0)
        taken = gl.full([ROWS, BLOCK_E], False, gl.int1, layout)
        slot = gl.full([ROWS, BLOCK_E], 0, gl.int32, layout)
        hot = gl.full([ROWS], 0, gl.int32, row_layout)
        pen = gl.full([ROWS, BLOCK_E], 0.0, gl.float32, layout)
        if UNROLL:
            for i in gl.static_range(KEEP):
                taken, slot, hot, pen = _row_step(
                    z, offs, rank, live, taken, slot, hot, pen,
                    i, True, N_RANKS, E, LOAD_PENALTY)
            for i in gl.static_range(K - KEEP):
                taken, slot, hot, pen = _row_step(
                    z, offs, rank, live, taken, slot, hot, pen,
                    KEEP + i, False, N_RANKS, E, LOAD_PENALTY)
        else:
            for i in range(KEEP):
                taken, slot, hot, pen = _row_step(
                    z, offs, rank, live, taken, slot, hot, pen,
                    i, True, N_RANKS, E, LOAD_PENALTY)
            for i in range(K - KEEP):
                taken, slot, hot, pen = _row_step(
                    z, offs, rank, live, taken, slot, hot, pen,
                    KEEP + i, False, N_RANKS, E, LOAD_PENALTY)
        zt = gl.where(taken, z, -float("inf"))
        p = gl.exp(zt - gl.max(zt, 1)[:, None])
        weights = p / gl.sum(p, 1)[:, None]
        if HAS_SCALE:
            weights *= gl.load(scale_ptr + offs, mask=offs < E, other=1.0)[None, :].to(gl.float32)
        ids, _ = gl.broadcast(offs[None, :], z)
        if HAS_PAD:
            padding = gl.load(pad_ptr + rows, mask=rows < T, other=False)
            ids = gl.where(padding[:, None], -1, ids)
        output = rows[:, None] * K + slot
        gl.store(out_ptr + output, ids, mask=taken & valid)
        gl.store(w_ptr + output, weights, mask=taken & valid)


if triton is not None:

    @triton.jit
    def _par_select_rows_tl(logits_ptr, out_ptr, w_ptr, pad_ptr, scale_ptr,
                            stride_lm, T, E: tl.constexpr,
                            K: tl.constexpr, KEEP: tl.constexpr,
                            N_RANKS: tl.constexpr, BLOCK_E: tl.constexpr,
                            ROWS: tl.constexpr, HAS_PAD: tl.constexpr,
                            HAS_SCALE: tl.constexpr, LOAD_PENALTY: tl.constexpr,
                            UNROLL: tl.constexpr):
        # A comparison implementation: let Triton's layout solver assign rows
        # instead of constraining the warp-to-row mapping with Gluon.
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        offs = tl.arange(0, BLOCK_E)
        live = offs[None, :] < E
        valid = rows[:, None] < T
        z = tl.load(logits_ptr + rows[:, None] * stride_lm + offs[None, :],
                    mask=valid & live, other=-float("inf")).to(tl.float32)
        rank = tl.where(offs < E, offs * N_RANKS // E, 0)
        taken = tl.full([ROWS, BLOCK_E], False, tl.int1)
        slot = tl.full([ROWS, BLOCK_E], 0, tl.int32)
        hot = tl.full([ROWS], 0, tl.int32)
        pen = tl.full([ROWS, BLOCK_E], 0.0, tl.float32)
        # loop_unroll_factor leaves the loop intact at 1, unrolls it at KEEP/K.
        for i in tl.range(0, KEEP, loop_unroll_factor=KEEP if UNROLL else 1):
            e = tl.argmax(tl.where(live & ~taken, z, -float("inf")), 1)
            selected = offs[None, :] == e[:, None]
            taken |= selected
            slot = tl.where(selected, i, slot)
            re = e * N_RANKS // E
            hot |= 1 << re
            if LOAD_PENALTY != 0.0:
                pen += tl.where(rank[None, :] == re[:, None], LOAD_PENALTY, 0.0)
        on_hot = (((hot[:, None] >> rank[None, :]) & 1) == 1) & live
        for i in tl.range(0, K - KEEP, loop_unroll_factor=K if UNROLL else 1):
            e = tl.argmax(tl.where(on_hot & ~taken, z - pen, -float("inf")), 1)
            selected = offs[None, :] == e[:, None]
            taken |= selected
            slot = tl.where(selected, KEEP + i, slot)
            if LOAD_PENALTY != 0.0:
                pen += tl.where(rank[None, :] == (e * N_RANKS // E)[:, None],
                                LOAD_PENALTY, 0.0)
        zt = tl.where(taken, z, -float("inf"))
        p = tl.exp(zt - tl.max(zt, 1)[:, None])
        weights = p / tl.sum(p, 1)[:, None]
        if HAS_SCALE:
            weights *= tl.load(scale_ptr + offs, mask=offs < E, other=1.0)[None, :].to(tl.float32)
        ids = tl.broadcast_to(offs[None, :], [ROWS, BLOCK_E])
        if HAS_PAD:
            padding = tl.load(pad_ptr + rows, mask=rows < T, other=False)
            ids = tl.where(padding[:, None], -1, ids)
        output = rows[:, None] * K + slot
        tl.store(out_ptr + output, ids, mask=taken & valid)
        tl.store(w_ptr + output, weights, mask=taken & valid)


def launch_preallocated(logits, weights, ids, k, keep, n_ranks,
                        is_padding=None, scale=None, load_penalty=0.0, *,
                        rows_per_program=4, num_warps=4, unroll=True,
                        backend="gluon"):
    """CUDA-only direct launch for the benchmark; caller owns output buffers.

    Input may have a larger row stride but must be contiguous within each row.
    Public selectors validate/contiguize inputs and guard the current device.
    """
    _require_legacy_cuda_kernel()
    tokens, experts = logits.shape
    if not tokens:
        return
    if rows_per_program not in (4, 8, 16) or num_warps not in (4, 8):
        raise ValueError("multirow candidates support rows4/8/16 and warps4/8")
    args = (logits, ids, weights, ids if is_padding is None else is_padding,
            weights if scale is None else scale, logits.stride(0), tokens, experts)
    kwargs = dict(K=k, KEEP=keep, N_RANKS=n_ranks,
                  BLOCK_E=triton.next_power_of_2(experts), ROWS=rows_per_program,
                  HAS_PAD=is_padding is not None, HAS_SCALE=scale is not None,
                  LOAD_PENALTY=float(load_penalty), UNROLL=unroll, num_warps=num_warps)
    grid = (triton.cdiv(tokens, rows_per_program),)
    if backend == "gluon":
        if gluon is None:
            raise RuntimeError("Gluon is unavailable in this Triton installation")
        _par_select_rows_gl[grid](*args, WARPS=num_warps, **kwargs)
    elif backend == "triton":
        _par_select_rows_tl[grid](*args, **kwargs)
    else:
        raise ValueError(f"unknown experimental backend: {backend}")


def select_multirow(logits, k, keep, n_ranks, is_padding=None, scale=None,
                    indices_dtype=None, load_penalty=0.0, *, rows_per_program=4,
                    num_warps=4, unroll=True, backend="gluon", direct_indices=False):
    """Multiple independent routing rows per block, preserving production semantics."""
    tokens, _ = _validate(logits, k, keep, n_ranks, is_padding, scale, load_penalty)
    if direct_indices:
        indices_dtype = _direct_dtype(indices_dtype)
    if triton is None or not logits.is_cuda:
        return _production().select(logits, k, keep, n_ranks, is_padding, scale,
                                    indices_dtype, load_penalty)
    logits = logits.contiguous()
    weights, ids = _outputs(logits, k, tokens,
                            indices_dtype if direct_indices else torch.int32)
    if torch.cuda.current_device() != logits.device.index:
        with torch.cuda.device(logits.device):
            launch_preallocated(logits, weights, ids, k, keep, n_ranks,
                                is_padding, scale, load_penalty,
                                rows_per_program=rows_per_program, num_warps=num_warps,
                                unroll=unroll, backend=backend)
    else:
        launch_preallocated(logits, weights, ids, k, keep, n_ranks,
                            is_padding, scale, load_penalty,
                            rows_per_program=rows_per_program, num_warps=num_warps,
                            unroll=unroll, backend=backend)
    return weights, ids if direct_indices else ids.to(indices_dtype or torch.int32)


def configured_selector(name):
    """Resolve once during kernel import; no environment reads in the hot path."""
    if name == "warp1":
        return select_onewarp
    if name == "warp1_direct":
        return partial(select_onewarp, direct_indices=True)
    match = re.fullmatch(r"rows(4|8|16)(?:w(4|8))?(_loop)?(_tl)?(_direct)?", name)
    if match:
        return partial(select_multirow, rows_per_program=int(match[1]),
                       num_warps=int(match[2] or 4), unroll=not bool(match[3]),
                       backend="triton" if match[4] else "gluon",
                       direct_indices=bool(match[5]))
    raise ValueError(f"unknown experimental selector {name!r}; choose baseline,warp1[_direct],rows4/8/16[w4/w8][_loop][_tl][_direct]")
