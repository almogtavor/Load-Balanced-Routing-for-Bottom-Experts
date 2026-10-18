"""Readable finite-input oracle and lazy Torch reference for stochastic tails.

Weighted-hot is a placement adaptation, not the paper's unrestricted global
range. Allowed rank positions1..32 include the native head; the tail samples
positionskeep+1..32. Uniform-hot samples all remaining hot-rank experts. Their
comparison changes both weighting and range. Uniform-range is a diagnostic
matched-pool control. No load-balancing price is applied to stochastic arms.
"""
from __future__ import annotations
import math
import struct

MASK = (1 << 32) - 1
MODES = ("uniform_hot", "weighted_hot", "uniform_range")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mix32(value):
    value &= MASK
    value ^= value >> 16
    value = value * 0x7FEB352D & MASK
    value ^= value >> 15
    value = value * 0x846CA68B & MASK
    return (value ^ (value >> 16)) & MASK


def rng_key(seed, layer, call, physical_rank=0):
    require(all(type(x) is int and 0 <= x <= MASK for x in (seed, call)), "Seed/call must fit uint32")
    require(type(layer) is int and 0 <= layer < 48, "Layer must be native Qwen index0..47")
    require(type(physical_rank) is int and 0 <= physical_rank < 8, "Physical RNG rank must be0..7")
    return mix32(seed) ^ mix32(layer + 0x9E3779B9) ^ mix32(call + 0x85EBCA6B) ^ mix32(physical_rank + 0xC2B2AE35) ^ 0xD1B54A35


def random_word(seed, layer, call, row, expert, experts=128, physical_rank=0):
    require(type(row) is int and type(expert) is int and 0 <= row * experts + expert <= MASK
            and 0 <= expert < experts, "Row/expert counter overflow or range error")
    return mix32((row * experts + expert) ^ rng_key(seed, layer, call, physical_rank))


def uniform_open(word):
    """Exactly representable float32 open interval;23 high random bits."""
    return ((word >> 9) + .5) * (2.0 ** -23)


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def gumbel(word):
    # CPU mathematical oracle. CUDA log implementations can differ slightly;
    # GPU gate checks scores/margins and never silently blesses arbitrary IDs.
    return f32(-math.log(-math.log(uniform_open(word))))


def parameters(experts, k, keep, ranks, mode, range_limit, temperature):
    require(experts == 128 and k == 8 and keep in (4, 6, 8), "Only Qwen E128/K8/keep4,6,8 is supported")
    require(ranks in (4, 8) and experts % ranks == 0 and experts // ranks >= k, "Need linear EP4/EP8 with enough rank-local experts")
    require(mode in MODES and type(range_limit) is int and k <= range_limit <= experts, "Invalid stochastic mode/allowed range")
    require(type(temperature) in (float, int) and math.isfinite(temperature) and temperature > 0, "Temperature must be finite and positive")


def select_python(logits, keep, ranks, mode, *, seed, layer, call, range_limit=32,
                  temperature=1.0, padding=None, physical_rank=0):
    require(isinstance(logits, list) and logits, "CPU oracle requires nonempty list rows")
    e, k = len(logits[0]), 8
    parameters(e, k, keep, ranks, mode, range_limit, temperature)
    rng_key(seed, layer, call, physical_rank)
    require(physical_rank < ranks, "Physical RNG rank outside EP world")
    require(all(len(row) == e and all(math.isfinite(z) for z in row) for row in logits), "Finite rectangular logits required")
    require(padding is None or len(padding) == len(logits), "Padding shape differs")
    all_ids, weights = [], []
    for r, original in enumerate(logits):
        z = [f32(x) for x in original]
        order = sorted(range(e), key=lambda j: (-z[j], j))
        head = order[:keep]
        hot = {j * ranks // e for j in head}
        allowed = [j for j in order if j * ranks // e in hot]
        pool = allowed[keep:] if mode == "uniform_hot" else allowed[keep:range_limit]
        require(len(pool) >= k - keep, "Candidate pool too small; no silent expansion")
        if mode.startswith("uniform"):
            tail = sorted(pool, key=lambda j: (-random_word(seed, layer, call, r, j, e, physical_rank), j))[:k - keep]
        else:
            tail = sorted(pool, key=lambda j: (-f32(f32(z[j] / temperature) + gumbel(random_word(seed, layer, call, r, j, e, physical_rank))), j))[:k - keep]
        ids = head + tail
        selected = [z[j] for j in ids]
        maximum = max(selected)
        values = [math.exp(x - maximum) for x in selected]
        total = sum(values)
        if padding is not None and padding[r]:
            all_ids.append([-1] * k)
            weights.append([0.0] * k)
        else:
            all_ids.append(ids)
            weights.append([f32(x / total) for x in values])
    return weights, all_ids


def mix32_torch(value):
    value = value & MASK
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & MASK
    value = ((value ^ (value >> 15)) * 0x846CA68B) & MASK
    return (value ^ (value >> 16)) & MASK


def select_torch(logits, keep, ranks, mode, *, seed, layer, call, range_limit=32,
                 temperature=1.0, padding=None, indices_dtype=None, return_debug=False, physical_rank=0):
    import torch
    require(logits.ndim == 2, "Logits must be[T,E]")
    t, e = logits.shape
    parameters(e, 8, keep, ranks, mode, range_limit, temperature)
    key = rng_key(seed, layer, call, physical_rank)
    require(physical_rank < ranks, "Physical RNG rank outside EP world")
    require(t * e <= MASK, "Local row counter would overflow")
    require(padding is None or (padding.shape == (t,) and padding.dtype == torch.bool and padding.device == logits.device), "Padding mask differs")
    z = logits.float()
    require(bool(torch.isfinite(z).all()), "Reference requires finite logits")
    order = z.argsort(dim=-1, descending=True, stable=True)
    head = order[:, :keep]
    rank_ids = torch.arange(e, device=logits.device) * ranks // e
    hot = torch.zeros(t, ranks, dtype=torch.bool, device=logits.device)
    hot.scatter_(1, rank_ids[head], True)
    allowed = hot[:, rank_ids]
    taken = torch.zeros(t, e, dtype=torch.bool, device=logits.device)
    taken.scatter_(1, head, True)
    if mode != "uniform_hot":
        allowed_sorted = allowed.gather(1, order)
        in_range_sorted = allowed_sorted & (allowed_sorted.long().cumsum(-1) <= range_limit)
        candidate = torch.zeros_like(allowed)
        candidate.scatter_(1, order, in_range_sorted)
        candidate &= ~taken
    else:
        candidate = allowed & ~taken
    counter = torch.arange(t * e, device=logits.device, dtype=torch.int64).reshape(t, e)
    words = mix32_torch(counter ^ key)
    if mode.startswith("uniform"):
        scores = words
        ranked = scores.masked_fill(~candidate, -1).argsort(dim=-1, descending=True, stable=True)
    else:
        u = ((words >> 9).float() + .5) * (2.0 ** -23)
        scores = z / float(temperature) - torch.log(-torch.log(u))
        ranked = scores.masked_fill(~candidate, -float("inf")).argsort(dim=-1, descending=True, stable=True)
    ids = torch.cat([head, ranked[:, :8 - keep]], -1)
    weights = z.gather(1, ids).softmax(-1)
    if padding is not None:
        ids = ids.masked_fill(padding[:, None], -1)
        weights = weights.masked_fill(padding[:, None], 0.0)
    result = weights, ids.to(indices_dtype or torch.int32)
    if return_debug:
        return *result, {"head": head, "candidate": candidate, "words": words, "scores": scores}
    return result
