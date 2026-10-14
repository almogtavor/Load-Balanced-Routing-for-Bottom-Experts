"""The placement-aware rewrite of a top-k selection, as a kernel.

A top-k MoE router sends every token to `k` experts. Under expert parallelism those experts
sit on up to `k` different ranks, and the all-to-all dispatch moves a token's hidden state
once per *destination rank* - so a layer's communication volume is `sum_t d(t)`, the per-token
rank fan-out, not `sum_t k`. The expert GEMM work is `sum_t k` and nothing here changes it.

The rule: keep the top `keep` picks by score, then refill the remaining `k - keep` slots with
the highest-scoring experts that live on ranks the head already forces a visit to. Fan-out
collapses to `|ranks of the top keep|` and every destination stays one the token was already
paying for.

`load_penalty` is the correction for what that rule gets wrong. Summed volume is not what sets a step:
the all-to-all and the expert GEMM behind it both end when the *slowest rank* ends, and total
expert work is `sum_t k` whatever `keep` is - so only its spread over ranks can move. Taking the
highest logit on a hot rank piles the tail onto whichever rank the head already loaded most,
and `experiments/fanout.py` measures the busiest rank's work rising 10-20% while the summed
volume falls 24-52%. `load_penalty` charges each tail pick that much per slot already placed on its
rank, which removes the imbalance at the same volume. It is a different selection, so unlike
`keep` alone it has to be re-measured for quality, not just for rate.

`select` *is* the routing, not a pass over its output: it takes the gate logits and returns
the selection and its gate weights, replacing the router's own top-k instead of following it.
One Triton launch per MoE layer, where stock vLLM also spends one, so `keep == top_k` asks
for no substitution and costs nothing measurable - which is what makes that arm a control.

`rewrite_torch` is the readable reference and the specification; `_par_select` is one program
per token with no `[T, E]` temporary, and is the one that runs under vLLM.
"""

from __future__ import annotations

import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                                    # CPU-only dev boxes
    triton = None

__all__ = ["select", "rewrite_torch", "gate_weights", "expected_fanout"]


def expected_fanout(n_picks: int, n_ranks: int, n_experts: int) -> float:
    """Mean distinct ranks a top-`n_picks` selection touches, for uniform random logits.

    A rank holds `E/G` of the `E` experts, so a top-k drawn without replacement misses it with
    probability `prod_i (E - E/G - i)/(E - i)`. The rewrite leaves a token's rank set equal to
    its top-`keep`'s, so this is the baseline fan-out at `n_picks = k` and the rewritten one at
    `n_picks = keep` - and their ratio is the ceiling on what the rule can cut. It is governed
    by `n_ranks`, not by how many tokens are in flight.
    """
    import math

    return n_ranks * (1 - math.prod((n_experts - n_experts // n_ranks - i) / (n_experts - i)
                                    for i in range(n_picks)))


def rewrite_torch(logits: torch.Tensor, topk_ids: torch.Tensor, keep: int,
                  n_ranks: int, load_penalty: float = 0.0) -> torch.Tensor:
    """The reference rewrite. `logits` is `[T, E]`, `topk_ids` is `[T, k]`.

    Rank is `e * G // E`, which is the map `determine_expert_map` gives rank `r` the experts
    `[r * E/G, (r+1) * E/G)`. The head is ranked by gathered score rather than by re-running a
    top-k, which is cheaper and robust to a router that returns its picks unsorted.
    """
    t, e = logits.shape
    k = topk_ids.shape[1]
    ids = topk_ids.long()
    rank = torch.arange(e, device=logits.device) * n_ranks // e
    order = logits.gather(1, ids).argsort(dim=-1, descending=True)
    head = ids.gather(1, order[:, :keep])                              # [T, keep]
    if k <= keep:
        return head.to(topk_ids.dtype)
    hot = torch.zeros(t, n_ranks, dtype=torch.bool, device=logits.device)
    hot.scatter_(1, rank[head], torch.ones_like(head, dtype=torch.bool))
    taken = torch.zeros_like(logits, dtype=torch.bool)
    taken.scatter_(1, head, torch.ones_like(head, dtype=torch.bool))
    # Highest scores among experts on a hot rank and not already in the head. A rank block
    # holds `E/G >= k` experts and at most `keep` of them are head, so the pool is never empty.
    pool = hot[:, rank] & ~taken
    z = logits.masked_fill(~pool, torch.finfo(logits.dtype).min)
    if not load_penalty:
        sub = z.topk(k - keep, dim=-1).indices
        return torch.cat([head, sub], dim=1).to(topk_ids.dtype)
    # With a load penalty the picks stop being independent: each one changes what the next
    # one is choosing between, so this is `k - keep` argmaxes and not one top-k.
    load = torch.zeros(t, n_ranks, device=logits.device).scatter_add_(
        1, rank[head], torch.ones(head.shape, device=logits.device))
    rows = torch.arange(t, device=logits.device)
    sub = []
    for _ in range(k - keep):
        pick = (z - load_penalty * load[:, rank]).argmax(-1)
        sub.append(pick)
        z[rows, pick] = torch.finfo(logits.dtype).min
        load[rows, rank[pick]] += 1
    return torch.cat([head, torch.stack(sub, 1)], dim=1).to(topk_ids.dtype)


def gate_weights(logits: torch.Tensor, topk_ids: torch.Tensor,
                 renormalize: bool) -> torch.Tensor:
    """The gate weights the router would have given these experts.

    The rewrite changes *which* experts a token visits; each one still carries the weight the
    gate itself assigns it. With renormalization that is the softmax over the selection, which
    is identical to `softmax(all) / sum(selected)` and reaches it in one step instead of two.
    """
    z = logits.float()
    zs = z.gather(1, topk_ids.long())
    return zs.softmax(-1) if renormalize else (zs - z.logsumexp(-1, keepdim=True)).exp()


if triton is not None:

    @triton.jit
    def _par_select(logits_ptr, out_ptr, w_ptr, pad_ptr, scale_ptr, stride_lm, E,
                    K: tl.constexpr, KEEP: tl.constexpr, N_RANKS: tl.constexpr,
                    BLOCK_E: tl.constexpr, HAS_PAD: tl.constexpr, HAS_SCALE: tl.constexpr,
                    LOAD_PENALTY: tl.constexpr = 0.0):
        """Top-k and the placement-aware rewrite, in one pass over one token's logit row."""
        m = tl.program_id(0)
        offs = tl.arange(0, BLOCK_E)
        live = offs < E
        z = tl.load(logits_ptr + m * stride_lm + offs, mask=live,
                    other=-float("inf")).to(tl.float32)
        # Masked-off lanes must never contribute a rank bit, so their shift is forced to 0.
        rank = tl.where(live, offs * N_RANKS // E, 0)

        taken = offs < 0                                               # all-false bool block
        slot = offs * 0
        hot = 0
        # Slots already placed on this lane's rank, scaled. A constexpr 0.0 compiles both this
        # and its two updates away, so the plain rule pays nothing for the option.
        pen = tl.zeros([BLOCK_E], tl.float32)
        # The head: `KEEP` argmax passes over the whole row, which is the global top-`KEEP`
        # by construction - nothing on the way in has to be sorted, or trusted to be.
        for i in range(KEEP):
            e = tl.argmax(tl.where(live & ~taken, z, -float("inf")), axis=0)
            taken = taken | (offs == e)
            slot = tl.where(offs == e, i, slot)
            re = e * N_RANKS // E
            hot = hot | (1 << re)
            if LOAD_PENALTY != 0.0:
                pen = pen + tl.where(rank == re, LOAD_PENALTY, 0.0)

        # The rest, drawn only from ranks the head already forces a visit to.
        on_hot = (((hot >> rank) & 1) == 1) & live
        for i in range(K - KEEP):
            e = tl.argmax(tl.where(on_hot & ~taken, z - pen, -float("inf")), axis=0)
            taken = taken | (offs == e)
            slot = tl.where(offs == e, KEEP + i, slot)
            if LOAD_PENALTY != 0.0:
                pen = pen + tl.where(rank == e * N_RANKS // E, LOAD_PENALTY, 0.0)

        zt = tl.where(taken, z, -float("inf"))
        p = tl.exp(zt - tl.max(zt, axis=0))
        w = p / tl.sum(p, axis=0)
        if HAS_SCALE:
            w = w * tl.load(scale_ptr + offs, mask=live, other=1.0).to(tl.float32)
        ids = offs
        if HAS_PAD:
            # vLLM's own sentinel: `VLLM_MOE_SKIP_PADDING` (on by default) forces a cudagraph
            # or DP padding token's expert ids to -1, and the dispatch drops the row on that
            # exact value. Routing it would add volume this arm does not owe.
            ids = tl.where(tl.load(pad_ptr + m), -1, offs)
        tl.store(out_ptr + m * K + slot, ids, mask=taken)
        tl.store(w_ptr + m * K + slot, w, mask=taken)


def select(logits: torch.Tensor, k: int, keep: int, n_ranks: int,
           is_padding: torch.Tensor | None = None, scale: torch.Tensor | None = None,
           indices_dtype: torch.dtype | None = None, load_penalty: float = 0.0):
    """`(gate weights, ids)`. Triton where it can run, the reference everywhere else.

    `is_padding` is vLLM's `[T]` bool mask of cudagraph/DP padding rows, `scale` an optional
    `[E]` per-expert factor on the gate weights (Gemma's, folded in here so its path costs no
    launch either). The dummy pointers for an absent one are never dereferenced - the `HAS_*`
    constexprs compile the load away.
    """
    if triton is not None and logits.is_cuda:
        logits = logits.contiguous()
        t, e = logits.shape
        out = torch.empty(t, k, dtype=torch.int32, device=logits.device)
        w = torch.empty(t, k, dtype=torch.float32, device=logits.device)
        _par_select[(t,)](logits, out, w, out if is_padding is None else is_padding,
                          w if scale is None else scale, logits.stride(0), e,
                          K=k, KEEP=keep, N_RANKS=n_ranks, BLOCK_E=triton.next_power_of_2(e),
                          HAS_PAD=is_padding is not None, HAS_SCALE=scale is not None,
                          LOAD_PENALTY=float(load_penalty), num_warps=4)
        # int32 in the kernel and the cast after: `topk_indices_dtype` is unsigned on some
        # backends, and -1 reinterprets there as the 4294967295 vLLM's own padding produces.
        return w, out.to(indices_dtype or torch.int32)
    ids = rewrite_torch(logits, logits.topk(k, -1).indices, keep, n_ranks, load_penalty)
    w = gate_weights(logits, ids, renormalize=True)
    if scale is not None:
        w = w * scale[ids.long()].to(w.dtype)
    if is_padding is not None:
        ids = ids.masked_fill(is_padding.unsqueeze(1), -1)
    return w, ids.to(indices_dtype or torch.int32)


def _fanout(topk_ids: torch.Tensor, n_ranks: int, n_experts: int) -> torch.Tensor:
    """`[T]`: distinct ranks each token's selection touches. The comm volume unit."""
    rank = (torch.arange(n_experts, device=topk_ids.device) * n_ranks
            // n_experts)[topk_ids.long()]
    hot = torch.zeros(topk_ids.shape[0], n_ranks, dtype=torch.bool, device=topk_ids.device)
    hot.scatter_(1, rank, torch.ones_like(rank, dtype=torch.bool))
    return hot.sum(1)


def _check() -> None:
    """The rule's load-bearing properties, against closed-form expectations.

    Uniform random logits make the fan-out analytic (see `expected_fanout`), so an off-by-one
    in the rank map, a head that does not survive, or a substitute taken off a cold rank all
    move the measured fan-out away from the closed form.
    """
    torch.manual_seed(0)
    E, G, K, T = 256, 4, 8, 20000
    z = torch.randn(T, E)
    base = z.topk(K, -1).indices
    devs = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    assert abs(_fanout(base, G, E).float().mean() - expected_fanout(K, G, E)) < 0.02

    for keep, load_penalty in ((K, 0.0), (6, 0.0), (4, 0.0), (2, 0.0), (4, 1.0), (2, 0.5)):
        want = expected_fanout(keep, G, E)
        for dev in devs:
            zz, bb = z.to(dev), base.to(dev)
            gw, got = select(zz, K, keep, G, load_penalty=load_penalty)
            sel = got.long().sort(-1).values
            # The fan-out assertion below is the one that matters for `load_penalty`: a penalised
            # pick is still confined to the head's ranks, so the closed form is unchanged.
            assert torch.equal(sel,
                               rewrite_torch(zz, bb, keep, G, load_penalty).long().sort(-1).values)
            assert sel.diff(dim=-1).min() > 0, "repeated expert"
            assert bool((got.unsqueeze(2) == bb[:, :keep].unsqueeze(1)).any(1).all()), "head"
            f = _fanout(got, G, E).float().mean().item()
            assert abs(f - want) < 0.02, f"keep={keep}: {f:.3f} vs {want:.3f}"
            # At `keep == K` the rule asks for nothing and must return the selection itself,
            # or every other arm is measuring the rewrite's own perturbation.
            if keep == K:
                assert torch.equal(sel, bb.long().sort(-1).values), "no-op"
            # Checked on the ids the kernel returned, not the reference's, which may order
            # them differently. float32 softmax in registers vs. torch's, hence 1e-6.
            assert torch.allclose(gw, gate_weights(zz, got, renormalize=True), atol=1e-6), \
                f"{dev} keep={keep} weights"
            sc = torch.rand(E, device=dev) + 0.5
            sw, sids = select(zz, K, keep, G, scale=sc, load_penalty=load_penalty)
            assert torch.equal(sids, got), f"{dev} keep={keep} scale changed the selection"
            assert torch.allclose(sw, gw * sc[got.long()], atol=1e-6), f"{dev} scale fold"
        # Gemma keeps its own gate and the plugin reproduces its weights rather than running
        # it again. The reference is vLLM's `gemma4_routing_function_torch`, inlined so that a
        # change to it fails here: softmax over every expert, masked to the selection,
        # renormalized over it, then scaled. The first three are `gate_weights` - the
        # renormalization divides out the denominator the mask changed - so only the scale is
        # folded on. If this stops holding, Gemma's path has to call the router's own function.
        ids = rewrite_torch(z, base, keep, G, load_penalty)
        scale = torch.rand(E) + 0.5
        probs = z.softmax(-1) * torch.nn.functional.one_hot(ids.long(), E).sum(-2)
        ref = (probs / probs.sum(-1, keepdim=True)).gather(1, ids.long()) * scale[ids.long()]
        w = gate_weights(z, ids, renormalize=True) * scale[ids.long()]
        assert torch.allclose(w, ref, atol=1e-6), "gemma scale fold"
        print(f"  keep={keep} load_penalty={load_penalty} fan-out {_fanout(ids, G, E).float().mean():.3f} "
              f"(expected {want:.3f})")

    if torch.cuda.is_available():
        zz = z[:64].cuda()
        # The dispatch drops a padding row on the exact value -1, including under an unsigned
        # index dtype, where it has to arrive as the 4294967295 vLLM's own path produces.
        pad = torch.zeros(64, dtype=torch.bool, device="cuda")
        pad[::3] = True
        for dt in (torch.int32, torch.uint32):
            _, got = select(zz, K, 4, G, is_padding=pad, indices_dtype=dt)
            got = got.view(torch.int32)                 # torch's unsigned dtypes do little
            assert bool((got[::3] == -1).all()), f"padded rows must be the sentinel ({dt})"
            assert bool((got[1::3] < E).all()), "real rows in the same batch are still routed"
        # The rule stated exactly, rather than "the row changed", which is only probable:
        # the row's rank set *is* its head's rank set.
        _, got = select(zz, K, 4, G)
        head = {int(e) // (E // G) for e in zz[0].topk(4).indices}
        assert {int(e) // (E // G) for e in got[0]} == head, "row collapsed onto its head"
        # `keep == top_k` carries the whole claim that the rule is free to install, so it is
        # checked against vLLM's own `topk_softmax` and not just against the reference.
        try:
            from placement_router.vendor.fused_topk_router import vllm_topk_softmax
        except Exception:                                              # no vLLM on this box
            vllm_topk_softmax = None
        if vllm_topk_softmax is not None:
            vw = torch.empty(64, K, dtype=torch.float32, device="cuda")
            vi = torch.empty(64, K, dtype=torch.int32, device="cuda")
            vt = torch.empty(64, K, dtype=torch.int32, device="cuda")
            vllm_topk_softmax(vw, vi, vt, zz, True)
            w, got = select(zz, K, K, G)
            # By expert id, not by slot: which slot an expert lands in is the router's own
            # business and changes nothing downstream, the id-to-weight map does.
            o, vo = got.argsort(-1), vi.argsort(-1)
            assert torch.equal(got.gather(1, o), vi.gather(1, vo)), "keep == top_k is not stock"
            assert torch.allclose(w.gather(1, o), vw.gather(1, vo), atol=1e-6), "stock weights"
            print("  keep == top_k reproduces vLLM's own topk_softmax, ids and weights")

    print(f"placement_router.kernel: ok (triton {'present' if triton else 'absent'})")


if __name__ == "__main__":
    # The Triton half is skipped wherever CUDA is missing, which is right on a laptop and
    # wrong on a GPU job: a node whose driver is sick reports no devices, and then the one
    # gate that proves the kernel agrees with the reference *on this hardware* passes by
    # testing nothing. `--cpu` is the laptop's way of asking for the reduced check.
    if "--cpu" not in sys.argv:
        assert torch.cuda.is_available(), "no CUDA device: run with --cpu to check the reference"
    _check()
