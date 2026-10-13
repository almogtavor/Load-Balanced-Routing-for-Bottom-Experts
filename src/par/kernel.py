"""The placement-aware rewrite of a top-k selection, as a kernel.

A top-k MoE router sends every token to `k` experts. Under expert parallelism those experts
sit on up to `k` different ranks, and the all-to-all dispatch moves a token's hidden state
once per *destination rank* - so a layer's communication volume is `sum_t d(t)`, the per-token
rank fan-out, not `sum_t k`. The expert GEMM work is `sum_t k` and nothing here changes it.

The rule: keep the top `keep` picks by score, then refill the remaining `k - keep` slots with
the highest-scoring experts that live on ranks the head already forces a visit to. Fan-out
collapses to `|ranks of the top keep|` and every destination stays one the token was already
paying for. The premise that the bottom ranks of a top-k selection are substitutable is
published; `experiments/` is what measures the price on real checkpoints.

`select` *is* the routing, not a pass over its output: it takes the gate logits and returns
the selection and its gate weights, replacing the router's own top-k instead of following it.
That is the whole performance story. One Triton launch per MoE layer, where stock vLLM also
spends one, so `keep == top_k` asks for no substitution and costs nothing measurable - which
is what makes that arm a usable control. Following the router instead costs a second launch
per MoE layer per forward, and at 48 layers against a ~7 ms decode step that is ~3% of
throughput, which CUDA graphs cannot amortise away because DeepEP with data parallelism
disables them.

Two implementations, and the torch one is the specification:

* `rewrite_torch` is the readable reference, and the shape the experiment harness used: it
  builds `[T, E]` boolean temporaries, which is fine for an offline sweep and wasteful in a
  serving loop (~20 MB of traffic and several launches per MoE layer at T=16k, E=256).
* `_par_select` is one program per token with no `[T, E]` temporary at all: `k` register-
  resident argmax passes over an `E`-wide block, then the softmax over what they picked. It is
  the one that runs under vLLM.

The head is `keep` argmax passes over the whole logit row, so it is the global top-`keep` by
construction - nothing on the way in has to be sorted, or trusted to be.
"""

from __future__ import annotations

import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                                    # CPU-only dev boxes
    triton = None

__all__ = ["rank_of", "select", "select_triton", "rewrite_torch", "fanout", "gate_weights",
           "expected_fanout"]


def expected_fanout(n_picks: int, n_ranks: int, n_experts: int) -> float:
    """Mean distinct ranks a top-`n_picks` selection touches, for uniform random logits.

    A rank holds `E/G` of the `E` experts, so a top-k drawn without replacement misses it with
    probability `prod_i (E - E/G - i)/(E - i)`. The rewrite leaves a token's rank set equal to
    its top-`keep`'s rank set, so this is both the baseline fan-out at `n_picks = k` and the
    rewritten fan-out at `n_picks = keep` - and the ratio between them is the ceiling on what
    the rule can cut from dispatch volume, before any of it reaches a real routing
    distribution. It is governed by `n_ranks`, not by how many tokens are in flight, which is
    why the headroom grows with the width of the deployment and not with sequence length.
    """
    import math

    return n_ranks * (1 - math.prod((n_experts - n_experts // n_ranks - i) / (n_experts - i)
                                    for i in range(n_picks)))


def rank_of(n_experts: int, n_ranks: int, device=None) -> torch.Tensor:
    """Expert -> rank under vLLM's own contiguous expert-parallel split.

    `determine_expert_map` gives rank `r` the experts `[r * E/G, (r+1) * E/G)`, which is
    exactly `e * G // E`. Nothing here needs the map materialized - the kernel computes it
    arithmetically - but the reference path and the fan-out accounting read better with it.
    """
    return torch.arange(n_experts, device=device) * n_ranks // n_experts


def fanout(topk_ids: torch.Tensor, n_ranks: int, n_experts: int) -> torch.Tensor:
    """`[T]`: how many distinct ranks each token's selection touches. The comm volume unit."""
    rank = rank_of(n_experts, n_ranks, topk_ids.device)[topk_ids.long()]
    hot = torch.zeros(topk_ids.shape[0], n_ranks, dtype=torch.bool, device=topk_ids.device)
    hot.scatter_(1, rank, torch.ones_like(rank, dtype=torch.bool))
    return hot.sum(1)


def rewrite_torch(logits: torch.Tensor, topk_ids: torch.Tensor, keep: int,
                  n_ranks: int) -> torch.Tensor:
    """The reference rewrite. `logits` is `[T, E]`, `topk_ids` is `[T, k]`."""
    t, e = logits.shape
    k = topk_ids.shape[1]
    ids = topk_ids.long()
    rank = rank_of(e, n_ranks, logits.device)
    # The head is the top `keep` *of the incoming selection*, which for a global top-k is the
    # global top `keep`. Ranking the gathered scores rather than re-running a top-k over all
    # `E` is both cheaper and robust to a router that returns its picks unsorted.
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
    sub = logits.masked_fill(~pool, torch.finfo(logits.dtype).min).topk(k - keep, dim=-1).indices
    return torch.cat([head, sub], dim=1).to(topk_ids.dtype)


if triton is not None:

    @triton.jit
    def _par_select(logits_ptr, out_ptr, w_ptr, pad_ptr, scale_ptr, stride_lm, E,
                    K: tl.constexpr, KEEP: tl.constexpr, N_RANKS: tl.constexpr,
                    BLOCK_E: tl.constexpr, HAS_PAD: tl.constexpr, HAS_SCALE: tl.constexpr):
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
        # The head: `KEEP` argmax passes over the whole row, which is the global top-`KEEP`.
        for i in range(KEEP):
            e = tl.argmax(tl.where(live & ~taken, z, -float("inf")), axis=0)
            taken = taken | (offs == e)
            slot = tl.where(offs == e, i, slot)
            hot = hot | (1 << (e * N_RANKS // E))

        # The rest, drawn only from ranks the head already forces a visit to. A rank block
        # holds `E/G >= K` experts of which at most `KEEP` are head, so it is never empty.
        on_hot = (((hot >> rank) & 1) == 1) & live
        for i in range(K - KEEP):
            e = tl.argmax(tl.where(on_hot & ~taken, z, -float("inf")), axis=0)
            taken = taken | (offs == e)
            slot = tl.where(offs == e, KEEP + i, slot)

        zt = tl.where(taken, z, -float("inf"))
        p = tl.exp(zt - tl.max(zt, axis=0))
        w = p / tl.sum(p, axis=0)
        if HAS_SCALE:
            w = w * tl.load(scale_ptr + offs, mask=live, other=1.0).to(tl.float32)
        ids = offs
        if HAS_PAD:
            # vLLM's own sentinel: `VLLM_MOE_SKIP_PADDING` (on by default) forces a cudagraph
            # or DP padding token's expert ids to -1 and the dispatch and the experts drop the
            # row on that exact value. Routing it instead would add dispatch volume this arm
            # does not owe and stop `keep == top_k` from being a no-op.
            ids = tl.where(tl.load(pad_ptr + m), -1, offs)
        tl.store(out_ptr + m * K + slot, ids, mask=taken)
        tl.store(w_ptr + m * K + slot, w, mask=taken)


def select_triton(logits: torch.Tensor, k: int, keep: int, n_ranks: int,
                  is_padding: torch.Tensor | None = None,
                  scale: torch.Tensor | None = None,
                  indices_dtype: torch.dtype | None = None):
    """One program per token, no `[T, E]` temporary. `(gate weights, ids)`.

    `is_padding` is vLLM's `[T]` bool mask of cudagraph/DP padding rows, `scale` an optional
    `[E]` per-expert factor on the gate weights (Gemma's, folded in here so its path costs no
    launch either). The dummy pointers for an absent one are never dereferenced - the `HAS_*`
    constexprs compile the load away.
    """
    t, e = logits.shape
    out = torch.empty(t, k, dtype=torch.int32, device=logits.device)
    w = torch.empty(t, k, dtype=torch.float32, device=logits.device)
    _par_select[(t,)](logits, out, w, out if is_padding is None else is_padding,
                      w if scale is None else scale, logits.stride(0), e,
                      K=k, KEEP=keep, N_RANKS=n_ranks, BLOCK_E=triton.next_power_of_2(e),
                      HAS_PAD=is_padding is not None, HAS_SCALE=scale is not None,
                      num_warps=4)
    # int32 in the kernel and the cast after: `topk_indices_dtype` is unsigned on some
    # backends, and -1 reinterprets there as the 4294967295 vLLM's own padding path produces.
    return w, out.to(indices_dtype or torch.int32)


def select(logits: torch.Tensor, k: int, keep: int, n_ranks: int,
           is_padding: torch.Tensor | None = None, scale: torch.Tensor | None = None,
           indices_dtype: torch.dtype | None = None):
    """`select_triton` where it can run, the reference everywhere else."""
    if triton is not None and logits.is_cuda:
        return select_triton(logits.contiguous(), k, keep, n_ranks, is_padding, scale,
                             indices_dtype)
    ids = rewrite_torch(logits, logits.topk(k, -1).indices, keep, n_ranks)
    w = gate_weights(logits, ids, renormalize=True)
    if scale is not None:
        w = w * scale[ids.long()].to(w.dtype)
    if is_padding is not None:
        ids = ids.masked_fill(is_padding.unsqueeze(1), -1)
    return w, ids.to(indices_dtype or torch.int32)


def gate_weights(logits: torch.Tensor, topk_ids: torch.Tensor,
                 renormalize: bool) -> torch.Tensor:
    """The gate weights the router would have given these experts.

    The rewrite changes *which* experts a token visits; each one still carries the weight the
    gate itself assigns it. With renormalization that is the softmax over the selection, which
    is identical to `softmax(all) / sum(selected)` and reaches it in one step instead of two;
    without it, it is the full softmax read at the selected indices.
    """
    z = logits.float()
    zs = z.gather(1, topk_ids.long())
    return zs.softmax(-1) if renormalize else (zs - z.logsumexp(-1, keepdim=True)).exp()


def _check() -> None:
    """The rule's load-bearing properties, against closed-form expectations.

    Uniform random logits make the fan-out analytic: a rank holding `E/G` of `E` experts is
    untouched by a top-k drawn without replacement with probability `prod_i (E-E/G-i)/(E-i)`,
    so a token touches `G (1 - p(k))` ranks at the baseline and `G (1 - p(keep))` after the
    rewrite. An off-by-one in the rank map, a head that does not survive, or a substitute taken
    off a cold rank all move these away from the closed form.
    """
    torch.manual_seed(0)
    E, G, K, T = 256, 4, 8, 20000
    z = torch.randn(T, E)
    base = z.topk(K, -1).indices

    def closed_form(n):
        return expected_fanout(n, G, E)

    assert abs(fanout(base, G, E).float().mean() - closed_form(K)) < 0.02

    for keep in (K, 6, 4, 2):
        for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            zz, bb = z.to(dev), base.to(dev)
            gw, got = select(zz, K, keep, G)
            sel = got.long().sort(-1).values
            ref = rewrite_torch(zz, bb, keep, G)
            assert torch.equal(sel, ref.long().sort(-1).values), f"{dev} keep={keep}"
            # Exactly `K` distinct experts, and the head survives: the cost being measured has
            # to be the cost of substituting bottom picks, not of discarding good ones.
            assert sel.diff(dim=-1).min() > 0, "repeated expert"
            assert bool((got.unsqueeze(2) == bb[:, :keep].unsqueeze(1)).any(1).all()), "head"
            f = fanout(got, G, E).float().mean().item()
            assert abs(f - closed_form(keep)) < 0.02, f"keep={keep}: {f:.3f}"
            # At `keep == K` the rule asks for nothing and must return the selection itself,
            # or every other arm is measuring the rewrite's own perturbation.
            if keep == K:
                assert torch.equal(sel, bb.long().sort(-1).values), "no-op"
            # The weights the kernel emitted alongside the ids are the whole reason it can
            # replace a second routing pass, so they are checked against the reference on the
            # ids it actually returned - not on the reference's, which may order them
            # differently. float32 softmax in registers vs. torch's, hence 1e-6.
            assert torch.allclose(gw, gate_weights(zz, got, renormalize=True), atol=1e-6), \
                f"{dev} keep={keep} weights"
            # Gemma's per-expert output scale, folded inside the kernel rather than gathered
            # after it. Same selection, weights multiplied through - so the fold is free.
            sc = torch.rand(E, device=dev) + 0.5
            sw, sids = select(zz, K, keep, G, scale=sc)
            assert torch.equal(sids, got), f"{dev} keep={keep} scale changed the selection"
            assert torch.allclose(sw, gw * sc[got.long()], atol=1e-6), f"{dev} scale fold"
        ids = rewrite_torch(z, base, keep, G)
        w = gate_weights(z, ids, renormalize=True)
        assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-5)
        # Gemma keeps its own gate, and the plugin reproduces its weights rather than running
        # it a second time. The reference below is vLLM's `gemma4_routing_function_torch`,
        # inlined so that a change to it fails here: softmax over every expert, masked to the
        # selection, renormalized over it, then scaled per expert. The first three steps are
        # `gate_weights` - renormalizing divides out the denominator the mask changed - so
        # only the scale has to be folded on. If this identity ever stops holding, Gemma's
        # path has to go back to calling the router's own routing function.
        scale = torch.rand(E) + 0.5
        probs = z.softmax(-1) * torch.nn.functional.one_hot(ids.long(), E).sum(-2)
        ref = (probs / probs.sum(-1, keepdim=True)).gather(1, ids.long()) * scale[ids.long()]
        assert torch.allclose(w * scale[ids.long()], ref, atol=1e-6), "gemma scale fold"
        print(f"  keep={keep} fan-out {fanout(rewrite_torch(z, base, keep, G), G, E).float().mean():.3f} "
              f"(expected {closed_form(keep):.3f})")
    if torch.cuda.is_available():
        zz = z[:64].cuda()
        # vLLM marks cudagraph and DP padding tokens by forcing their expert ids to -1, and
        # the dispatch and the experts drop a row on that exact value - so the kernel has to
        # emit -1 and not merely something out of range, including under an unsigned index
        # dtype, where it has to arrive as the 4294967295 vLLM's own padding path produces.
        pad = torch.zeros(64, dtype=torch.bool, device="cuda")
        pad[::3] = True
        for dt in (torch.int32, torch.uint32):
            _, got = select(zz, K, 4, G, is_padding=pad, indices_dtype=dt)
            # Read back through int32 rather than widening: torch's unsigned dtypes do not
            # implement much, and the bit pattern is the whole point of the assertion.
            got = got.view(torch.int32)
            assert bool((got[::3] == -1).all()), f"padded rows must be the sentinel ({dt})"
            assert bool((got[1::3] < E).all()), "real rows in the same batch are still routed"
        # Not "the row changed" - that is only true with high probability. The rule itself is
        # exact: every expert the rewritten row visits lives on a rank its top-`keep` already
        # forced a visit to, so the row's rank set *is* its head's rank set.
        _, got = select(zz, K, 4, G)
        head = {int(e) // (E // G) for e in zz[0].topk(4).indices}
        assert {int(e) // (E // G) for e in got[0]} == head, "row collapsed onto its head"
        # And at `keep == top_k` the kernel has to *be* the router it replaces, weights
        # included. That arm carries the whole claim that the rule is free to install, so it
        # is checked against vLLM's own `topk_softmax` and not just against the reference.
        try:
            # The vendored copy's own function, which is the one being replaced - not
            # `ops.topk_softmax` directly, whose signature is vLLM's to change.
            from par.vendor.fused_topk_router import vllm_topk_softmax
        except Exception:                                              # no vLLM on this box
            vllm_topk_softmax = None
        if vllm_topk_softmax is not None:
            vw = torch.empty(64, K, dtype=torch.float32, device="cuda")
            vi = torch.empty(64, K, dtype=torch.int32, device="cuda")
            vt = torch.empty(64, K, dtype=torch.int32, device="cuda")
            vllm_topk_softmax(vw, vi, vt, zz, True)
            w, got = select(zz, K, K, G)
            # Compared by expert id rather than by slot: which slot an expert lands in is the
            # router's own business and changes nothing downstream, the id-to-weight map does.
            o, vo = got.argsort(-1), vi.argsort(-1)
            assert torch.equal(got.gather(1, o), vi.gather(1, vo)), "keep == top_k is not stock"
            assert torch.allclose(w.gather(1, o), vw.gather(1, vo), atol=1e-6), "stock weights"
            print("  keep == top_k reproduces vLLM's own topk_softmax, ids and weights")

    # The headroom, for the deployment widths the rule is aimed at. Printed because the number
    # that matters for throughput is not measured on this box: a rank count of 4 leaves almost
    # nothing to cut, so an arm that comes back flat there has not tested the rule.
    print(f"  headroom at E={128}, k={8} (dispatch volume cut, uniform logits):")
    for g in (2, 4, 8, 16, 32):
        base_f = expected_fanout(8, g, 128)
        cut = "  ".join(f"keep={keep} {100 * (1 - expected_fanout(keep, g, 128) / base_f):5.1f}%"
                        for keep in (6, 4, 2))
        print(f"    ep_size={g:<3} fan-out {base_f:5.2f}/{g:<3} {cut}")

    print(f"par.kernel: ok (triton {'present' if triton else 'absent'})")


if __name__ == "__main__":
    # The Triton half of this check is skipped wherever CUDA is missing, which is right on a
    # laptop and wrong on a GPU job: a node whose driver is sick reports no devices, and then
    # the one gate that proves the kernel agrees with the reference *on this hardware* passes
    # by testing nothing. `--cpu` is the laptop's way of asking for the reduced check, so a
    # bare run on a broken node fails here instead of measuring an engine that fell back.
    if "--cpu" not in sys.argv:
        assert torch.cuda.is_available(), "no CUDA device: run with --cpu to check the reference"
    _check()
