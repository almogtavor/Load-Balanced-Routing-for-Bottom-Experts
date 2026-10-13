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

The kernel returns the gate weights with the selection, not just the ids. That is not a
convenience: the alternative is to hand the rewritten ids back to the router and let it weight
them, which costs a `[T, E]` mask and a second full routing pass per MoE layer per forward -
work that a serving loop pays on every token and that CUDA graphs cannot remove under DeepEP
with data parallelism, because that combination disables them. One program already has the
token's whole logit row resident, so the renormalized softmax over the new selection is one
reduction on registers.

Two implementations, and the torch one is the specification:

* `rewrite_torch` is the readable reference, and the shape the experiment harness used: it
  builds `[T, E]` boolean temporaries, which is fine for an offline sweep and wasteful in a
  serving loop (~20 MB of traffic and several launches per MoE layer at T=16k, E=256).
* `rewrite_triton` is one program per token with no `[T, E]` temporary at all: `k` register-
  resident argmax passes over an `E`-wide block, then the softmax over what they picked. It is
  the one that runs under vLLM.

Both take the selection *as it came out of the router* and never assume it is sorted by score,
so they are correct on any top-k implementation whose output is the global top-k.
"""

from __future__ import annotations

import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                                    # CPU-only dev boxes
    triton = None

__all__ = ["rank_of", "rewrite", "rewrite_torch", "rewrite_triton", "fanout", "gate_weights",
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
    def _par_rewrite(logits_ptr, ids_ptr, win_ptr, out_ptr, w_ptr, stride_lm, E,
                     K: tl.constexpr, KEEP: tl.constexpr, N_RANKS: tl.constexpr,
                     BLOCK_E: tl.constexpr):
        m = tl.program_id(0)
        offs = tl.arange(0, BLOCK_E)
        live = offs < E
        z = tl.load(logits_ptr + m * stride_lm + offs, mask=live,
                    other=-float("inf")).to(tl.float32)
        # Masked-off lanes must never contribute a rank bit, so their shift is forced to 0.
        rank = tl.where(live, offs * N_RANKS // E, 0)

        cur = offs < 0                                                 # all-false bool block
        for i in range(K):
            # `& live` is the range check: an id outside `[0, E)` lights no lane, so the count
            # below sees it as missing rather than as an expert at a dead lane.
            cur = cur | ((offs == tl.load(ids_ptr + m * K + i).to(tl.int32)) & live)

        if tl.sum(tl.where(cur, 1, 0)) == K:
            taken = offs < 0
            slot = offs * 0
            hot = 0
            # The head, recovered in descending-score order by `KEEP` argmax passes over the
            # incoming selection. Each pass also records the rank it forces a visit to, and the
            # output slot its winner owns - nothing is written out until both loops are done, so
            # the whole selection stays in registers and the weights below can see it.
            for i in range(KEEP):
                e = tl.argmax(tl.where(cur & ~taken, z, -float("inf")), axis=0)
                taken = taken | (offs == e)
                slot = tl.where(offs == e, i, slot)
                hot = hot | (1 << (e * N_RANKS // E))

            on_hot = (((hot >> rank) & 1) == 1) & live
            for i in range(K - KEEP):
                e = tl.argmax(tl.where(on_hot & ~taken, z, -float("inf")), axis=0)
                taken = taken | (offs == e)
                slot = tl.where(offs == e, KEEP + i, slot)

            # The gate weights of the new selection, from the row still in registers. With
            # renormalization an expert's weight is the softmax over the selection alone, so the
            # `E - K` other lanes contribute nothing and this is one max and one sum: they sit at
            # -inf and exponentiate to zero, which is why they can be summed over regardless.
            zt = tl.where(taken, z, -float("inf"))
            p = tl.exp(zt - tl.max(zt, axis=0))
            # Exactly `K` lanes are taken and they hold distinct slots - the count above is what
            # makes that true, and a hot rank's block holds `E/G >= K` experts so neither loop
            # can run out of candidates - so these two masked scatters write each output
            # position once. Were a slot left unwritten it would keep whatever `torch.empty`
            # handed us, and the engine indexes real buffers with these ids.
            tl.store(out_ptr + m * K + slot, offs, mask=taken)
            tl.store(w_ptr + m * K + slot, p / tl.sum(p, axis=0), mask=taken)
        else:
            # Not `K` distinct in-range experts, so not a top-k selection this rule can rewrite
            # - and vLLM does hand those over: `fused_topk` allocates `topk_ids` with
            # `torch.empty` and its top-k kernel leaves padded tokens untouched, so a padded
            # row holds whatever was in that memory. Copy the row straight through. The engine
            # discards those rows anyway; what it will not survive is an id invented for one.
            sel = offs < K
            tl.store(out_ptr + m * K + offs,
                     tl.load(ids_ptr + m * K + offs, mask=sel, other=0), mask=sel)
            tl.store(w_ptr + m * K + offs,
                     tl.load(win_ptr + m * K + offs, mask=sel, other=0.0), mask=sel)


def rewrite_triton(logits: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                   keep: int, n_ranks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One program per token, no `[T, E]` temporary. Returns `(ids, gate weights)`.

    `topk_weights` is only read for a row the rewrite declines, which it returns unchanged.
    """
    t, e = logits.shape
    k = topk_ids.shape[1]
    out = torch.empty_like(topk_ids, dtype=torch.int32)
    w = torch.empty(t, k, dtype=torch.float32, device=logits.device)
    _par_rewrite[(t,)](logits, topk_ids.to(torch.int32).contiguous(),
                       topk_weights.to(torch.float32).contiguous(), out, w, logits.stride(0),
                       e, K=k, KEEP=keep, N_RANKS=n_ranks,
                       BLOCK_E=triton.next_power_of_2(e), num_warps=4)
    return out.to(topk_ids.dtype), w


def rewrite(logits: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
            keep: int, n_ranks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`rewrite_triton` where it can run, the reference everywhere else.

    `(ids, weights)`, because a caller that has to weight the selection itself is a caller
    paying for a second routing pass - see the module docstring.
    """
    if triton is not None and logits.is_cuda:
        return rewrite_triton(logits.contiguous(), topk_ids, topk_weights, keep, n_ranks)
    # The reference is the offline path and assumes a well-formed selection; the kernel is the
    # one that meets vLLM's padded rows, and it is the one that checks for them.
    ids = rewrite_torch(logits, topk_ids, keep, n_ranks)
    return ids, gate_weights(logits, ids, renormalize=True)


def unroutable(topk_ids: torch.Tensor, e: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`([T, 1]` mask of rows that are not a selection, ids clamped into `[0, e)`).

    A caller that has to index an `[T, E]` tensor with these ids needs both: vLLM hands over
    padded rows (see the kernel's pass-through branch) and the kernel returns them untouched,
    so the ids can be anything at all. The test is "outside the expert range" rather than
    "negative", because `topk_indices_dtype` is unsigned on some backends and vLLM's own -1
    padding sentinel arrives there as 4294967295 - which is non-negative and still walks a
    `scatter_` off the end of the row.
    """
    ids = topk_ids.long()
    return ((ids < 0) | (ids >= e)).any(1, keepdim=True), ids.clamp(0, e - 1)


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
            # A router hands its picks over in whatever order its top-k produced, so the
            # rewrite is checked on a shuffled selection too.
            perm = bb.gather(1, torch.rand(T, K, device=dev).argsort(-1))
            # Zeros for the incoming weights on purpose: a well-formed selection must never
            # reach the kernel's pass-through branch, so the weight assertion below doubles as
            # the canary for a gate that fires when it should not.
            got, gw = rewrite(zz, perm, torch.zeros(T, K, device=dev), keep, G)
            ref = rewrite_torch(zz, bb, keep, G)
            assert torch.equal(got.sort(-1).values, ref.sort(-1).values), f"{dev} keep={keep}"
            # Exactly `K` distinct experts, and the head survives: the cost being measured has
            # to be the cost of substituting bottom picks, not of discarding good ones.
            assert got.sort(-1).values.diff(dim=-1).min() > 0, "repeated expert"
            assert bool((got.unsqueeze(2) == bb[:, :keep].unsqueeze(1)).any(1).all()), "head"
            f = fanout(got, G, E).float().mean().item()
            assert abs(f - closed_form(keep)) < 0.02, f"keep={keep}: {f:.3f}"
            # At `keep == K` the rule asks for nothing and must return the selection itself,
            # or every other arm is measuring the rewrite's own perturbation.
            if keep == K:
                assert torch.equal(got.sort(-1).values, bb.sort(-1).values), "no-op"
            # The weights the kernel emitted alongside the ids are the whole reason it can
            # replace a second routing pass, so they are checked against the reference on the
            # ids it actually returned - not on the reference's, which may order them
            # differently. float32 softmax in registers vs. torch's, hence 1e-6.
            assert torch.allclose(gw, gate_weights(zz, got, renormalize=True), atol=1e-6), \
                f"{dev} keep={keep} weights"
        ids = rewrite_torch(z, base, keep, G)
        w = gate_weights(z, ids, renormalize=True)
        assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-5)
        # Gemma keeps its own gate: the plugin masks every unselected expert to the dtype
        # floor and lets the model's routing function weight the selection itself. That is
        # only exact because renormalizing over the selection divides out the softmax
        # denominator the mask changed, which is the identity asserted here - the same one
        # that lets the kernel emit the weights directly for every other router.
        masked = torch.full_like(z, torch.finfo(z.dtype).min).scatter_(1, ids.long(),
                                                                      z.gather(1, ids.long()))
        assert torch.allclose(masked.softmax(-1).gather(1, ids.long()), w, atol=1e-6)
        print(f"  keep={keep} fan-out {fanout(rewrite_torch(z, base, keep, G), G, E).float().mean():.3f} "
              f"(expected {closed_form(keep):.3f})")
    if torch.cuda.is_available():
        # vLLM leaves a padded token's `topk_ids` row uninitialized, so the kernel is handed
        # ids that are no selection at all. Such a row has to come back exactly as it arrived:
        # the engine indexes real buffers with these ids, and the throughput phase of a serving
        # benchmark is where the padded rows show up.
        bad = base[:4].clone()
        bad[0, 3] = -1                                                 # a sentinel
        bad[1, 0] = 1 << 20                                            # plain garbage
        bad[2, 2] = bad[2, 5]                                          # a repeat is not a top-k
        bad, zz = bad.cuda(), z[:4].cuda()
        win = torch.rand(4, K, device="cuda")
        got, gw = rewrite(zz, bad, win, 4, G)
        assert torch.equal(got[:3], bad[:3]), "declined rows must pass through unchanged"
        assert torch.allclose(gw[:3], win[:3]), "declined rows keep their own weights"
        assert got[3].sort().values.diff().min() > 0, "a sound row in the batch still rewrites"
        # Gemma's path indexes `[T, E]` tensors with whatever came back, so it needs every
        # declined row flagged - including under an unsigned index dtype, where -1 is a large
        # positive number and a `< 0` test reads a padded row as a real expert.
        for ids in (got, got.to(torch.uint32)):
            miss, safe = unroutable(ids, E)
            assert bool(miss[:3].all()) and not bool(miss[3]), "declined rows must be flagged"
            assert int(safe.min()) >= 0 and int(safe.max()) < E, "clamped into range"
        # Not "the row changed" - that is only true with high probability. The rule itself is
        # exact: every expert the rewritten row visits lives on a rank its top-`keep` already
        # forced a visit to, so the row's rank set *is* its head's rank set.
        head = {int(e) // (E // G) for e in zz[3].topk(4).indices}
        assert {int(e) // (E // G) for e in got[3]} == head, "row collapsed onto its head"
        print("  padded/garbage rows pass through, sound rows in the same batch do not")

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
