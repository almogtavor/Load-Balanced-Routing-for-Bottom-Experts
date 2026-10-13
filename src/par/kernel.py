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

Two implementations, and the torch one is the specification:

* `rewrite_torch` is the readable reference, and the shape the experiment harness used: it
  builds `[T, E]` boolean temporaries, which is fine for an offline sweep and wasteful in a
  serving loop (~20 MB of traffic and several launches per MoE layer at T=16k, E=256).
* `rewrite_triton` is one program per token with no `[T, E]` temporary at all: `k` register-
  resident argmax passes over an `E`-wide block. It is the one that runs under vLLM.

Both take the selection *as it came out of the router* and never assume it is sorted by score,
so they are correct on any top-k implementation whose output is the global top-k.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                                    # CPU-only dev boxes
    triton = None

__all__ = ["rank_of", "rewrite", "rewrite_torch", "rewrite_triton", "fanout", "gate_weights"]


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
    def _par_rewrite(logits_ptr, ids_ptr, out_ptr, stride_lm, E,
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
            cur = cur | (offs == tl.load(ids_ptr + m * K + i).to(tl.int32))

        taken = offs < 0
        hot = 0
        # The head, recovered in descending-score order by `KEEP` argmax passes over the
        # incoming selection. Each pass also records the rank it forces a visit to.
        for i in range(KEEP):
            e = tl.argmax(tl.where(cur & ~taken, z, -float("inf")), axis=0)
            taken = taken | (offs == e)
            hot = hot | (1 << (e * N_RANKS // E))
            tl.store(out_ptr + m * K + i, e)

        on_hot = ((hot >> rank) & 1) == 1
        for i in range(K - KEEP):
            e = tl.argmax(tl.where(on_hot & ~taken, z, -float("inf")), axis=0)
            taken = taken | (offs == e)
            tl.store(out_ptr + m * K + KEEP + i, e)


def rewrite_triton(logits: torch.Tensor, topk_ids: torch.Tensor, keep: int,
                   n_ranks: int) -> torch.Tensor:
    """One program per token, no `[T, E]` temporary. Same contract as `rewrite_torch`."""
    t, e = logits.shape
    k = topk_ids.shape[1]
    out = torch.empty_like(topk_ids, dtype=torch.int32)
    _par_rewrite[(t,)](logits, topk_ids.to(torch.int32).contiguous(), out, logits.stride(0), e,
                       K=k, KEEP=keep, N_RANKS=n_ranks,
                       BLOCK_E=triton.next_power_of_2(e), num_warps=4)
    return out.to(topk_ids.dtype)


def rewrite(logits: torch.Tensor, topk_ids: torch.Tensor, keep: int,
            n_ranks: int) -> torch.Tensor:
    """`rewrite_triton` where it can run, the reference everywhere else."""
    if triton is not None and logits.is_cuda:
        return rewrite_triton(logits.contiguous(), topk_ids, keep, n_ranks)
    return rewrite_torch(logits, topk_ids, keep, n_ranks)


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
    import math

    torch.manual_seed(0)
    E, G, K, T = 256, 4, 8, 20000
    z = torch.randn(T, E)
    base = z.topk(K, -1).indices

    def closed_form(n):
        return G * (1 - math.prod((E - E // G - i) / (E - i) for i in range(n)))

    assert abs(fanout(base, G, E).float().mean() - closed_form(K)) < 0.02

    for keep in (K, 6, 4, 2):
        for dev in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            zz, bb = z.to(dev), base.to(dev)
            # A router hands its picks over in whatever order its top-k produced, so the
            # rewrite is checked on a shuffled selection too.
            perm = bb.gather(1, torch.rand(T, K, device=dev).argsort(-1))
            got = rewrite(zz, perm, keep, G)
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
        ids = rewrite_torch(z, base, keep, G)
        w = gate_weights(z, ids, renormalize=True)
        assert torch.allclose(w.sum(-1), torch.ones(T), atol=1e-5)
        # The plugin does not use `gate_weights` on a real router: it masks every unselected
        # expert to the dtype floor and lets the router weight the selection itself. That is
        # only exact because renormalizing over the selection divides out the softmax
        # denominator the mask changed, which is the identity asserted here.
        masked = torch.full_like(z, torch.finfo(z.dtype).min).scatter_(1, ids.long(),
                                                                      z.gather(1, ids.long()))
        assert torch.allclose(masked.softmax(-1).gather(1, ids.long()), w, atol=1e-6)
        print(f"  keep={keep} fan-out {fanout(rewrite_torch(z, base, keep, G), G, E).float().mean():.3f} "
              f"(expected {closed_form(keep):.3f})")
    print(f"par.kernel: ok (triton {'present' if triton else 'absent'})")


if __name__ == "__main__":
    _check()
