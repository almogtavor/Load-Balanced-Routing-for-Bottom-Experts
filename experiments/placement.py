"""Placement-aware re-routing of the bottom experts: the rule, and the fan-out it buys.

A top-k MoE router sends every token to k experts. Under expert parallelism those experts sit
on up to k different GPUs, and the all-to-all dispatch moves a token's hidden state once per
*destination GPU* - so a layer's communication volume is `sum_t d(t)`, the per-token GPU
fan-out, not `sum_t k`. Expert GEMM work, by contrast, is `sum_t k` and nothing here changes
it. The saving this repo measures is therefore purely communication, which is why it has to
be measured on real GPUs at a real batch size rather than counted.

The premise is published and was re-confirmed in the parent study: the top ranks of the
selection carry the decision and the bottom ranks are substitutable. If that holds, the
bottom `k - keep` picks are free to be chosen for *placement* instead of score - take them
from the GPUs the top `keep` experts already force a visit to, and `d(t)` collapses to
`|GPUs of the top keep|` while every destination stays one we were paying for anyway.

The rewrite is expressed as a **mask over the router logits**, not as a rewritten index list.
Every mask here leaves exactly `top_k` experts allowed - the top `keep`, which are the highest
logits and are never masked, plus the chosen substitutes - so whichever top-k the family's own
gate runs has no choice left to make and must return precisely that set. That is what makes
one hook correct for three incompatible gate contracts, and it is what makes the no-op arm
reproduce the baseline's selection token for token: leaving the allowed set wide and letting the
gate pick the substitutes would be the same rule in expectation, but bf16 logits tie often
enough that `topk` resolves a masked row differently from an unmasked one, and the no-op arm
would then be measuring tie-breaks. Order within the selection may still vary on a tie; that is
harmless, because tied logits carry equal weight. The softmax that weights the selection sees a
changed denominator, which the top-k renormalization divides back out - exactly in exact
arithmetic, to rounding in bf16, which is why the no-op arm is checked against the paired
test's resolution rather than against a literal zero.
"""

from __future__ import annotations

import math

import torch

__all__ = ["allowed_mask", "fanout", "gpu_of", "mask_logits", "random_mask"]


def gpu_of(n_experts: int, n_gpus: int, device=None) -> torch.Tensor:
    """Expert -> GPU under contiguous-block placement, balanced for any `(E, G)`.

    Contiguous blocks are what a plain `--expert-parallel-size` launch gives you, so this is
    the placement the measured baseline actually runs under. It is also the adversarial choice
    for us: any placement that deliberately co-locates frequently co-routed experts would make
    the baseline fan-out smaller and our headroom narrower, so a benefit measured here is not
    an artifact of a pathological layout.
    """
    return torch.arange(n_experts, device=device) * n_gpus // n_experts


def allowed_mask(logits: torch.Tensor, keep: int, gpu: torch.Tensor, n_gpus: int,
                 top_k: int) -> torch.Tensor:
    """`[T, E]` bool: the top-`keep`, plus the best substitutes already-visited GPUs offer.

    Exactly `top_k` experts, so the gate cannot deviate from this selection. A GPU block holds
    `E / G >= top_k` experts and at most `keep` of them are head, so there are always enough
    substitutes to fill the remaining slots.
    """
    head = logits.topk(keep, dim=-1).indices                       # [T, keep]
    hot = torch.zeros(logits.shape[0], n_gpus, dtype=torch.bool, device=logits.device)
    hot.scatter_(1, gpu[head], torch.ones_like(head, dtype=torch.bool))
    m = torch.zeros_like(logits, dtype=torch.bool)
    m.scatter_(1, head, torch.ones_like(head, dtype=torch.bool))
    if top_k > keep:
        # Highest logits among experts that are on a hot GPU and not already in the head.
        sub = mask_logits(logits, hot[:, gpu] & ~m).topk(top_k - keep, dim=-1).indices
        m.scatter_(1, sub, torch.ones_like(sub, dtype=torch.bool))
    return m                                                       # [T, E]


def random_mask(logits: torch.Tensor, keep: int, top_k: int,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Control arm: the same count of substitutions, chosen with no regard to placement.

    Without this arm the experiment cannot tell "the bottom experts do not matter" apart from
    "placement-aware selection is good". The random arm makes exactly as many substitutions
    and buys exactly no fan-out, so the accuracy gap between the two arms is the part of the
    cost that is attributable to *our* choice of replacement rather than to substituting at
    all.

    The allowed set is exactly `top_k` experts wide, so the gate's top-k selects all of it and
    the top-`keep` survive by construction - the same guarantee the placement arm gets from
    masking only below its own head.
    """
    head = logits.topk(keep, dim=-1).indices
    m = torch.zeros_like(logits, dtype=torch.bool)
    m.scatter_(1, head, torch.ones_like(head, dtype=torch.bool))
    # Ranking a uniform score with the kept experts pushed below the floor draws
    # `top_k - keep` distinct substitutes per token without a per-token loop.
    r = torch.rand(logits.shape, device=logits.device, generator=generator)
    r.masked_fill_(m, -1.0)
    m.scatter_(1, r.topk(top_k - keep, dim=-1).indices,
               torch.ones(logits.shape[0], top_k - keep, dtype=torch.bool, device=logits.device))
    return m


def mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Push disallowed experts below every allowed one, in the logits' own dtype.

    `dtype.min` rather than a literal: `-1e30` is finite in bf16 and fp32 but overflows fp16
    to `-inf`, and a row that went all-`-inf` would hand the gate's softmax a NaN. Every mask
    here leaves at least `top_k` experts allowed, so no row can go all-min.
    """
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def fanout(idx: torch.Tensor, gpu: torch.Tensor, n_gpus: int) -> torch.Tensor:
    """`[T]`: how many distinct GPUs each token's selection touches. The comm volume unit."""
    hot = torch.zeros(idx.shape[0], n_gpus, dtype=torch.bool, device=idx.device)
    hot.scatter_(1, gpu[idx.long()], torch.ones_like(idx, dtype=torch.bool))
    return hot.sum(1)


def _check() -> None:
    """The rule's three load-bearing properties, against closed-form expectations.

    Uniform random logits make the fan-out analytic: a GPU holding `E/G` of `E` experts is
    untouched by a top-k drawn *without* replacement with the hypergeometric probability
    `prod_i (E - E/G - i)/(E - i)`, so the baseline expects `G (1 - p(k))` GPUs per token and
    the placement arm expects `G (1 - p(keep))`. Anything that breaks the masking - a wrong
    allowed set, a head that does not survive, an off-by-one in the block map - moves these
    away from the closed form, which an eyeballed index dump would not have caught. The
    with-replacement approximation `((G-1)/G)^k` is 0.03 GPUs off here, which is the size of
    the effect at `keep = k - 2`, so it is not good enough to check against.
    """
    torch.manual_seed(0)
    E, G, K, T = 128, 4, 8, 20000
    gpu = gpu_of(E, G)
    assert gpu.max().item() == G - 1 and torch.bincount(gpu).unique().numel() == 1, "unbalanced"
    z = torch.randn(T, E)
    base = z.topk(K, -1).indices

    def closed_form(n):
        return G * (1 - math.prod((E - E // G - i) / (E - i) for i in range(n)))

    assert abs(fanout(base, gpu, G).float().mean() - closed_form(K)) < 0.02

    # A no-op mask must be the baseline selection itself, not merely a superset of it, or the
    # gate's own tie-breaking decides the substitutes and the no-op arm drifts off zero.
    assert torch.equal(allowed_mask(z, K, gpu, G, K).nonzero()[:, 1].view(-1, K).sort(-1).values,
                       base.sort(-1).values), "no-op mask is not the baseline selection"
    # bf16 ties are common enough at E=128 that this is the case the wide mask got wrong.
    zb = z.bfloat16()
    for keep in (K, 6, 4, 2):
        idx = mask_logits(zb, allowed_mask(zb, keep, gpu, G, K)).topk(K, -1).indices
        assert torch.equal(idx.sort(-1).values,
                           allowed_mask(zb, keep, gpu, G, K).nonzero()[:, 1].view(-1, K)), \
            f"bf16 keep={keep}: top-k did not return the allowed set"

    for keep in (6, 4, 2):
        for name, m in (("placement", allowed_mask(z, keep, gpu, G, K)),
                        ("random", random_mask(z, keep, K))):
            idx = mask_logits(z, m).topk(K, -1).indices
            # The head must survive in every arm, or the accuracy cost being measured is the
            # cost of discarding good experts rather than of substituting bad ones.
            head = base[:, :keep]
            assert bool((idx.unsqueeze(2) == head.unsqueeze(1)).any(1).all()), name
            assert idx.sort(-1).values.diff(dim=-1).min() > 0, f"{name}: repeated expert"
            assert int(m.sum(1).unique()) == K, f"{name}: allowed set is not top_k wide"
            got = fanout(idx, gpu, G).float().mean().item()
            want = closed_form(keep if name == "placement" else K)
            assert abs(got - want) < 0.02, f"{name} keep={keep}: fan-out {got:.3f} != {want:.3f}"
            print(f"  keep={keep} {name:10s} fan-out {got:.3f} (expected {want:.3f})")
    print("placement.py: ok")


if __name__ == "__main__":
    _check()
