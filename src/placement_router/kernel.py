"""The placement-aware rewrite of a top-k selection, as a kernel.

A top-k MoE router sends every token to `k` experts. Under expert parallelism those experts
sit on up to `k` different ranks, and the all-to-all dispatch moves a token's hidden state
once per *destination rank* - so a layer's communication volume is `sum_t d(t)`, the per-token
rank fan-out, not `sum_t k`. The expert GEMM work is `sum_t k` and nothing here changes it.

The rule: keep the top `keep` picks by score, then replace the remaining `k - keep` picks with
the highest-scoring experts that live on ranks the head already forces a visit to. Fan-out
collapses to `|ranks of the top keep|` and every destination stays one the token was already
paying for.

`load_penalty` is the correction for what that rule gets wrong. Summed volume is not what sets a step:
the all-to-all and the expert GEMM behind it both end when the *slowest rank* ends, and total
expert work is `sum_t k` whatever `keep` is - so only its spread over ranks can move. Taking the
highest logit on a hot rank piles the tail onto whichever rank the head already loaded most,
and `experiments/imbalance.py` measures the busiest rank's work rising 10-20% while the summed
volume falls 24-52%. The busiest rank is a property of the batch, not of one token, so the
price is too: each tail pass charges an expert `load_penalty` times the picks already on its
rank per token, counted over a tile of `block_t` consecutive tokens, and every token in the tile
pays that same price. A rank the batch is piling onto gets dearer for all of them at once.
`block_t=1` is the per-token price, which sees only a token's own picks and so cannot see the
pile-up; on Qwen3-30B's logits at `load_penalty=1` it left the busiest rank 3-32% busier than
default vLLM at `keep <= 4`. At `block_t=16` and `load_penalty=8`, decoding 256 sequences of text
at EP 4-16, the busiest rank does 15-24% less expert work than under default vLLM at keep 6 and 4,
in at least 99.8% of calls; at keep 2 only 4-9% less, and in 60-88% of calls.
In prefill forwards of 16384 tokens of the same text it is 14-21% less at keep 6 and 4, in at least
97% of calls, and at keep 2 between 5% less and 2% more.

None of this touches traffic. At any price the tail stays on the head's ranks, so each rank
receives exactly what `keep` alone sends it, never more than under the default top-k. The counts
live in one program's registers, so the price costs no launch, no memory traffic and no
collective. It does cost kernel time, since one program runs a tile's passes in series: on an
H100 at 32-1024 tokens the call takes 6.6-8.5 us at E=128 and 10.3-12.2 us at E=256, against
3.8-5.4 us for default vLLM's top-k, and at 16384 tokens 31-37 and 67-78 us against 17 and 33.
That is also why it is a tile and not the whole batch: sharing counts across programs
takes atomics, and those make the routing depend on scheduling order. A batch whose tiles are each
balanced is balanced. It is a different selection, so unlike `keep` alone it has to be
re-measured for quality, not just for rate, and a token's experts now depend on its tile.

`select` *is* the routing, not a pass over its output: it takes the gate logits and returns
the selection and its gate weights, replacing the router's own top-k instead of following it.
One Triton launch per MoE layer, where default vLLM also spends one, so `keep == top_k` asks
for no substitution and costs nothing measurable - which is what makes that arm a control.

`rewrite_torch` is the readable reference and the specification; `_par_select` is one program
per tile of tokens with no `[T, E]` temporary, and is the one that runs under vLLM.
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

# Tokens per load-balancing tile. Big enough that one token's picks barely move the price, small
# enough that a tile's tokens do not all herd onto the same cheap rank in one pass.
LOAD_TILE = 16


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
                  n_ranks: int, load_penalty: float = 0.0, block_t: int = LOAD_TILE,
                  is_padding: torch.Tensor | None = None) -> torch.Tensor:
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
    # With a load penalty the picks stop being independent: each pass prices a rank by what the
    # whole tile has placed on it so far, so this is `k - keep` argmaxes and not one top-k.
    # Padding rows are routed and then dropped by the dispatch, so they are never counted.
    rows = torch.arange(t, device=logits.device)
    tile = rows // block_t
    real = torch.ones(t, device=logits.device) if is_padding is None else (~is_padding).float()
    n = torch.zeros(-(-t // block_t), device=logits.device).index_add_(0, tile, real).clamp_(min=1)
    inv = torch.full_like(n, load_penalty) / n                     # a true divide, as `div_rn`
    load = torch.zeros(n.numel(), n_ranks, device=logits.device)

    def place(picks):
        load.index_put_((tile[:, None].expand_as(picks), rank[picks]),
                        real[:, None].expand_as(picks), accumulate=True)

    place(head)
    sub = []
    for _ in range(k - keep):
        # `load * (penalty / n)` in float32, the kernel's exact operation order.
        pick = (z - load[tile][:, rank] * inv[tile, None]).argmax(-1)
        sub.append(pick)
        z[rows, pick] = torch.finfo(logits.dtype).min
        place(pick[:, None])
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
    def _par_select(logits_ptr, out_ptr, w_ptr, pad_ptr, scale_ptr, stride_lm, T, E,
                    K: tl.constexpr, KEEP: tl.constexpr, N_RANKS: tl.constexpr,
                    BLOCK_T: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_G: tl.constexpr,
                    HAS_PAD: tl.constexpr, HAS_SCALE: tl.constexpr,
                    LOAD_PENALTY: tl.constexpr = 0.0):
        """Top-k and the placement-aware rewrite, in one pass over a tile of `BLOCK_T` logit rows."""
        rows = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        offs = tl.arange(0, BLOCK_E)
        live = offs < E
        row_ok = rows < T
        z = tl.load(logits_ptr + rows[:, None] * stride_lm + offs[None, :],
                    mask=row_ok[:, None] & live[None, :], other=-float("inf")).to(tl.float32)
        # Masked-off lanes must never contribute a rank bit, so their shift is forced to 0.
        rank = tl.where(live, offs * N_RANKS // E, 0)
        pad = rows < 0
        if HAS_PAD:
            pad = tl.load(pad_ptr + rows, mask=row_ok, other=0) != 0
        # The rows a load is counted over: the dispatch drops padding, so it adds no work.
        real = row_ok & ~pad
        # One correctly rounded divide per program; a tail pass then prices with one multiply.
        inv = tl.math.div_rn(tl.full([], LOAD_PENALTY, tl.float32),
                             tl.maximum(tl.sum(real.to(tl.float32), axis=0), 1.0))
        ranks = tl.arange(0, BLOCK_G)

        col = tl.zeros([BLOCK_T, BLOCK_E], tl.int32)
        taken = col != 0                                               # all-false bool block
        hot = tl.zeros([BLOCK_T], tl.int32)
        # Picks the tile has placed on each rank, shared by every row in it. Counted per rank
        # and not per lane, so an update is a `[BLOCK_T, BLOCK_G]` sum and not `[BLOCK_T, BLOCK_E]`.
        # A constexpr 0.0 compiles it and its updates away, so the plain rule pays nothing for it.
        head_cnt = tl.zeros([BLOCK_T, BLOCK_G], tl.int32)
        # The head: `KEEP` argmax passes over each whole row, which is the global top-`KEEP`
        # by construction - nothing on the way in has to be sorted, or trusted to be.
        for i in range(KEEP):
            e = tl.argmax(tl.where(live[None, :] & ~taken, z, -float("inf")), axis=1)
            hit = offs[None, :] == e[:, None]
            taken = taken | hit
            col = tl.where(hit, i, col)
            re = e * N_RANKS // E
            hot = hot | (1 << re)
            if LOAD_PENALTY != 0.0:
                # Per row, so the head needs one cross-row sum after its loop and not one per pass.
                head_cnt += ((ranks[None, :] == re[:, None]) & real[:, None]).to(tl.int32)
        if LOAD_PENALTY != 0.0:
            cnt = tl.sum(head_cnt, axis=0)

        # The rest, drawn only from ranks each row's head already forces a visit to.
        on_hot = (((hot[:, None] >> rank[None, :]) & 1) == 1) & live[None, :]
        for i in range(K - KEEP):
            s = z
            if LOAD_PENALTY != 0.0:
                # One price per rank for the whole tile, priced before the gather so the multiply
                # cannot be contracted into an FMA with the subtract: the reference matches bit for bit.
                s = z - tl.gather(cnt.to(tl.float32) * inv, rank, 0)[None, :]
            e = tl.argmax(tl.where(on_hot & ~taken, s, -float("inf")), axis=1)
            hit = offs[None, :] == e[:, None]
            taken = taken | hit
            col = tl.where(hit, KEEP + i, col)
            if LOAD_PENALTY != 0.0:
                re = e * N_RANKS // E
                cnt += tl.sum(((ranks[None, :] == re[:, None]) & real[:, None]).to(tl.int32), axis=0)

        zt = tl.where(taken, z, -float("inf"))
        p = tl.exp(zt - tl.max(zt, axis=1)[:, None])
        w = p / tl.sum(p, axis=1)[:, None]
        if HAS_SCALE:
            w = w * tl.load(scale_ptr + offs, mask=live, other=1.0).to(tl.float32)[None, :]
        ids = tl.broadcast_to(offs[None, :], [BLOCK_T, BLOCK_E])
        if HAS_PAD:
            # vLLM's own sentinel: `VLLM_MOE_SKIP_PADDING` (on by default) forces a cudagraph
            # or DP padding token's expert ids to -1, and the dispatch drops the row on that
            # exact value. Routing it would add volume this arm does not owe.
            ids = tl.where(pad[:, None], -1, ids)
        out = taken & row_ok[:, None]
        tl.store(out_ptr + rows[:, None] * K + col, ids, mask=out)
        tl.store(w_ptr + rows[:, None] * K + col, w, mask=out)


def select(logits: torch.Tensor, k: int, keep: int, n_ranks: int,
           is_padding: torch.Tensor | None = None, scale: torch.Tensor | None = None,
           indices_dtype: torch.dtype | None = None, load_penalty: float = 0.0,
           block_t: int = LOAD_TILE):
    """`(gate weights, ids)`. Triton where it can run, the reference everywhere else.

    `is_padding` is vLLM's `[T]` bool mask of cudagraph/DP padding rows, `scale` an optional
    `[E]` per-expert factor on the gate weights (Gemma's, folded in here so its path costs no
    launch either). The dummy pointers for an absent one are never dereferenced - the `HAS_*`
    constexprs compile the load away. `block_t` is the load-balancing tile, a power of two; without
    a `load_penalty` there is nothing to share and each program takes one token, as before.
    """
    if triton is not None and logits.is_cuda:
        logits = logits.contiguous()
        t, e = logits.shape
        out = torch.empty(t, k, dtype=torch.int32, device=logits.device)
        w = torch.empty(t, k, dtype=torch.float32, device=logits.device)
        bt = block_t if load_penalty else 1
        # A lone tile is latency-bound and splits over 8 warps; once the grid fills the GPU, 4
        # win (H100, 32-16384 tokens). `t` is fixed per captured graph, so this is two compiles.
        warps = 8 if bt > 1 and t <= 1024 else 4
        _par_select[(triton.cdiv(t, bt),)](logits, out, w, out if is_padding is None else is_padding,
                                           w if scale is None else scale, logits.stride(0), t, e,
                                           K=k, KEEP=keep, N_RANKS=n_ranks, BLOCK_T=bt,
                                           BLOCK_E=triton.next_power_of_2(e),
                                           BLOCK_G=triton.next_power_of_2(n_ranks),
                                           HAS_PAD=is_padding is not None,
                                           HAS_SCALE=scale is not None,
                                           LOAD_PENALTY=float(load_penalty), num_warps=warps)
        # int32 in the kernel and the cast after: `topk_indices_dtype` is unsigned on some
        # backends, and -1 reinterprets there as the 4294967295 vLLM's own padding produces.
        return w, out.to(indices_dtype or torch.int32)
    ids = rewrite_torch(logits, logits.topk(k, -1).indices, keep, n_ranks, load_penalty, block_t,
                        is_padding)
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

    for keep, load_penalty, bt in ((K, 0.0, 1), (6, 0.0, 1), (4, 0.0, 1), (2, 0.0, 1), (4, 1.0, 1),
                                   (2, 0.5, 1), (6, 8.0, 16), (4, 8.0, 16), (2, 16.0, 32)):
        want = expected_fanout(keep, G, E)
        for dev in devs:
            zz, bb = z.to(dev), base.to(dev)
            gw, got = select(zz, K, keep, G, load_penalty=load_penalty, block_t=bt)
            sel = got.long().sort(-1).values
            # The fan-out assertion below is the one that matters for `load_penalty`: a penalised
            # pick is still confined to the head's ranks, so the closed form is unchanged.
            assert torch.equal(sel,
                               rewrite_torch(zz, bb, keep, G, load_penalty, bt).long().sort(-1).values)
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
            sw, sids = select(zz, K, keep, G, scale=sc, load_penalty=load_penalty, block_t=bt)
            assert torch.equal(sids, got), f"{dev} keep={keep} scale changed the selection"
            assert torch.allclose(sw, gw * sc[got.long()], atol=1e-6), f"{dev} scale fold"
        # Gemma keeps its own gate and the plugin reproduces its weights rather than running
        # it again. The reference is vLLM's `gemma4_routing_function_torch`, inlined so that a
        # change to it fails here: softmax over every expert, masked to the selection,
        # renormalized over it, then scaled. The first three are `gate_weights` - the
        # renormalization divides out the denominator the mask changed - so only the scale is
        # folded on. If this stops holding, Gemma's path has to call the router's own function.
        ids = rewrite_torch(z, base, keep, G, load_penalty, bt)
        scale = torch.rand(E) + 0.5
        probs = z.softmax(-1) * torch.nn.functional.one_hot(ids.long(), E).sum(-2)
        ref = (probs / probs.sum(-1, keepdim=True)).gather(1, ids.long()) * scale[ids.long()]
        w = gate_weights(z, ids, renormalize=True) * scale[ids.long()]
        assert torch.allclose(w, ref, atol=1e-6), "gemma scale fold"
        print(f"  keep={keep} load_penalty={load_penalty} block_t={bt} fan-out {_fanout(ids, G, E).float().mean():.3f} "
              f"(expected {want:.3f})")

    # The tile price, on a ragged last tile with padding in it. Padding is dropped by the
    # dispatch, so it must not move a real row's choice: flipping its logits must change nothing.
    pad = torch.zeros(100, dtype=torch.bool)
    pad[::3] = True
    zp = z[:100].clone()
    ref = rewrite_torch(zp, zp.topk(K, -1).indices, 4, G, 8.0, 16, pad)
    for dev in devs:
        for zq in (zp, torch.where(pad[:, None], -zp, zp)):
            _, got = select(zq.to(dev), K, 4, G, is_padding=pad.to(dev), load_penalty=8.0, block_t=16)
            got = got.cpu().long()
            assert bool((got[pad] == -1).all()), f"{dev} padded rows must be the sentinel"
            assert torch.equal(got[~pad].sort(-1).values, ref[~pad].long().sort(-1).values), \
                f"{dev} padding moved a real row's tail"

    # The purpose. The batch's busiest rank, counted in expert assignments, must end up below
    # default vLLM's, on logits where some experts are popular across the whole batch, which is
    # where piling the tail onto the head's ranks hurts. Volume is the fan-out check above.
    zs = z + 2 * torch.randn(E)
    busiest = lambda ids: torch.bincount((ids.long() * G // E).flatten(), minlength=G).max().item()
    for keep in (6, 4, 2):
        top = busiest(zs.topk(K, -1).indices)
        per_token = busiest(select(zs, K, keep, G, load_penalty=1.0, block_t=1)[1])
        tiled = busiest(select(zs, K, keep, G, load_penalty=8.0, block_t=16)[1])
        assert tiled < top, f"keep={keep}: busiest rank {tiled} vs default vLLM's {top}"
        print(f"  keep={keep} busiest rank: default vLLM {top}, per-token price {per_token}, "
              f"tile price {tiled}")

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
            # By expert id, not by column: which column an expert lands in is the router's own
            # business and changes nothing downstream, the id-to-weight map does.
            o, vo = got.argsort(-1), vi.argsort(-1)
            assert torch.equal(got.gather(1, o), vi.gather(1, vo)), "keep == top_k is not default vLLM"
            assert torch.allclose(w.gather(1, o), vw.gather(1, vo), atol=1e-6), "default vLLM weights"
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
