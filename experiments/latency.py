"""Phase B: the measured time of one expert-parallel MoE layer on four real GPUs.

    torchrun --nproc_per_node=4 latency.py

Phase A recorded, for every arm, which experts each token was routed to. This replays those
exact selections through a real expert-parallel MoE layer - experts sharded in contiguous
blocks across the four GPUs, dispatch and combine over NCCL `all_to_all_single` - and times
it. Nothing is modelled: the fan-out reduction is an analytic quantity, but whether it turns
into wall clock depends on the batch size, the interconnect and the comm/compute ratio, which
is exactly what a count cannot tell you.

**Why the layer and not just the collective.** Re-routing the bottom experts leaves expert
GEMM work untouched - still `top_k` experts per token - so it can only shrink communication.
Reporting the collective alone would quote a number no deployment ever sees; halving a
communication that is 40% of the layer is a 20% layer speedup, and the honest headline is the
second one. Both are recorded, so the comm fraction is visible rather than asserted.

**Why the batch sweep is the result, not a parameter.** A token's hidden state is a few KB, so
at a small batch the dispatch is latency-bound and moving half the bytes buys exactly nothing;
the saving only appears once the collective is bandwidth-bound. Reporting a single batch would
be picking which of those two regimes to tell the reader about, so every batch is recorded and
the figure draws the largest - the bandwidth-bound one, which is also the prefill regime a
deployment spends its time in - with its batch size printed on the axis.

**Why the capacity is a fixed factor.** The local experts run as one `bmm` over an
`[n_loc, cap, H]` bucket buffer, so `cap` *is* this layer's compute. Sizing it per arm to that
arm's own hottest expert looks more faithful and is in fact the one choice that makes the arms
incomparable: re-routing concentrates tokens onto the experts of already-hot GPUs, so the
placement arms would be charged padding the baseline does not pay, while the random control -
which spreads tokens uniformly and buys *no* fan-out at all - would be handed a large fake
speedup for shrinking its own padding. Measured, that artifact was +68% on an arm whose fan-out
did not move. So `cap` is a capacity factor over the mean load, identical for every arm and
every rank, which is both what a GShard-style deployment runs and what makes the compute term
cancel out of the comparison: what is left between two arms is the communication. Tokens past
capacity are dropped, as they are in such a deployment; the fraction is reported, and no
accuracy claim rests on this script - `accuracy.py` scores the drop-free layer.

Expert weights are random tensors of the shapes in the checkpoint's config: timing depends on
shapes and on the routing pattern, not on weight values, and the same weights serve every arm.
The dispatch *plan* (see `plan`) is built outside the timed region, and the first arm of each
batch is timed once and thrown away, because it pays the NCCL channel setup and the cuBLAS
autotune for these shapes and would otherwise bank them as everyone else's speedup. The
`noop` arm - identical routing to the baseline - is the check that this worked: whatever
speedup it still shows is the harness's own resolution, and it is asserted small.

One deliberate ceiling, biasing *against* the result: no computation is overlapped with the
collectives, which a production dispatch would do.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import statistics

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import placement as P

TAG = os.environ.get("TAG", "")
BATCHES = tuple(int(x) for x in os.environ.get("LBATCH", "1024,4096,16384").split(","))
PLOT_BATCH = int(os.environ.get("PLOT_BATCH", max(BATCHES)))
# Capacity per expert as a multiple of the mean load `B k / E`. 1.25 is the usual setting;
# what matters here is that it does not depend on the arm.
CAP_FACTOR = float(os.environ.get("CAP_FACTOR", 1.25))
WARMUP = int(os.environ.get("WARMUP", 10))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 50))
RUNS = pathlib.Path("runs")


def plan(idx_all: torch.Tensor, rank: int, world: int, gpu: torch.Tensor, n_loc: int,
         cap: int | None = None) -> dict:
    """Everything about one step's dispatch that follows from the routing alone.

    `idx_all` is `[B, k]` global expert ids for the whole batch; rows `rank * B/world ...` are
    this rank's own tokens. Returns the send order and the per-rank row counts, and - for the
    receive side - the map from a received row to a slot of the expert-capacity buffer.

`cap` is the per-expert capacity; `None` sizes it to the hottest expert, which drops
    nothing and is what the self-checks compare against the dense layer.

    This sits outside the timed region on purpose. A real dispatcher builds the same plan, but
    on the device and overlapped with the previous layer, whereas building it inside the timed
    step costs a device-to-host sync per step for the exchange counts. That sync is the same
    for every arm, so leaving it in would have charged all of them an identical Python cost
    and buried the only thing that differs between them. The routing metadata is still *sent*
    (see `ep_layer`), so the bytes an arm moves are the bytes it would move in production.
    """
    B, k = idx_all.shape
    loc = B // world
    need = torch.zeros(B, world, dtype=torch.bool)
    need.scatter_(1, gpu[idx_all], torch.ones(B, k, dtype=torch.bool))

    # Send side: rows grouped by destination rank. The send buffer's layout and the combine's
    # scatter-back index are the same list, which is why a token visiting three GPUs gets
    # three partial sums added to the same output row.
    nd = need[rank * loc : (rank + 1) * loc]
    rows = torch.cat([nd[:, h].nonzero(as_tuple=True)[0] for h in range(world)])
    sl = nd.sum(0).tolist()

    # Receive side. Every rank holds the whole trace, so what will arrive - and in which order
    # - is known exactly: sender `r`'s rows that need this rank, in `r`'s own order.
    src = [need[r * loc : (r + 1) * loc, rank].nonzero(as_tuple=True)[0] + r * loc
           for r in range(world)]
    rl = [len(s) for s in src]
    ridx = idx_all[torch.cat(src)] if rl else idx_all[:0]           # [m, k] in receive order
    j, slot = ((ridx >= rank * n_loc) & (ridx < (rank + 1) * n_loc)).nonzero(as_tuple=True)
    e = ridx[j, slot] - rank * n_loc
    # Stable sort by local expert, so each expert's rows are contiguous and a row's position
    # inside its expert's bucket is its offset from that expert's start.
    o = torch.argsort(e, stable=True)
    j, slot, e = j[o], slot[o], e[o]
    cnt = torch.bincount(e, minlength=n_loc)
    cap = max(int(cnt.max()), 1) if cap is None else cap
    off = torch.arange(len(e)) - (cnt.cumsum(0) - cnt)[e]
    # Rows past an expert's capacity are dropped. The bucket buffer is `n_loc * cap` either
    # way, so dropping changes no timing - it is what keeps the padding, and therefore the
    # compute, from depending on how skewed this arm's load happens to be.
    hit, n = off < cap, len(e)
    j, slot, e, off = j[hit], slot[hit], e[hit], off[hit]
    return {"rows": rows, "sl": sl, "rl": rl, "m": sum(rl), "cap": cap,
            "j": j, "slot": slot, "pos": e * cap + off,
            "drop": 1 - len(e) / max(n, 1),
            # Occupancy, not a cost that differs between arms: `n_loc * cap` is fixed here.
            "pad": n_loc * cap / max(len(e), 1)}


def on(pl: dict, dev) -> dict:
    """The plan's index tensors on `dev`. The counts stay host-side; the collective wants ints."""
    return {k: v.to(dev) if torch.is_tensor(v) else v for k, v in pl.items()}


def ep_layer(x, meta, pl, w13, w2, shared=None, comm_only=False):
    """One expert-parallel MoE layer over this rank's token slice.

    `x` is `[loc, H]` local tokens and `meta` is `[loc, 2k]` the routing metadata that travels
    with them. A token is sent once per *destination GPU*, not once per expert - that dedup is
    the whole point of the fan-out metric, and it is what a real dispatch (DeepEP and friends)
    does too. Hidden state and metadata go in one message rather than three, again as a real
    dispatch does: the metadata is a percent of the bytes, so three collectives would have
    measured launch latency three times over.

    `shared`, when the family has always-on experts, is their weights replicated on every rank
    and applied to the local tokens before dispatch - which is where a real deployment runs
    them, since a replicated expert needs no communication.
    """
    H, k = x.shape[1], meta.shape[1] // 2
    n_loc, cap = w13.shape[0], pl["cap"]
    send = torch.cat([x[pl["rows"]], meta[pl["rows"]]], 1)
    recv = torch.empty(pl["m"], send.shape[1], dtype=send.dtype, device=x.device)
    dist.all_to_all_single(recv, send, pl["rl"], pl["sl"])
    rx, rw = recv[:, :H], recv[:, H + k :]

    y = torch.zeros_like(rx)
    if not comm_only:
        # One `bmm` over capacity-padded buckets rather than a loop over local experts: the
        # loop needs `if this expert got any rows`, which is a device sync per expert, and at
        # 32 experts per GPU that dominated everything this file is trying to measure.
        buck = torch.zeros(n_loc * cap, H, dtype=x.dtype, device=x.device)  # drop-free
        buck.index_copy_(0, pl["pos"], rx[pl["j"]])
        a, b = torch.bmm(buck.view(n_loc, cap, H), w13).chunk(2, -1)
        g = torch.bmm(F.silu(a) * b, w2).reshape(n_loc * cap, H)
        y.index_add_(0, pl["j"], g[pl["pos"]] * rw[pl["j"], pl["slot"]].unsqueeze(-1))

    back = torch.empty(len(pl["rows"]), H, dtype=y.dtype, device=x.device)
    dist.all_to_all_single(back, y, pl["sl"], pl["rl"])
    out = torch.zeros_like(x)
    out.index_add_(0, pl["rows"], back)
    if shared is not None and not comm_only:
        a, b = (x @ shared[0]).chunk(2, -1)
        out += (F.silu(a) * b) @ shared[1]
    return out


def dense_layer(x, idx, wt, w13, w2):
    """The same layer computed without any sharding. The reference for the self-check."""
    out = torch.zeros_like(x)
    for e in range(w13.shape[0]):
        hit = idx == e
        sel = hit.any(1)
        if not bool(sel.any()):
            continue
        a, b = (x[sel] @ w13[e]).chunk(2, -1)
        out[sel] += ((F.silu(a) * b) @ w2[e]) * wt[sel][hit[sel]].unsqueeze(-1)
    return out


def _toy(world: int, seed: int = 0):
    """A small exact case: `(x, idx, wt, w13, w2)` for `world * n` tokens over `E` experts."""
    E, H, I, k, n = 8, 16, 8, 3, 12
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(world * n, H, generator=g),
            torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(world * n)]),
            torch.rand(world * n, k, generator=g),
            torch.randn(E, H, 2 * I, generator=g), torch.randn(E, I, H, generator=g))


def self_check(rank, world, dev):
    """The sharded layer must equal the unsharded one, or the timings are of the wrong thing.

    Run small and in fp32: the failures this guards against (a send buffer ordered differently
    from the scatter-back index, a bucket slot assigned to the wrong expert, a combine that
    loses one of a token's partial sums) are all exact, so a tight tolerance is what detects
    them - at bf16 they would hide inside the accumulation error.
    """
    x, idx, wt, w13, w2 = _toy(world)
    n_loc, n = w13.shape[0] // world, x.shape[0] // world
    s = slice(rank * n, (rank + 1) * n)
    pl = on(plan(idx, rank, world, P.gpu_of(w13.shape[0], world), n_loc), dev)
    meta = torch.cat([idx.float(), wt], 1)[s].to(dev)
    got = ep_layer(x[s].to(dev), meta, pl, w13[rank * n_loc : (rank + 1) * n_loc].to(dev),
                   w2[rank * n_loc : (rank + 1) * n_loc].to(dev))
    want = dense_layer(x[s], idx[s], wt[s], w13, w2).to(dev)
    err = (got - want).abs().max().item() / want.abs().max().item()
    assert err < 1e-5, f"rank {rank}: expert-parallel layer differs from dense by {err:.2e}"
    if rank == 0:
        print(f"self-check ok (max relative error {err:.1e})", flush=True)


def time_steps(plans, x, meta, w13, w2, shared, comm_only):
    """Median per-step layer time in ms, taken as the slowest rank each step."""
    beg, end = torch.cuda.Event(True), torch.cuda.Event(True)
    out = []
    for i, pl in enumerate(plans):
        if i >= WARMUP:
            dist.barrier()
            beg.record()
        ep_layer(x, meta, pl, w13, w2, shared, comm_only)
        if i >= WARMUP:
            end.record()
            torch.cuda.synchronize()
            t = torch.tensor([beg.elapsed_time(end)], device=x.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            out.append(t.item())
    return statistics.median(out)


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    self_check(rank, world, dev)

    acc = json.loads((RUNS / f"acc_{TAG}.json").read_text())
    E, k, H, I = acc["n_experts"], acc["top_k"], acc["hidden"], acc["moe_inter"]
    S = acc["shared_inter"]
    assert acc["n_gpus"] == world, f"trace is for {acc['n_gpus']} GPUs, running on {world}"
    npz = np.load(RUNS / f"trace_{TAG}.npz")
    n_loc = E // world
    lo_e = rank * n_loc
    w13 = torch.randn(n_loc, H, 2 * I, dtype=torch.bfloat16, device=dev)
    w2 = torch.randn(n_loc, I, H, dtype=torch.bfloat16, device=dev)
    shared = None if S == 0 else (
        torch.randn(H, 2 * S, dtype=torch.bfloat16, device=dev),
        torch.randn(S, H, dtype=torch.bfloat16, device=dev))
    if rank == 0:
        print(f"{TAG}: E={E} k={k} H={H} I={I} shared={S} -> {n_loc} experts/GPU", flush=True)

    gpu_cpu = P.gpu_of(E, world)
    res: dict[str, dict] = {}
    for B in BATCHES:
        assert B % world == 0
        loc = B // world
        cap = max(1, math.ceil(CAP_FACTOR * B * k / E))
        x = torch.randn(loc, H, dtype=torch.bfloat16, device=dev)
        meta = torch.randn(loc, 2 * k, dtype=torch.bfloat16, device=dev)
        res[str(B)] = {}
        for arm in npz.files:
            tr = npz[arm]                                      # [layers, tokens, k]
            offs = range(0, tr.shape[1] - B + 1, B)
            steps = [(l, o) for l in range(tr.shape[0]) for o in offs][: MAX_STEPS + WARMUP]
            assert len(steps) > WARMUP, f"trace of {tr.shape[1]} tokens too short for batch {B}"
            plans = [on(plan(torch.from_numpy(tr[l, o : o + B].astype(np.int64)),
                             rank, world, gpu_cpu, n_loc, cap), dev) for l, o in steps]
            if not res[str(B)]:
                # The first arm of a batch pays the NCCL channel setup, the allocator growth
                # and the cuBLAS autotune for these shapes. Timed and discarded, or it banks
                # them as every later arm's speedup: measured, that was +394% on `noop`.
                time_steps(plans, x, meta, w13, w2, shared, False)
                time_steps(plans, x, meta, w13, w2, shared, True)
            ms = time_steps(plans, x, meta, w13, w2, shared, False)
            cm = time_steps(plans, x, meta, w13, w2, shared, True)
            fan = P.fanout(torch.from_numpy(tr.reshape(-1, k).astype(np.int64)),
                           gpu_cpu, world).float().mean().item()
            pad = statistics.mean(p["pad"] for p in plans)
            drop = statistics.mean(p["drop"] for p in plans)
            del plans
            res[str(B)][arm] = {"ms": ms, "ms_disp": cm, "fanout": fan, "pad": pad,
                                "drop": drop}
            if rank == 0:
                b = res[str(B)].get("baseline")
                # `dispatch` is the layer with the expert GEMMs skipped and everything else
                # - both collectives, the gather, the scatter-back - left in place, so it is
                # an upper bound on the communication rather than the collective alone.
                print(f"B={B:6d} {arm:14s} {ms:7.3f} ms  dispatch {cm:6.3f} ms "
                      f"({cm / ms:.0%})  fan-out {fan:.3f}  pad {pad:.2f}x drop {drop:.1%}"
                      + (f"  speedup {b['ms'] / ms - 1:+.1%}" if b else ""), flush=True)

    if rank == 0:
        (RUNS / f"lat_{TAG}.json").write_text(json.dumps({
            "tag": TAG, "n_gpus": world, "hidden": H, "moe_inter": I, "shared_inter": S,
            "n_experts": E, "top_k": k, "plot_batch": PLOT_BATCH, "cap_factor": CAP_FACTOR,
            "gpu": torch.cuda.get_device_name(0), "batches": res,
        }, indent=1) + "\n")
        print(f"WROTE {RUNS}/lat_{TAG}.json", flush=True)
    dist.destroy_process_group()

    # The no-op arm routes exactly as the baseline does, so whatever separates their timings is
    # this harness and nothing else: it bounds every speedup reported here, and it is a hard
    # stop rather than a note. Checked after the write, so a run that fails it leaves the
    # numbers that diagnose it on disk.
    for B, b in res.items():
        up = b["baseline"]["ms"] / b["noop"]["ms"] - 1
        assert abs(up) < 0.03, f"B={B}: no-op arm timed {up:+.1%} off the baseline"


def _check() -> None:
    """`plan` and the bucket math, against the dense layer, with the collectives emulated.

    The distributed `self_check` can only run inside the job; this one runs anywhere and
    covers the half that is actually easy to get wrong - that each rank's receive-side map
    describes exactly what the other ranks will send it, in the order they will send it.
    """
    for world in (1, 2, 4):
        x, idx, wt, w13, w2 = _toy(world)
        E, n = w13.shape[0], x.shape[0] // world
        n_loc, gpu = E // world, P.gpu_of(E, world)
        pls = [plan(idx, r, world, gpu, n_loc) for r in range(world)]
        # What rank r sends to rank h, in r's own layout: the slice of its `rows` for h.
        send = [[pls[r]["rows"].split(pls[r]["sl"])[h] + r * n for h in range(world)]
                for r in range(world)]
        for r in range(world):
            assert pls[r]["rl"] == [len(send[s][r]) for s in range(world)], f"counts {r}"
            got_rows = torch.cat([send[s][r] for s in range(world)])
            pl, lo = pls[r], r * n_loc
            # Re-derive the receive buffer the way `ep_layer` consumes it, then run the local
            # experts through the bucket map and check one rank's contribution to each token.
            buck = torch.zeros(n_loc * pl["cap"], x.shape[1])
            buck.index_copy_(0, pl["pos"], x[got_rows][pl["j"]])
            a, b = torch.bmm(buck.view(n_loc, pl["cap"], -1), w13[lo : lo + n_loc]).chunk(2, -1)
            g = torch.bmm(F.silu(a) * b, w2[lo : lo + n_loc]).reshape(n_loc * pl["cap"], -1)
            y = torch.zeros(len(got_rows), x.shape[1])
            rw = wt[got_rows]
            y.index_add_(0, pl["j"], g[pl["pos"]] * rw[pl["j"], pl["slot"]].unsqueeze(-1))
            # The reference for one shard is the dense layer restricted to its experts.
            mask = (idx[got_rows] >= lo) & (idx[got_rows] < lo + n_loc)
            want = dense_layer(x[got_rows], torch.where(mask, idx[got_rows] - lo, -1),
                               torch.where(mask, wt[got_rows], torch.zeros(())),
                               w13[lo : lo + n_loc], w2[lo : lo + n_loc])
            err = (y - want).abs().max().item() / want.abs().max().item()
            assert err < 1e-5, f"world={world} rank={r}: bucket math off by {err:.2e}"
        print(f"  world={world} plan ok (pad {statistics.mean(p['pad'] for p in pls):.2f}x)")
    print("latency.py: ok")


if __name__ == "__main__":
    if os.environ.get("CHECK"):
        _check()
    else:
        main()
