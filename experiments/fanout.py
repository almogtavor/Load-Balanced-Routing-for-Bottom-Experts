"""Where the fan-out saving lands, and whether anything on the critical path moves with it.

The rule shrinks the number of distinct ranks a token is dispatched to. Three separate
questions decide whether that becomes throughput, and this answers all three offline, because
each one is a property of the selection alone:

  volume  total bytes on the wire, proportional to summed per-token distinct ranks
  tier    DeepEP charges NVLink per distinct *rank* and RDMA per distinct *node*, and counts
          them separately (`num_tokens_per_rank` vs `num_tokens_per_rdma_rank`). A saving on
          the intra-node tier is a saving on the wire that was never the constraint.
  step    the all-to-all is one collective per layer per step over the whole in-flight batch.
          `union` is how many ranks take part at all - a collective waits for every one of
          them - and `max` is the busiest rank's share, which is what the others wait on.
  max     both the collective and the expert GEMM behind it end when the slowest rank ends, so
          the summed volume is not what sets the step. Two maxima matter and they move in
          opposite directions: the busiest rank's *dispatch* count falls with the saving, while
          its *expert work* rises, because the substitute is the highest logit on a rank the
          head already visits and the head visits popular ranks. Total work is `B * K` in every
          arm, so a rise in that maximum is pure imbalance on the critical path. The kernel's
          `alpha` charges a tail pick per slot already on its rank, which flattens that
          maximum at the same volume - the last column.

Logits here are iid Gaussian, with an optional skew to make expert choice non-uniform. This
tests the *mechanism*, not the magnitude: a trained router's logits are correlated across
tokens and layers in ways no synthetic draw reproduces.

    python3 experiments/fanout.py
"""
import numpy as np

E, K, T = 128, 8, 8192


def sel_rank(lg, k, keep, g, alpha=0.0):
    """The shipped rule: collapse the tail onto ranks the head already visits.

    `alpha` is the kernel's load penalty: charge a tail pick that much per slot already placed
    on its rank. Same hot-rank restriction, so the same volume - it only changes which of those
    ranks the tail lands on. With a penalty the picks stop being independent, so this is
    `k - keep` argmaxes rather than one top-k, which is also what the kernel does.
    """
    b = lg.shape[0]
    rank = np.arange(E) * g // E
    o = np.argsort(-lg, 1)
    head = o[:, :keep]
    hot = np.zeros((b, g), bool)
    np.put_along_axis(hot, head * g // E, True, 1)
    z = np.where(hot[:, rank], lg, -np.inf).copy()
    np.put_along_axis(z, head, -np.inf, 1)
    if not alpha or k <= keep:
        return np.concatenate([head, np.argsort(-z, 1)[:, :k - keep]], 1)
    load = np.zeros((b, g))
    np.add.at(load, (np.arange(b)[:, None], head * g // E), 1.0)
    sub = []
    for _ in range(k - keep):
        e = (z - alpha * load[:, rank]).argmax(1)
        sub.append(e)
        z[np.arange(b), e] = -np.inf
        load[np.arange(b), rank[e]] += 1
    return np.concatenate([head, np.stack(sub, 1)], 1)


def cost(sel, g, gpn):
    """(summed distinct ranks, summed distinct nodes) over tokens."""
    vis = np.zeros((sel.shape[0], g), bool)
    np.put_along_axis(vis, sel * g // E, True, 1)
    nod = np.zeros((sel.shape[0], g // gpn), bool)
    np.put_along_axis(nod, sel * g // E // gpn, True, 1)
    return vis.sum(), nod.sum()


def step(rng, g, arms, B, skew, gpn=1, steps=400):
    """Per-step volume, participation and busiest-destination load, one tier at a time.

    `arms` is a list of `(keep, alpha)`, all scored on the *same* logit draw - the arms differ
    by a few percent and independent draws at this `steps` do not resolve that.

    `gpn=1` buckets by rank, which is the NVLink tier. Any larger `gpn` buckets ranks into
    `g // gpn` nodes, which is the RDMA tier - the one that is charged separately and is two
    orders of magnitude slower.
    """
    n = g // gpn
    acc = np.zeros((len(arms), 4))
    for _ in range(steps):
        lg = rng.standard_normal((B, E)).astype(np.float32)
        if skew:
            lg += skew * np.log(rng.gamma(0.4, 1.0, E) + 1e-6)
        for i, (keep, alpha) in enumerate(arms):
            dst = sel_rank(lg, K, keep, g, alpha) * g // E // gpn
            vis = np.zeros((B, n), bool)
            np.put_along_axis(vis, dst, True, 1)
            # The last one is expert slots landing on the busiest destination. Summed over
            # destinations that is `B * K` whatever `keep` is, so only its spread can change.
            acc[i] += (vis.sum(), vis.any(0).sum(), vis.sum(0).max(),
                       np.bincount(dst.ravel(), minlength=n).max())
    return acc / steps


def main():
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((T, E)).astype(np.float32)

    # keep == k asks for no substitution, so it must reproduce plain top-k exactly. This is the
    # same invariant the GPU self-check asserts, and it is what makes that arm a control.
    base = np.sort(np.argsort(-lg, 1)[:, :K], 1)
    assert (np.sort(sel_rank(lg, K, K, 4), 1) == base).all()
    assert (np.sort(sel_rank(lg, K, K, 4, alpha=1.0), 1) == base).all()

    print("Which tier the saving lands on (ep_size, gpus/node):")
    for g, gpn in ((4, 2), (8, 2), (8, 1), (16, 2)):
        br, bn = cost(np.argsort(-lg, 1)[:, :K], g, gpn)
        print(f"  ep={g} gpn={gpn} ({g // gpn} nodes)  base ranks {br} nodes {bn}")
        for keep in (6, 4, 2):
            r, n = cost(sel_rank(lg, K, keep, g), g, gpn)
            print(f"    keep={keep}  ranks (NVLink) {(r / br - 1) * 100:+6.1f}%"
                  f"   nodes (RDMA) {(n / bn - 1) * 100:+6.1f}%")

    # Participation is the one that decides whether anything waits less. The all-to-all is one
    # collective per layer per step over the whole in-flight batch, so if the union of
    # destinations is already everything, a smaller payload is all the rule can buy.
    print("\nPer step, keep=8 (control) vs keep=4:")
    for skew, lab in ((0.0, "iid"), (1.0, "skewed")):
        for g, gpn, B in ((4, 1, 16), (4, 1, 256), (8, 1, 16), (8, 1, 256),
                          (8, 2, 256), (16, 2, 256), (16, 1, 256), (32, 2, 256)):
            (v8, u8, m8, w8), (v4, u4, m4, w4), (va, _, ma, wa) = step(
                rng, g, [(8, 0.0), (4, 0.0), (4, 1.0)], B, skew, gpn)
            # `alpha` keeps the tail on the head's ranks, so it moves neither the volume nor
            # the busiest rank's dispatch count - only which of those ranks the slots land on.
            assert va == v4 and ma == m4, f"alpha moved volume {va} vs {v4}, max {ma} vs {m4}"
            dst = "ranks" if gpn == 1 else f"{g // gpn} nodes"
            print(f"  {lab:6} ep={g:<2} B={B:<4} over {dst:<8}"
                  f" volume {(v4 / v8 - 1) * 100:+6.1f}%"
                  f"   taking part {u8:5.2f} -> {u4:5.2f}"
                  f"   busiest dispatch {(m4 / m8 - 1) * 100:+6.1f}%"
                  f"   busiest work {(w4 / w8 - 1) * 100:+6.1f}%"
                  f"   with alpha=1 {(wa / w8 - 1) * 100:+6.1f}%")


if __name__ == "__main__":
    main()
