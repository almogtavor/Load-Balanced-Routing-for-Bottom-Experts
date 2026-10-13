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

Logits here are iid Gaussian, with an optional skew to make expert choice non-uniform. This
tests the *mechanism*, not the magnitude: a trained router's logits are correlated across
tokens and layers in ways no synthetic draw reproduces.

    python3 experiments/fanout.py
"""
import numpy as np

E, K, T = 128, 8, 8192


def sel_rank(lg, k, keep, g):
    """The shipped rule: collapse the tail onto ranks the head already visits."""
    o = np.argsort(-lg, 1)
    head = o[:, :keep]
    hot = np.zeros((lg.shape[0], g), bool)
    np.put_along_axis(hot, head * g // E, True, 1)
    z = np.where(hot[:, np.arange(E) * g // E], lg, -np.inf).copy()
    np.put_along_axis(z, head, -np.inf, 1)
    return np.concatenate([head, np.argsort(-z, 1)[:, :k - keep]], 1)


def cost(sel, g, gpn):
    """(summed distinct ranks, summed distinct nodes) over tokens."""
    vis = np.zeros((sel.shape[0], g), bool)
    np.put_along_axis(vis, sel * g // E, True, 1)
    nod = np.zeros((sel.shape[0], g // gpn), bool)
    np.put_along_axis(nod, sel * g // E // gpn, True, 1)
    return vis.sum(), nod.sum()


def step(rng, g, keep, B, skew, steps=400):
    """Per-step volume, rank participation and busiest-rank load."""
    vol = uni = mx = 0.0
    for _ in range(steps):
        lg = rng.standard_normal((B, E)).astype(np.float32)
        if skew:
            lg += skew * np.log(rng.gamma(0.4, 1.0, E) + 1e-6)
        vis = np.zeros((B, g), bool)
        np.put_along_axis(vis, sel_rank(lg, K, keep, g) * g // E, True, 1)
        vol += vis.sum()
        uni += vis.any(0).sum()
        mx += vis.sum(0).max()
    return vol / steps, uni / steps, mx / steps


def main():
    rng = np.random.default_rng(0)
    lg = rng.standard_normal((T, E)).astype(np.float32)

    # keep == k asks for no substitution, so it must reproduce plain top-k exactly. This is the
    # same invariant the GPU self-check asserts, and it is what makes that arm a control.
    base = np.sort(np.argsort(-lg, 1)[:, :K], 1)
    assert (np.sort(sel_rank(lg, K, K, 4), 1) == base).all()

    print("Which tier the saving lands on (ep_size, gpus/node):")
    for g, gpn in ((4, 2), (8, 2), (8, 1), (16, 2)):
        br, bn = cost(np.argsort(-lg, 1)[:, :K], g, gpn)
        print(f"  ep={g} gpn={gpn} ({g // gpn} nodes)  base ranks {br} nodes {bn}")
        for keep in (6, 4, 2):
            r, n = cost(sel_rank(lg, K, keep, g), g, gpn)
            print(f"    keep={keep}  ranks (NVLink) {(r / br - 1) * 100:+6.1f}%"
                  f"   nodes (RDMA) {(n / bn - 1) * 100:+6.1f}%")

    print("\nPer step, keep=8 (control) vs keep=4:")
    for skew, lab in ((0.0, "iid"), (1.0, "skewed")):
        for g in (4, 8):
            for B in (16, 256):
                v8, u8, m8 = step(rng, g, 8, B, skew)
                v4, u4, m4 = step(rng, g, 4, B, skew)
                print(f"  {lab:6} ep={g} B={B:<4} volume {(v4 / v8 - 1) * 100:+6.1f}%"
                      f"   ranks taking part {u8:.2f} -> {u4:.2f}"
                      f"   busiest rank {(m4 / m8 - 1) * 100:+6.1f}%")


if __name__ == "__main__":
    main()
