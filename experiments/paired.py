"""The serve throughput table. A launch is the unit of analysis, not a repeat.

An unmodified engine, loaded twice on the same node for the same workload, measured +5.4% and
+5.9% against itself. So a keep arm is only ever divided by the mean of the base arms *from its
own launch*, repeat 0 is discarded as cold upstream, and the within-launch decay of the base
arms is printed beside every row as the floor on what that row can resolve.

    python3 experiments/paired.py                      # decode rate, runs/serve_*.json
    METRIC=tot_tok_per_s python3 experiments/paired.py  # prefill+decode rate
    METRIC=ib_b_per_tok python3 experiments/paired.py   # IB bytes per token, lower is better
"""
import collections
import glob
import json
import os
import re
import statistics as st

# Artefacts written before the padding-sentinel fix landed are not comparable: the bug charged
# dispatch and expert GEMM for vLLM's padded rows on the rule's side of the comparison only.
# `written` is an ISO timestamp, so this is a string compare.
CUT = "2026-10-13T10:40"

def arm_key(a):
    """`4` is keep=4, `4p0.5` is keep=4 with the load penalty. Descending keep, ascending load_penalty."""
    k, _, al = a.partition("p")
    return -int(k), float(al or 0.0)


def arm_label(a: str) -> str:
    """Axis label for an arm. `8` is the control: the kernel runs and selects nothing different."""
    k, _, pen = a.partition("p")
    if k == "8":
        return "keep=8\n(control: no\nsubstitution)"
    return f"keep={k}" + ("\n+load balance" if pen and float(pen) else "")


# The rule moves communication volume, which is `ib_b_per_tok` directly and
# `tot_tok_per_s` at a prefill-heavy shape. On the bytes metric a negative row is the
# rule working; on the rate metrics a positive one is.
METRIC = os.environ.get("METRIC") or "out_tok_per_s"


def load(pat="runs/serve_*.json"):
    """`(tag, rep)` -> arm -> record, keeping only the shape a tag was last swept at.

    A tag accumulates artefacts across sweeps, and an arm from an earlier sweep at a smaller
    step budget runs several times faster - mixing one into a launch would move a baseline by
    more than the effect. So a record whose shape is not the tag's newest is dropped, not
    paired."""
    runs = collections.defaultdict(dict)       # (tag, rep) -> arm -> record
    recs, newest = [], {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        if d.get("written", "") < CUT:
            continue
        m = re.match(r"serve_(.+)_([^_]+)_r(\d+)\.json$", os.path.basename(f))
        if not re.fullmatch(r"base\d*|\d+(p[\d.]+)?", m.group(2)):
            continue       # an arm from the pre-rename sweeps, e.g. `4a1`
        shape = (d.get("prompt_len"), d.get("batch_tok"), d.get("n_req"))
        recs.append((m.group(1), int(m.group(3)), m.group(2), shape, d))
        if d["written"] > newest.get((m.group(1), shape), ("",))[0]:
            newest[(m.group(1), shape)] = (d["written"], shape)
    # Per tag, the shape of its most recently written artefact.
    last = {}
    for (tag, shape), (w, _) in newest.items():
        if w > last.get(tag, ("",))[0]:
            last[tag] = (w, shape)
    skipped = 0
    for tag, rep, arm, shape, d in recs:
        if shape != last[tag][1]:
            skipped += 1
            continue
        runs[(tag, rep)][arm] = d
    return runs, skipped


def decay(d):
    """First warm repeat to last, as a percentage. On a base arm this is instrument drift."""
    w = d["warm_tok_per_s"]
    return (w[0] - w[-1]) / w[0] * 100 if len(w) > 1 else float("nan")


def main():
    runs, skipped = load()
    # Read the arms off the artefacts rather than listing them here, so an arm added to a
    # sweep shows up in the table without a second edit.
    keeps = sorted({a for arms in runs.values() for a in arms if not a.startswith("base")},
                   key=arm_key)
    print(f"excluded {skipped} artefacts: pre-fix ({CUT}) or an older shape")
    print(f"metric: {METRIC}")
    per_tag = collections.defaultdict(lambda: collections.defaultdict(list))
    print(f"{'tag':8} {'btok':>5} {'rep':>3} {'nb':>2} {'base':>8} {'spread%':>7} "
          + " ".join(f"{'k=' + k:>8}" for k in keeps) + "  decay%")
    for (tag, rep), arms in sorted(runs.items()):
        bases = [v[METRIC] for a, v in arms.items()
                 if a.startswith("base") and v.get(METRIC) is not None]
        if not bases:
            continue
        bm = st.mean(bases)
        # The scheduler's per-step token budget is the dispatch payload per all-to-all, so a
        # tag measured at two budgets is two different experiments. It keys the aggregate
        # rather than being averaged into it. Artefacts from before the knob existed ran at
        # vLLM's batched-DP default of 256.
        bts = {v.get("batch_tok", 256) for v in arms.values()}
        assert len(bts) == 1, f"{tag} r{rep} mixes token budgets {bts}"
        bt = bts.pop()
        row = []
        for k in keeps:
            if arms.get(k, {}).get(METRIC) is None:
                row.append(f"{'-':>8}")
                continue
            r = (arms[k][METRIC] / bm - 1) * 100
            per_tag[(tag, bt)][k].append(r)
            row.append(f"{r:+8.2f}")
        dec = st.mean([decay(v) for a, v in arms.items() if a.startswith("base")])
        print(f"{tag:8} {bt:>5} {rep:>3} {len(bases):>2} {bm:>8.1f} "
              f"{(max(bases) - min(bases)) / bm * 100:>7.2f} " + " ".join(row) + f"  {dec:+6.1f}")

    print("\nacross launches (mean of the per-launch ratios, n launches):")
    for key in sorted(per_tag):
        cells = (f"k={k} " + (f"{st.mean(v):+7.2f} (n={len(v)})" if (v := per_tag[key][k])
                              else f"{'-':>13}") for k in keeps)
        print(f"  {key[0]:8} btok={key[1]:<5} " + " ".join(cells))


if __name__ == "__main__":
    main()
