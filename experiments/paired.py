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

KEEPS = ["8", "6", "4", "2", "1"]
# The rule moves communication volume, which is `ib_b_per_tok` directly and
# `tot_tok_per_s` at a prefill-heavy shape. On the bytes metric a negative row is the
# rule working; on the rate metrics a positive one is.
METRIC = os.environ.get("METRIC") or "out_tok_per_s"


def load(pat="runs/serve_*.json"):
    runs = collections.defaultdict(dict)       # (tag, rep) -> arm -> record
    skipped = 0
    for f in glob.glob(pat):
        d = json.load(open(f))
        if d.get("written", "") < CUT:
            skipped += 1
            continue
        m = re.match(r"serve_(.+)_([^_]+)_r(\d+)\.json$", os.path.basename(f))
        runs[(m.group(1), int(m.group(3)))][m.group(2)] = d
    return runs, skipped


def decay(d):
    """First warm repeat to last, as a percentage. On a base arm this is instrument drift."""
    w = d["warm_tok_per_s"]
    return (w[0] - w[-1]) / w[0] * 100 if len(w) > 1 else float("nan")


def main():
    runs, skipped = load()
    print(f"excluded {skipped} artefacts written before the kernel fix ({CUT})")
    print(f"metric: {METRIC}")
    per_tag = collections.defaultdict(lambda: collections.defaultdict(list))
    print(f"{'tag':8} {'rep':>3} {'nb':>2} {'base':>8} {'spread%':>7} "
          + " ".join(f"{'k=' + k:>8}" for k in KEEPS) + "  decay%")
    for (tag, rep), arms in sorted(runs.items()):
        bases = [v[METRIC] for a, v in arms.items()
                 if a.startswith("base") and v.get(METRIC) is not None]
        if not bases:
            continue
        bm = st.mean(bases)
        row = []
        for k in KEEPS:
            if arms.get(k, {}).get(METRIC) is None:
                row.append(f"{'-':>8}")
                continue
            r = (arms[k][METRIC] / bm - 1) * 100
            per_tag[tag][k].append(r)
            row.append(f"{r:+8.2f}")
        dec = st.mean([decay(v) for a, v in arms.items() if a.startswith("base")])
        print(f"{tag:8} {rep:>3} {len(bases):>2} {bm:>8.1f} "
              f"{(max(bases) - min(bases)) / bm * 100:>7.2f} " + " ".join(row) + f"  {dec:+6.1f}")

    print("\nacross launches (mean of the per-launch ratios, n launches):")
    for tag in sorted(per_tag):
        cells = (f"k={k} " + (f"{st.mean(v):+7.2f} (n={len(v)})" if (v := per_tag[tag][k])
                              else f"{'-':>13}") for k in KEEPS)
        print(f"  {tag:8} " + " ".join(cells))


if __name__ == "__main__":
    main()
