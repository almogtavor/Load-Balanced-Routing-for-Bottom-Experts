"""In-engine MoE layer time, per substitution budget, against the launch-to-launch floor.

    python3 experiments/moeplot.py

Reads `runs/moetime_{tag}_{arm}_r{rep}_{rank}.json` written by `moetime.py` riding along on
every serve launch. Same unit of analysis as `paired.py`: an arm is divided by the base arms of
*its own launch*, because an unmodified engine loaded twice measured several percent against
itself.

The bucket reported is the largest token count both the arm and its baseline recorded - the
prefill forward, where the all-to-all is bandwidth-bound and the byte saving can pay. Expert
GEMM work is conserved by construction (`imbalance.py` asserts it per router call), so at a
fixed token count the only term left to move is the collective.

`keep=8` is the zero-substitution control: the kernel runs and selects nothing different.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import re
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paired import arm_key, arm_label  # noqa: E402

# Which span to plot: the whole MoE layer, or one half of the collective. The whole layer nets
# the byte saving against the expert GEMM, whose grouped group sizes skew when the allowed
# expert set narrows - in the engine nothing caps an expert's token count, unlike the offline
# simulator. `PHASE=disp` puts the clock on the term the rule acts on.
PHASE = os.environ.get("PHASE", "moe")
KEY = "buckets" if PHASE == "moe" else PHASE
NAME = {"moe": "MoE layer", "disp": "DeepEP dispatch", "comb": "DeepEP combine"}[PHASE]
STEM = f"runs/{'moetime' if PHASE == 'moe' else PHASE + 'time'}"

# `q30c4` and `q30e8` are the same model on the same host type at two world sizes, so the
# ep scaling can be read without crossing clusters. DeepEP only accepts a power-of-two rank
# count, so there is no ep=6 point between them.
LABEL = {"q30pf": "Qwen3-30B-A3B  E=128  ep=4", "q35pf": "Qwen3.5-35B  E=256  ep=4",
         "gmpf": "Gemma-4-26B-A4B  E=128  ep=4", "q30c4": "Qwen3-30B-A3B  E=128  ep=4  (host B)",
         "q30e8": "Qwen3-30B-A3B  E=128  ep=8  (host B)"}
COLOR = {"q30pf": "#1f77b4", "q35pf": "#d62728", "gmpf": "#9467bd", "q30c4": "#ff7f0e",
         "q30e8": "#2ca02c"}
OUT = f"{STEM}.png"


def load() -> dict:
    """`{(tag, rep): {arm: {bucket: median ms per MoE forward, averaged over ranks}}}`."""
    runs: dict = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for f in glob.glob("runs/moetime_*.json"):
        m = re.match(r"moetime_(.+)_([^_]+)_r(\d+)_(\d+)\.json$", os.path.basename(f))
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue   # a worker was killed mid-write, or two raced on one path
        for b, (ms, n) in d.get(KEY, {}).items():
            if n >= 64:   # a bucket seen a handful of times is a transient, not a shape
                runs[(m.group(1), int(m.group(3)))][m.group(2)][int(b)].append(ms)
    return {k: {a: {b: st.mean(v) for b, v in bk.items()} for a, bk in arms.items()}
            for k, arms in runs.items()}


def gains() -> tuple[dict, dict, dict]:
    """`{tag: {arm: [per-launch %]}}`, the floor per tag, and the stock ms per tag.

    The floor is per tag, not global: a cell in one row is only ever divided by launches of
    its own tag, so another model's noisiest launch has no bearing on this row's resolution.
    One global `max(spread)/2` let a single wide launch inflate every row at once.
    """
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    stock = {}
    spreads = collections.defaultdict(list)
    for (tag, _rep), arms in load().items():
        base = {a: bk for a, bk in arms.items() if a.startswith("base")}
        if not base or tag not in LABEL:
            continue
        for a, bk in arms.items():
            if a.startswith("base"):
                continue
            # The largest forward size this arm and every base arm of this launch both saw.
            shared = set(bk) & set.intersection(*(set(v) for v in base.values()))
            if not shared:
                continue
            b = max(shared)
            bm = st.mean(v[b] for v in base.values())
            per[tag][a].append((bm / bk[b] - 1) * 100)
            stock[tag] = (bm, b)
        # The floor is base-against-base, so only a launch that loaded the stock engine twice
        # contributes to it; a launch with one base arm still yields paired gains above.
        bl = [v for v in base.values()]
        shared = set.intersection(*(set(v) for v in bl)) if len(bl) > 1 else set()
        if shared:
            b = max(shared)
            lo, hi = min(v[b] for v in bl), max(v[b] for v in bl)
            spreads[tag].append((hi - lo) / lo * 100)
    if not per:
        raise SystemExit("no paired moetime launches yet")
    # A row with no base-against-base launch has no measured resolution, so it is not
    # publishable yet - drop it rather than print a cell nothing bounds. A sweep in flight
    # hits this for a few minutes, because `base2` is the last arm to load.
    for t in [t for t in per if t not in spreads]:
        print(f"SKIP {t}: no launch loaded the stock engine twice, floor unmeasured")
        del per[t]
    if not per:
        raise SystemExit("no paired moetime launches with a measured floor yet")
    return per, {t: max(v) / 2 for t, v in spreads.items() if t in per}, stock


def main() -> None:
    per, floor, stock = gains()
    band = max(floor[t] for t in per)   # one shaded band, so it is the widest row's
    arms = sorted({a for t in per.values() for a in t}, key=arm_key)
    tags = [t for t in LABEL if t in per]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        xs = [j + (i - (len(tags) - 1) / 2) * w for j in range(len(arms))]
        ys = [st.mean(per[t][a]) if per[t][a] else 0.0 for a in arms]
        er = [st.stdev(per[t][a]) if len(per[t][a]) > 1 else 0.0 for a in arms]
        ax.bar(xs, ys, w * 0.92, yerr=er, capsize=3, color=COLOR[t],
               label=f"{LABEL[t]}  ({stock[t][0]:.2f} ms stock, floor +/-{floor[t]:.2f}%)")
    ax.axhspan(-band, band, color="0.82", zorder=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(arms)), [arm_label(a) for a in arms])
    ax.set_ylabel(f"{NAME} faster than stock routing (%)\n" + r"$\uparrow$ higher is better")
    ax.set_title(f"Placement-aware refill of the bottom experts: {NAME} time in vLLM\n"
                 "DeepEP high-throughput all-to-all, prefill forwards, paired within launch",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    rows = [f"| model | stock ms / {NAME.lower()} \u2193 | tokens / forward | "
            "floor | " + " | ".join(arm_label(a).replace("\n", " ") + " \u2191"
                                    for a in arms) + " |",
            "|---" * (len(arms) + 4) + "|"]
    for t in tags:
        # The launch count rides in the cell, not in one column: arms entered the grid at
        # different times, so a single per-row count would claim replication an arm measured
        # once does not have.
        cells = [f"**{st.mean(v):+.1f}%** x{len(v)}" if (v := per[t][a]) else "-" for a in arms]
        rows.append(f"| {LABEL[t]} | {stock[t][0]:.2f} | {stock[t][1]} | "
                    f"+/-{floor[t]:.2f}% | " + " | ".join(cells) + " |")
    md = "\n".join(rows)
    where = {"moe": "around `MoERunner._forward_impl` - routing, dispatch, expert GEMM, combine",
             "disp": "around `DeepEPHTPrepareAndFinalize.prepare_async` and the wait on its "
                     "completion event - the all-to-all the rule shrinks",
             "comb": "around `DeepEPHTPrepareAndFinalize._finalize` and the wait on its "
                     "completion event"}[PHASE]
    open(f"{STEM}.md", "w").write(
        f"{NAME} time inside a running vLLM 0.31 engine on the DeepEP high-throughput "
        f"all-to-all, timed with CUDA events {where}, and reduced to a median per token-count "
        "bucket. Each arm is divided by the base arms of its own launch; `keep=8` "
        "is the zero-substitution control. Positive is faster, so in every percentage column "
        "higher is better (marked \u2191 in the header; \u2193 marks the one column where lower "
        "is better). The floor is that model's own "
        "widest base-against-base spread, halved, over the launches in its row.\n\n"
        f"{md}\n")
    print(md)
    print(f"\nWROTE {OUT} and {STEM}.md")


if __name__ == "__main__":
    main()
