"""The real figure: what the kernel buys a running vLLM server, against what it costs.

    python compare.py

Pairs the per-arm files `vllm_run.py` wrote. y is end-to-end decode throughput relative to the
unmodified engine; x is the paired accuracy delta on the same 198 GPQA items, tested with the
same exact McNemar test `bench.py` uses, so the offline figure and this one can be read side by
side. Every number is a wall-clock measurement on a shared cluster, so the grey band is the
`base2` arm - the stock engine loaded and measured a second time - and a point inside that band
is one the instrument cannot call a speedup. The y error bar is the spread over warm repeats of
one engine, which is the smaller of the two uncertainties; the band is the honest one. A black
ring means McNemar resolves the accuracy difference at p < 0.05.

Unlike the offline sweep this leaves nothing to account for: the all-to-all, the expert GEMMs,
the scheduler and the KV cache are all vLLM's, and the only thing replaced is which experts the
router returns.
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench  # noqa: E402

TAGS = os.environ.get("TAGS", "q30rh,q35,hy13b,gm26b").split(",")
LABEL = {"q30rh": "Qwen3-30B-A3B", "q35": "Qwen3.5-35B",
         "hy13b": "Hunyuan-A13B", "gm26b": "Gemma-4-26B-A4B"}
COLOR = {"q30rh": "#1f77b4", "q35": "#d62728", "hy13b": "#2ca02c", "gm26b": "#9467bd"}
RUNS = pathlib.Path("runs")
OUT = RUNS / "vllm.png"


def warm(r: dict) -> np.ndarray:
    """The arm's warm throughput repeats. One number per repeat, the first one discarded."""
    return np.array(r.get("warm_tok_per_s") or [r["out_tok_per_s"]], float)


def perm_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation p for `mean(a) != mean(b)`.

    With five repeats an arm there are only 252 ways to split the ten numbers, so the exact
    test is cheaper than any approximation of it - and it needs no distributional assumption
    about wall-clock times on a shared cluster, which is exactly the assumption one should
    not want to make here.
    """
    pool, n = np.concatenate([a, b]), len(a)
    obs = abs(a.mean() - b.mean())
    splits = list(itertools.combinations(range(len(pool)), n))
    hits = sum(abs(pool[list(i)].mean() - pool[list(set(range(len(pool))) - set(i))].mean())
               >= obs - 1e-12 for i in splits)
    return hits / len(splits)


def arms(tag: str) -> dict[str, dict]:
    return {p.stem.split("_", 2)[2]: json.loads(p.read_text())
            for p in sorted(RUNS.glob(f"vllm_{tag}_*.json"))}


def rows(tag: str) -> list[dict]:
    """One row per arm of this model, measured against its own `base`."""
    a = arms(tag)
    if "base" not in a:
        return []
    base = a["base"]
    out = []
    for name, r in a.items():
        if name == "base":
            continue
        assert len(r["gpqa_hit"]) == len(base["gpqa_hit"]), f"{tag}/{name}: item count moved"
        m = bench.mcnemar(np.array(base["gpqa_hit"], bool), np.array(r["gpqa_hit"], bool))
        wa, wb = warm(r), warm(base)
        # Ratio of means with the delta method, so the error bar on the figure is the error on
        # the quantity actually plotted rather than on either arm alone.
        ra = wa.mean() / wb.mean()
        rse = ra * np.hypot(wa.std(ddof=1) / np.sqrt(len(wa)) / wa.mean(),
                            wb.std(ddof=1) / np.sqrt(len(wb)) / wb.mean()) if len(wa) > 1 else 0.0
        out.append({"arm": name, "keep": r["keep"],
                    "up": (ra - 1) * 100, "up_se": 100 * float(rse),
                    "p_tp": perm_p(wa, wb) if len(wa) > 1 and len(wb) > 1 else float("nan"),
                    "n_rep": len(wa),
                    "tok_s": wa.mean(), "base_tok_s": wb.mean(),
                    "dacc": 100 * m["dacc"], "p": m["p_mcnemar"], "sig": m["resolved"],
                    "n_disc": m["n_discordant"]})
    # `base2` first, then the keeps from the mildest down, so the table reads as a sweep.
    return sorted(out, key=lambda r: (r["keep"] is not None, -(r["keep"] or 0)))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    drawn, floor = [], []
    for tag in TAGS:
        rs = rows(tag)
        if not rs:
            print(f"[skip] {tag}: no vllm_{tag}_base.json")
            continue
        drawn.append(tag)
        c = COLOR.get(tag, "#777777")
        print(f"\n{LABEL.get(tag, tag)}  base {rs[0]['base_tok_s']:.1f} out-tok/s "
              f"(mean of {rs[0]['n_rep']} warm repeats)")
        for r in rs:
            print(f"  {r['arm']:>5}  {r['tok_s']:8.1f} tok/s  "
                  f"{r['up']:+6.2f}%+-{r['up_se']:.2f}  p_tput={r['p_tp']:.3f}   "
                  f"dacc {r['dacc']:+6.2f}  p_acc={r['p']:.3f}  discordant={r['n_disc']}"
                  f"{'  ACC-RESOLVED' if r['sig'] else ''}")
        for r in rs:
            if r["keep"] is None:
                floor.append(abs(r["up"]))
                continue
            ax.errorbar(r["dacc"], r["up"], yerr=r["up_se"], marker="o", ms=7, color=c,
                        mec=c, mfc=c, ecolor=c, elinewidth=0.9, capsize=2)
            ax.annotate(f" keep={r['keep']}", (r["dacc"], r["up"]), fontsize=7, color=c,
                        va="center", ha="left")
            if r["sig"]:
                ax.plot(r["dacc"], r["up"], ls="none", marker="o", ms=13, mfc="none",
                        mec="black", mew=1.2)
        keeps = [r for r in rs if r["keep"] is not None]
        ax.plot([r["dacc"] for r in keeps], [r["up"] for r in keeps], color=c, lw=1.4)

    if not drawn:
        raise SystemExit("no vllm_*_base.json in runs/")
    if floor:
        # The repeat baseline is the resolution of the y axis. Drawing it is the difference
        # between a measurement and a number.
        f = max(floor)
        ax.axhspan(-f, f, color="black", alpha=0.08, zorder=0)
        ax.annotate(f"repeat-baseline spread ($\\pm${f:.2f}%)", (0.02, 0.5 + f), xycoords=
                    ("axes fraction", "data"), fontsize=7, color="#444444", va="bottom")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"paired $\Delta$accuracy on GPQA, 198 items"
                  "\n(left is worse, black ring = McNemar $p<0.05$)")
    ax.set_ylabel("end-to-end decode throughput vs. unmodified vLLM (%)")
    ax.set_title("Placement-aware routing in vLLM, expert-parallel over 4 GPUs")
    ax.legend(handles=[plt.Line2D([], [], color=COLOR[t], marker="o", label=LABEL.get(t, t))
                       for t in drawn], fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
