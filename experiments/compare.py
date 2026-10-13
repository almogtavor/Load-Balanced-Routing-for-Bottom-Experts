"""The real figure: what the kernel buys a running vLLM server, against what it costs.

    python compare.py

Pairs the per-arm files `vllm_run.py` wrote. y is end-to-end decode throughput relative to the
unmodified engine; x is the paired accuracy delta on the same 198 GPQA items, tested with the
same exact McNemar test `bench.py` uses, so the offline figure and this one can be read side by
side. Every number is a wall-clock measurement on a shared cluster, so the grey band is the
`base2` arm - the stock engine loaded and measured a second time - and a point inside that band
is one the instrument cannot call a speedup.

A launch is the unit of analysis, not a repeat. The repeats inside one launch sample one
engine's steady state and spread under 1%, while `base` and `base2` - the same configuration
launched twice - disagreed by 5.3%, so what separates two arms is mostly which engine got
started. `REPS` sweeps every arm once per pass and writes `_r<n>`; the y error bar is the
spread of the per-sweep paired ratio across those passes, and the p is an exact sign-flip test
on it. An arm with one launch has no such spread: it is printed with `1 launch` and its bar
falls back to the within-launch repeats, which is a floor on the error and not the error.
A black ring means McNemar resolves the accuracy difference at p < 0.05.

Unlike the offline sweep this leaves nothing to account for: the all-to-all, the expert GEMMs,
the scheduler and the KV cache are all vLLM's, and the only thing replaced is which experts the
router returns.
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench  # noqa: E402

# A tag is one (model, all-to-all backend) pair, because the backend decides whether the
# dispatch volume this kernel shrinks is the volume the engine actually moves. The `dp` tags are
# DeepEP, measured through a real server by `serve_run.py` (`serve_*.json`); the others are vLLM
# 0.31's default `allgather_reducescatter`, measured offline by `vllm_run.py` (`vllm_*.json`).
#
# A `dp` run measures throughput only, and takes its x from the same model's allgather run. That
# is not an approximation: which experts a token is routed to is decided by the router, and the
# all-to-all backend only changes how the tokens then move. Same model, same rule, same items,
# same selection - so the accuracy is the same measurement, and running it twice would only add
# a second sample of GPQA noise.
ACC_FROM = {"q30dp": "q30rh", "q35dp": "q35", "gmdp": "gm26b"}
TAGS = os.environ.get("TAGS", "q30rh,q35,gm26b,q30dp,q35dp,gmdp").split(",")
LABEL = {"q30rh": "Qwen3-30B-A3B, allgather", "q35": "Qwen3.5-35B, allgather",
         "hy13b": "Hunyuan-A13B", "gm26b": "Gemma-4-26B-A4B, allgather",
         "q30dp": "Qwen3-30B-A3B, DeepEP", "q35dp": "Qwen3.5-35B, DeepEP",
         "gmdp": "Gemma-4-26B-A4B, DeepEP"}
COLOR = {"q30rh": "#aec7e8", "q35": "#ff9896", "hy13b": "#2ca02c", "gm26b": "#c5b0d5",
         "q30dp": "#1f77b4", "q35dp": "#d62728", "gmdp": "#9467bd"}
RUNS = pathlib.Path("runs")
OUT = RUNS / "vllm.png"


def warm(r: dict) -> np.ndarray:
    """The arm's warm throughput repeats. One number per repeat, the first one discarded."""
    return np.array(r.get("warm_tok_per_s") or [r["out_tok_per_s"]], float)


def sign_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip p for `mean(d) != 0` on a paired per-launch quantity.

    The arm and its base are measured in the same sweep, so the pairing absorbs whatever the
    cluster was doing that pass and only the sign of each difference is exchangeable. With a
    handful of sweeps all 2^n flips are cheaper than any approximation of them, and the test
    assumes nothing about the distribution of wall-clock times on a shared cluster - which is
    exactly the assumption one should not want to make here.
    """
    assert len(d) <= 16, f"{2 ** len(d)} sign flips is no longer the cheap way"
    obs = abs(d.mean())
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float((np.abs((flips * d).mean(1)) >= obs - 1e-12).mean())


def arms(tag: str) -> dict[str, dict[int, dict]]:
    """`arm -> {sweep index: that launch's record}`, from the `_r<n>` suffix in the filename."""
    pre = "serve" if tag in ACC_FROM else "vllm"
    out: dict[str, dict[int, dict]] = {}
    for p in sorted(RUNS.glob(f"{pre}_{tag}_*.json")):
        m = re.fullmatch(r"(.+)_r(\d+)", p.stem.split("_", 2)[2])
        assert m, f"{p.name}: no sweep index, so a launch cannot be identified"
        out.setdefault(m[1], {})[int(m[2])] = json.loads(p.read_text())
    return out


def first(launches: dict[int, dict]) -> dict:
    """The earliest sweep's record. For quality, where a launch is not a source of variance."""
    return launches[min(launches)]


def rows(tag: str) -> list[dict]:
    """One row per arm of this model, measured against its own `base`."""
    a = arms(tag)
    if "base" not in a:
        return []
    base = a["base"]
    # Where the accuracy of this arm was measured: its own file, or the same model's allgather
    # run for a throughput-only `dp` tag.
    acc = arms(ACC_FROM[tag]) if tag in ACC_FROM else a
    out = []
    for name, launches in a.items():
        if name == "base":
            continue
        # A `base*` arm is the stock engine under a different sweep position, so its accuracy
        # is `base`'s by construction and is not measured again. Every other arm needs its own.
        src = "base" if name.startswith("base") else name
        if src not in acc or "base" not in acc:
            print(f"[skip arm] {tag}/{name}: no accuracy run to pair it with")
            continue
        r = first(launches)
        ab, ar = first(acc["base"]), first(acc[src])
        assert len(ar["gpqa_hit"]) == len(ab["gpqa_hit"]), f"{tag}/{name}: item count moved"
        m = bench.mcnemar(np.array(ab["gpqa_hit"], bool), np.array(ar["gpqa_hit"], bool))
        # Language modelling, paired by window index. Same source as the accuracy: what the
        # model predicts is the router's business, not the dispatch backend's. `nan` for runs
        # written before this phase existed.
        if "nll_per_window" in ar and "nll_per_window" in ab:
            d = np.array(ar["nll_per_window"]) - np.array(ab["nll_per_window"])
            dnll, dnll_se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
        else:
            dnll = dnll_se = float("nan")
        # One ratio per sweep, arm over base from that same sweep, and the spread of those
        # ratios is the error bar. Sweeps the base did not run are dropped rather than
        # compared against a different pass's engine.
        reps = sorted(set(launches) & set(base))
        assert reps, f"{tag}/{name}: no sweep where both this arm and base ran"
        pair = np.array([warm(launches[i]).mean() / warm(base[i]).mean() for i in reps])
        ra = float(pair.mean())
        if len(pair) > 1:
            rse = float(pair.std(ddof=1) / np.sqrt(len(pair)))
            p_tp = sign_p(np.log(pair))
        else:
            # A floor, not an error: it is the spread of one engine's repeats, and the thing
            # that actually varies between arms is which engine was started.
            wa, wb = warm(launches[reps[0]]), warm(base[reps[0]])
            rse = ra * np.hypot(wa.std(ddof=1) / np.sqrt(len(wa)) / wa.mean(),
                                wb.std(ddof=1) / np.sqrt(len(wb)) / wb.mean()
                                ) if len(wa) > 1 else 0.0
            p_tp = float("nan")
        out.append({"arm": name, "keep": r["keep"],
                    "up": (ra - 1) * 100, "up_se": 100 * float(rse), "p_tp": p_tp,
                    "n_launch": len(pair),
                    "tok_s": float(np.mean([warm(launches[i]).mean() for i in reps])),
                    "base_tok_s": float(np.mean([warm(base[i]).mean() for i in reps])),
                    "dacc": 100 * m["dacc"], "p": m["p_mcnemar"], "sig": m["resolved"],
                    "dnll": dnll, "dnll_se": dnll_se,
                    "n_disc": m["n_discordant"]})
    # `base2` first, then the keeps from the mildest down, so the table reads as a sweep.
    return sorted(out, key=lambda r: (r["keep"] is not None, -(r["keep"] or 0)))


def main() -> None:
    # Two x axes against the same y: GPQA on the left, language modelling on the right.
    # They are the two halves of "did this cost anything", and they have very different
    # power - 198 items resolve a few points at best, 32 paired windows resolve a
    # thousandth of a nat - so showing one without the other would read as a stronger or
    # weaker result than the data supports.
    fig, (ax, axn) = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True)
    drawn, floor = [], []
    for tag in TAGS:
        rs = rows(tag)
        if not rs:
            pre = "serve" if tag in ACC_FROM else "vllm"
            print(f"[skip] {tag}: no {pre}_{tag}_base_r*.json")
            continue
        drawn.append(tag)
        c = COLOR.get(tag, "#777777")
        print(f"\n{LABEL.get(tag, tag)}  base {rs[0]['base_tok_s']:.1f} out-tok/s "
              f"(mean over {rs[0]['n_launch']} launch(es))")
        for r in rs:
            print(f"  {r['arm']:>5}  {r['tok_s']:8.1f} tok/s  "
                  f"{r['up']:+6.2f}%+-{r['up_se']:.2f}  p_tput={r['p_tp']:.3f}  "
                  f"{r['n_launch']} launch{'es' if r['n_launch'] > 1 else ' (bar is a floor)'}   "
                  f"dacc {r['dacc']:+6.2f}  p_acc={r['p']:.3f}  discordant={r['n_disc']}"
                  f"   dnll {r['dnll']:+.5f}+-{r['dnll_se']:.5f}"
                  f"{'  ACC-RESOLVED' if r['sig'] else ''}")
        for r in rs:
            if r["keep"] is None:
                floor.append(abs(r["up"]) + r["up_se"])
                continue
            ax.errorbar(r["dacc"], r["up"], yerr=r["up_se"], marker="o", ms=7, color=c,
                        mec=c, mfc=c, ecolor=c, elinewidth=0.9, capsize=2)
            ax.annotate(f" keep={r['keep']}", (r["dacc"], r["up"]), fontsize=7, color=c,
                        va="center", ha="left")
            if r["sig"]:
                ax.plot(r["dacc"], r["up"], ls="none", marker="o", ms=13, mfc="none",
                        mec="black", mew=1.2)
            if not np.isnan(r["dnll"]):
                axn.errorbar(r["dnll"], r["up"], yerr=r["up_se"], xerr=r["dnll_se"], marker="o",
                             ms=7, color=c, mec=c, mfc=c, ecolor=c, elinewidth=0.9, capsize=2)
                axn.annotate(f" keep={r['keep']}", (r["dnll"], r["up"]), fontsize=7, color=c,
                             va="center", ha="left")
                # Resolved at 3 sigma on a paired difference over 32 windows, which is the
                # threshold at which this axis says more than the GPQA one does.
                if abs(r["dnll"]) > 3 * r["dnll_se"]:
                    axn.plot(r["dnll"], r["up"], ls="none", marker="o", ms=13, mfc="none",
                             mec="black", mew=1.2)
        keeps = [r for r in rs if r["keep"] is not None]
        ax.plot([r["dacc"] for r in keeps], [r["up"] for r in keeps], color=c, lw=1.4)
        kn = [r for r in keeps if not np.isnan(r["dnll"])]
        axn.plot([r["dnll"] for r in kn], [r["up"] for r in kn], color=c, lw=1.4)

    if not drawn:
        raise SystemExit("no vllm_*_base.json in runs/")
    if floor:
        # `base2` against `base` is the stock engine against itself, so this band is the
        # resolution of the y axis: an arm inside it is not distinguishable from relaunching
        # the baseline. Drawing it is the difference between a measurement and a number.
        f = max(floor)
        ax.axhspan(-f, f, color="black", alpha=0.08, zorder=0)
        ax.annotate(f"baseline relaunched ($\\pm${f:.2f}%)", (0.02, 0.5 + f), xycoords=
                    ("axes fraction", "data"), fontsize=7, color="#444444", va="bottom")
        axn.axhspan(-f, f, color="black", alpha=0.08, zorder=0)
    for a in (ax, axn):
        a.axhline(0, color="black", lw=0.6)
        a.axvline(0, color="black", lw=0.6)
        a.grid(alpha=0.25)
    ax.set_xlabel(r"paired $\Delta$accuracy on GPQA, 198 items"
                  "\n(left is worse, black ring = McNemar $p<0.05$)")
    axn.set_xlabel(r"paired $\Delta$NLL on held-out text, 32 windows"
                   "\n(right is worse, black ring = $|\\Delta| > 3\\sigma$)")
    ax.set_ylabel("end-to-end decode throughput vs. unmodified vLLM (%)")
    fig.suptitle("Placement-aware routing in vLLM, expert-parallel over 4 GPUs")
    ax.legend(handles=[plt.Line2D([], [], color=COLOR[t], marker="o", label=LABEL.get(t, t))
                       for t in drawn], fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"\nWROTE {OUT}")


if __name__ == "__main__":
    main()
