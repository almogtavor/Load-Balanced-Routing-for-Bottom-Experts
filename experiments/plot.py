"""The one figure: what placement-aware re-routing costs in accuracy against what it buys.

    python plot.py

y, shared by both panels, is the measured speedup of one expert-parallel MoE layer on four
GPUs from Phase B. x is the accuracy cost, twice: the left panel is the paired `delta-NLL`
from Phase A with its standard error, the right is the paired `delta-accuracy` on a fixed
198-item sample of GPQA. Both panels are needed and neither is redundant - NLL is a sensitive continuous
measure with tight error bars but no units anyone cares about, and the benchmark is the thing
that matters but is blunt at 198 items. A claim that survives both is worth more than one that
needs a particular choice of metric.

Each colour is a model, filled and solid is placement-aware re-routing, open and dashed is the
random control that makes the same number of substitutions with no placement benefit. Every
curve starts at the origin, which is `keep = top_k` - the no-op arm, which both phases assert
is exactly the baseline. On the right panel a point ringed in black is one where McNemar
resolves the difference from the baseline at p < 0.05; everything unringed is a drop the
benchmark cannot distinguish from zero at this sample size, which is the honest reading and
not a null result.

Reading it: a point up and to the left of the origin is a free lunch, and the vertical gap
between the solid and dashed curve at equal `keep` is the part attributable to *placement*
rather than to the bottom experts being substitutable at all. The random control is why the
figure can make that second claim.
"""

from __future__ import annotations

import json
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAGS = os.environ.get("TAGS", "q30rh,q35,hy13b,gm26b").split(",")
LABEL = {"q30rh": "Qwen3-30B-A3B (E=128, k=8)",
         "q35": "Qwen3.5-35B (E=256, k=8)",
         "hy13b": "Hunyuan-A13B (E=64, k=8)",
         "gm26b": "Gemma-4-26B-A4B (E=128, k=8)"}
COLOR = {"q30rh": "#1f77b4", "q35": "#d62728", "hy13b": "#2ca02c", "gm26b": "#9467bd"}
RUNS = pathlib.Path("runs")
OUT = RUNS / "placement.png"


def series(acc: dict, lat: dict, bench: dict | None, arm: str) -> list[dict]:
    """One dict per `keep` of this arm, ordered from the no-op down.

    The no-op point is the *measured* no-op arm, not a drawn-in zero. It substitutes nothing, so
    whatever it shows is this model's floating-point floor, and every arm below it should be read
    against that floor rather than against an idealized origin. Its speedup is pinned to zero
    because Phase B times the baseline routing itself at `keep = top_k`.
    """
    b = lat["batches"][str(lat["plot_batch"])]
    n, bn = acc["arms"]["noop"], (bench or {}).get("arms", {}).get("noop")
    pts = [{"keep": acc["top_k"], "dnll": n["dnll"], "se": n["dnll_se"], "up": 0.0,
            "dacc": 100 * bn["dacc"] if bn else None, "sig": bool(bn and bn["resolved"])}]
    for name, r in acc["arms"].items():
        if not name.startswith(arm + "_"):
            continue
        assert name in b, f"Phase B has no timing for arm {name}"
        ba = (bench or {}).get("arms", {}).get(name)
        pts.append({"keep": int(name.rsplit("_", 1)[1]), "dnll": r["dnll"], "se": r["dnll_se"],
                    "up": (1 - b[name]["ms"] / b["baseline"]["ms"]) * 100,
                    "dacc": 100 * ba["dacc"] if ba else None,
                    "sig": bool(ba and ba["resolved"])})
    return sorted(pts, key=lambda p: -p["keep"])


def main() -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.4, 5.2), sharey=True)
    batch, drawn = None, []
    for tag in TAGS:
        a, l = RUNS / f"acc_{tag}.json", RUNS / f"lat_{tag}.json"
        if not (a.exists() and l.exists()):
            print(f"[skip] {tag}: need {a.name} and {l.name}")
            continue
        bf = RUNS / f"bench_{tag}.json"
        acc, lat = json.loads(a.read_text()), json.loads(l.read_text())
        bench = json.loads(bf.read_text()) if bf.exists() else None
        if bench is None:
            print(f"[warn] {tag}: no {bf.name}, benchmark panel will omit it")
        batch, c = lat["plot_batch"], COLOR.get(tag, "#777777")
        drawn.append(tag)
        for arm, style in (("placement", dict(ls="-", marker="o", mfc=c)),
                           ("random", dict(ls="--", marker="o", mfc="white"))):
            pts = series(acc, lat, bench, arm)
            ax.errorbar([p["dnll"] for p in pts], [p["up"] for p in pts],
                        xerr=[p["se"] for p in pts], color=c, ecolor=c, elinewidth=0.8,
                        capsize=2, lw=1.6, ms=6, mew=1.4, mec=c, **style)
            bp = [p for p in pts if p["dacc"] is not None]
            if bp:
                bx.plot([p["dacc"] for p in bp], [p["up"] for p in bp],
                        color=c, lw=1.6, ms=6, mew=1.4, mec=c, **style)
                # A black ring is the only thing on the figure that says "this drop is real".
                sig = [p for p in bp if p["sig"]]
                bx.plot([p["dacc"] for p in sig], [p["up"] for p in sig], ls="none",
                        marker="o", ms=11, mfc="none", mec="black", mew=1.2)
            if arm == "placement":
                for p in pts:
                    ax.annotate(f" keep={p['keep']}", (p["dnll"], p["up"]), fontsize=7,
                                color=c, va="center", ha="left")

    if batch is None:
        raise SystemExit("no acc_*.json / lat_*.json pairs in runs/")
    for a_, xl in ((ax, r"paired $\Delta$NLL vs. unmodified routing"
                        "\n" r"(right is worse, bars are $\pm$1 s.e.)"),
                   (bx, r"paired $\Delta$accuracy on GPQA, 198 items (points)"
                        "\n(left is worse, black ring = McNemar $p<0.05$)")):
        a_.axhline(0, color="black", lw=0.6)
        a_.axvline(0, color="black", lw=0.6)
        a_.set_xlabel(xl)
        a_.grid(alpha=0.25)
    ax.set_ylabel(f"measured MoE layer speedup on 4 GPUs (%)\n"
                  f"batch {batch} tokens, expert-parallel, NCCL all-to-all")
    fig.suptitle("Re-routing bottom experts to GPUs the token already visits")
    handles = [plt.Line2D([], [], color=COLOR[t], marker="o", label=LABEL.get(t, t))
               for t in drawn] \
        + [plt.Line2D([], [], color="black", ls="-", marker="o", label="placement-aware"),
           plt.Line2D([], [], color="black", ls="--", marker="o", mfc="white",
                      label="random control (same substitutions)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print(f"WROTE {OUT}")


if __name__ == "__main__":
    main()
