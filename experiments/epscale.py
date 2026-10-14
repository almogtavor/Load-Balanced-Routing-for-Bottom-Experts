"""How the saving scales with the expert-parallel world, measured inside a running engine.

    python experiments/epscale.py

Reads the `runs/imbalance_*_0.json` written by `imbalance.py`, which scores every arm on the
logits of the *stock* forward: one engine load per family, the same router calls for every
arm, no run-to-run variance between them at all. The expert-to-rank map is `e * G // E`,
arithmetic on the expert id, so one load reports for any `ep` - and over 8-GPU nodes the same
arithmetic is the node map, which makes the `ep=4` row the inter-node fan-out of a 32-GPU
deployment.

Left panel: summed dispatch pairs removed. This is the all-to-all payload, and it is what
`speedup.py` turns into wall-clock on a real NCCL collective at `ep=4`. The saving grows with
the world because a fixed `keep` reaches a shrinking fraction of it.

Right panel: pairs removed from the *busiest* rank. The tail of the collective, which is what
the step actually waits on. The rule only ever deletes visits, never adds one, so no rank can
receive more than it did under stock and the maximum falls with the sum.

`load_penalty=1` throughout: it reorders tail picks within the ranks the head already forces a
visit to, so the two panels are bit-identical to the plain rule while the expert-slot spread
over those ranks comes out flatter. The table carries that column too.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAGS = ["imb2", "imbq35", "imbgm", "imbhy"]
LABEL = {"imb2": "Qwen3-30B-A3B  E=128", "imbq35": "Qwen3.5-35B  E=256",
         "imbgm": "Gemma-4-26B-A4B  E=128", "imbhy": "Hunyuan-A13B  E=64"}
COLOR = {"imb2": "#1f77b4", "imbq35": "#d62728", "imbgm": "#9467bd", "imbhy": "#2ca02c"}
MARK = {"imb2": "o", "imbq35": "s", "imbgm": "^", "imbhy": "D"}
KEEPS = [6, 4, 2]
PEN = 1.0
VOL, RECV, WORK, EXPERT = 0, 1, 2, 3
RUNS = pathlib.Path("runs")


def read(tag: str) -> dict[int, dict]:
    """`{ep: {keep: [metrics relative to that ep's stock arm]}}` for one family."""
    p = RUNS / f"imbalance_{tag}_0.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    arms = {tuple(float(x) for x in k.split("/")): v for k, v in d["arms"].items()}
    out: dict[int, dict] = {}
    for ep in d["eps"]:
        here = {k: v for k, v in arms.items() if k[0] == ep}
        if not here:
            continue
        # The stock arm is the one that asks for no substitution, so `keep == top_k`, which is
        # the largest keep recorded for this ep. Everything is reported against it.
        k0 = max(k[1] for k in here)
        base = here[(ep, k0, 0.0)]
        out[ep] = {int(keep): [v / b - 1 for v, b in zip(here[(ep, keep, PEN)], base)]
                   for keep in KEEPS if (ep, keep, PEN) in here}
    return out


def loadpen() -> None:
    """`runs/loadpen.md`: where the load penalty stops buying anything.

    From a separate sweep of the same probe at `PENS=0,1,2,4,8`. Only the expert-slot column
    moves - volume and busiest-rank receipts are bit-identical across penalties, because the
    tail stays confined to the head's ranks either way - so that is the only one tabulated.
    """
    p = RUNS / "imbalance_imbpen_0.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    arms = {tuple(float(x) for x in k.split("/")): v for k, v in d["arms"].items()}
    pens = sorted({k[2] for k in arms} - {0.0} | {0.0})
    rows = ["| ep | keep | " + " | ".join(f"`load_penalty={q:g}`" for q in pens) + " |",
            "|---|---|" + "---|" * len(pens)]
    for ep in d["eps"]:
        k0 = max(k[1] for k in arms if k[0] == ep)
        base = arms[(ep, k0, 0.0)][WORK]
        for keep in KEEPS:
            cells = [f"{100 * (arms[(ep, keep, q)][WORK] / base - 1):+.1f}%" for q in pens
                     if (ep, keep, q) in arms]
            if cells:
                rows.append(f"| {int(ep)} | {int(keep)} | " + " | ".join(cells) + " |")
    (RUNS / "loadpen.md").write_text(
        "# Expert slots on the busiest rank, over the mean, against stock routing\n\n"
        "Qwen3-30B-A3B in-engine, `EPS=4,8,16 PENS=0,1,2,4,8`. Negative is a flatter spread "
        "than stock routing. Dispatch volume and busiest-rank receipts do not appear because "
        "they are bit-identical across every penalty in the row.\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {RUNS / 'loadpen.md'}")


def main() -> None:
    data = {t: read(t) for t in TAGS}
    have = [t for t in TAGS if data[t]]
    assert have, "no runs/imbalance_*_0.json - run imbalance.py through run.sh first"
    eps = sorted({e for t in have for e in data[t]})

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, metric, title in ((axes[0], VOL, "dispatch pairs removed"),
                              (axes[1], RECV, "pairs removed from the busiest rank")):
        for i, keep in enumerate(KEEPS):
            # One line per budget through the family mean, with every family's own point on it:
            # the spread at a given ep is the across-family agreement, not an error bar.
            xs, ys = [], []
            for ep in eps:
                v = [-100 * data[t][ep][keep][metric] for t in have
                     if ep in data[t] and keep in data[t][ep]]
                if v:
                    xs.append(ep)
                    ys.append(sum(v) / len(v))
            ax.plot(xs, ys, "-", color=f"C{i}", lw=2, zorder=2, label=f"keep={keep} of top-8")
            for t in have:
                for ep in eps:
                    if ep in data[t] and keep in data[t][ep]:
                        ax.plot([ep], [-100 * data[t][ep][keep][metric]], MARK[t],
                                color=f"C{i}", ms=6, mfc="white", mew=1.4, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(eps)
        ax.set_xticklabels([str(e) for e in eps])
        ax.set_xlabel("expert-parallel ranks (= nodes, over 8-GPU hosts)")
        ax.set_ylabel("removed vs stock routing (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=9)
    handles = [plt.Line2D([], [], marker=MARK[t], color="0.35", mfc="white", ls="none",
                          label=LABEL[t]) for t in have]
    axes[0].add_artist(axes[0].get_legend())
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, title="family")
    fig.suptitle("Placement-aware tail routing: what it removes from the all-to-all, "
                 "measured on the stock router's own logits", fontsize=11)
    fig.tight_layout()
    fig.savefig(RUNS / "epscale.png", dpi=160)

    rows = ["| model | ep | keep | dispatch pairs | busiest rank | busiest expert | "
            "expert slots, busiest rank |", "|---|---|---|---|---|---|---|"]
    for t in have:
        for ep in sorted(data[t]):
            for keep in KEEPS:
                m = data[t][ep].get(keep)
                if m is None:
                    continue
                rows.append(f"| {LABEL[t].split('  ')[0]} | {ep} | {keep} | "
                            f"{-100 * m[VOL]:+.1f}% | {-100 * m[RECV]:+.1f}% | "
                            f"{-100 * m[EXPERT]:+.1f}% | {-100 * m[WORK]:+.1f}% |")
    (RUNS / "epscale.md").write_text(
        "# What the rule removes from the all-to-all, per expert-parallel world size\n\n"
        f"Measured in a running vLLM engine on the stock router's logits, "
        f"`load_penalty={PEN:g}`, every arm scored on the same router calls. Positive is less "
        "work than stock routing.\n\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"wrote {RUNS / 'epscale.png'} and {RUNS / 'epscale.md'}")
    loadpen()


if __name__ == "__main__":
    main()
