"""In-engine prefill throughput, one bar per budget, against the launch-to-launch floor.

    python3 experiments/serveplot.py

Reads the same `runs/serve_*.json` artefacts as `paired.py` and the same way: a keep arm is
divided by the mean of the base arms *from its own launch*, because an unmodified engine
loaded twice on the same node measured several percent against itself. The grey band is that
floor, measured here rather than assumed - half the widest base-to-base spread seen in any
launch. `keep=8` is the zero-substitution control: the plugin runs, selects nothing different,
and whatever it costs shows up in that bar and in no other.

The shape is prefill-dominated on purpose (`PROMPT_LEN` tokens in, one token out, a per-step
budget large enough that the all-to-all is bandwidth-bound). The rule moves `V` in
`T = L + V/B_eff` and never `L`, so a decode-heavy shape spends its wall clock in the term the
rule cannot touch.
"""

from __future__ import annotations

import collections
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paired import arm_key, arm_label, load  # noqa: E402

METRIC = "tot_tok_per_s"
LABEL = {"q30pf": "Qwen3-30B-A3B  E=128  ep=4", "q35pf": "Qwen3.5-35B  E=256  ep=4",
         "gmpf": "Gemma-4-26B-A4B  E=128  ep=4", "q30e8": "Qwen3-30B-A3B  E=128  ep=8"}
COLOR = {"q30pf": "#1f77b4", "q35pf": "#d62728", "gmpf": "#9467bd", "q30e8": "#2ca02c"}
OUT = "runs/serve.png"


def gains() -> tuple[dict, float, dict]:
    """`{tag: {arm: [per-launch %]}}`, the control floor, and the per-tag base rate."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    rate, spreads = {}, []
    for (tag, _rep), arms in load()[0].items():
        bases = [v[METRIC] for a, v in arms.items() if a.startswith("base")]
        if len(bases) < 2 or tag not in LABEL:
            continue
        bm = st.mean(bases)
        spreads.append((max(bases) - min(bases)) / bm * 100)
        rate[tag] = bm
        for a, v in arms.items():
            if not a.startswith("base"):
                per[tag][a].append((v[METRIC] / bm - 1) * 100)
    if not per:
        raise SystemExit("no paired serve launches yet")
    return per, max(spreads) / 2, rate


def main() -> None:
    per, band, rate = gains()
    arms = sorted({a for t in per.values() for a in t}, key=arm_key)
    tags = [t for t in LABEL if t in per]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    w = 0.8 / len(tags)
    for i, t in enumerate(tags):
        xs = [j + (i - (len(tags) - 1) / 2) * w for j in range(len(arms))]
        ys = [st.mean(per[t][a]) if per[t][a] else 0.0 for a in arms]
        er = [st.stdev(per[t][a]) if len(per[t][a]) > 1 else 0.0 for a in arms]
        ax.bar(xs, ys, w * 0.92, yerr=er, capsize=3, color=COLOR[t],
               label=f"{LABEL[t]}  ({rate[t]:.0f} tok/s stock)")
    ax.axhspan(-band, band, color="0.82", zorder=0)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(range(len(arms)), [arm_label(a) for a in arms])
    ax.set_ylabel("prefill throughput vs stock routing (%)")
    ax.set_title("Placement-aware refill of the bottom experts, in vLLM, ep=4\n"
                 "4096-token prompts, 16384-token steps, DeepEP high-throughput dispatch",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    rows = ["| model | stock tok/s | " + " | ".join(arm_label(a).replace("\n", " ") for a in arms) + " | launches |",
            "|---" * (len(arms) + 3) + "|"]
    for t in tags:
        n = max(len(per[t][a]) for a in arms)
        cells = [f"**{st.mean(v):+.1f}%**" if (v := per[t][a]) else "-" for a in arms]
        rows.append(f"| {LABEL[t]} | {rate[t]:.0f} | " + " | ".join(cells) + f" | {n} |")
    md = "\n".join(rows)
    open("runs/serve.md", "w").write(
        "In-engine prefill throughput, vLLM 0.31 with the DeepEP high-throughput all-to-all, "
        f"ep=4, 4096-token prompts at a 16384-token step budget. Each arm is divided by the "
        f"base arms of its own launch; `keep=8` is the zero-substitution control. Launch-to-"
        f"launch floor +/-{band:.2f}%.\n\n{md}\n")
    print(md)
    print(f"\ncontrol floor +/-{band:.2f}%")
    print(f"WROTE {OUT} and runs/serve.md")


if __name__ == "__main__":
    main()
