"""The headline: what confining the tail to the head's ranks buys on a real all-to-all.

    python experiments/speedup.py

Reads the `runs/lat_*.json` written by `latency.py` - four model families, a real four-rank
NCCL `all_to_all_single`, expert capacity fixed per batch from `B, k, E` alone and therefore
identical in every arm, so the expert GEMM cancels and what differs between two arms is the
dispatch payload and nothing else.

Left panel: dispatch speedup against tokens per step. The rule moves `V` in `T = L + V/B_eff`
and not `L`, so it buys nothing while the collective is launch-bound and approaches the full
byte ratio once it is bandwidth-bound - which is why the curve rises with the batch and why
the honest thing is to plot all three batches rather than pick one. The grey band is the
measured control envelope: `noop` (routing identical to the baseline), `baseline2` (the same
configuration timed a second time) and `random_k` (the same number of substitutions, chosen
at random, which cuts no fan-out). Anything inside the band is harness resolution.

Right panel: the mechanism, at the bandwidth-bound batch. x is the speedup predicted by the
byte ratio alone - mean fan-out before over mean fan-out after, which is exactly the factor
the dispatch payload shrinks by. Four families and three budgets land on one line below the
diagonal: the rule delivers a fixed fraction of its own byte saving, and the shortfall is the
launch term it cannot touch. The random control sits at the origin, which is the claim the
figure exists to make - the gain is placement, not substitution.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TAGS = ["q30rh", "q35", "gm26b", "hy13b"]
LABEL = {"q30rh": "Qwen3-30B-A3B  E=128", "q35": "Qwen3.5-35B  E=256",
         "gm26b": "Gemma-4-26B-A4B  E=128", "hy13b": "Hunyuan-A13B  E=64"}
COLOR = {"q30rh": "#1f77b4", "q35": "#d62728", "gm26b": "#9467bd", "hy13b": "#2ca02c"}
KEEPS = [6, 4, 2]
RUNS = pathlib.Path("runs")
OUT = RUNS / "speedup.png"


def load() -> dict:
    d = {}
    for t in TAGS:
        p = RUNS / f"lat_{t}.json"
        if p.exists():
            d[t] = json.loads(p.read_text())
        else:
            print(f"[skip] {t}: no {p.name}")
    if not d:
        raise SystemExit("no runs/lat_*.json")
    return d


def up(arms: dict, name: str, field: str) -> float:
    """Speedup of `name` over the baseline, in percent. Positive is faster."""
    return (arms["baseline"][field] / arms[name][field] - 1) * 100


def control_band(runs: dict, B: str) -> float:
    """The largest dispatch speedup any control shows at this batch - the resolution floor.

    All three are arms that must not move: `noop` and `baseline2` do not change the routing at
    all, and `random_k` changes it without cutting fan-out. Their spread is what a real effect
    has to clear, and it is measured rather than assumed.
    """
    c = [abs(up(a, n, "ms_disp")) for r in runs.values() if B in r["batches"]
         for a in [r["batches"][B]]
         for n in ["noop", "baseline2", *(f"random_{k}" for k in KEEPS)] if n in a]
    return max(c)


def table(runs: dict) -> str:
    rows = ["| model | keep | mean fan-out | dispatch bytes | **dispatch** | layer | random control |",
            "|---|---|---|---|---|---|---|"]
    for t, r in runs.items():
        a = r["batches"][str(r["plot_batch"])]
        f0 = a["baseline"]["fanout"]
        for k in KEEPS:
            arm, rnd = f"placement_{k}", f"random_{k}"
            if arm not in a:
                continue
            rows.append(
                f"| {LABEL[t]} | {k} of {r['top_k']} | {f0:.2f} -> {a[arm]['fanout']:.2f} "
                f"| {a[arm]['fanout'] / f0 - 1:+.1%} | **{up(a, arm, 'ms_disp'):+.1f}%** "
                f"| {up(a, arm, 'ms'):+.1f}% "
                f"| {up(a, rnd, 'ms_disp'):+.1f}% |" if rnd in a else
                f"| {LABEL[t]} | {k} of {r['top_k']} | {f0:.2f} -> {a[arm]['fanout']:.2f} "
                f"| {a[arm]['fanout'] / f0 - 1:+.1%} | **{up(a, arm, 'ms_disp'):+.1f}%** "
                f"| {up(a, arm, 'ms'):+.1f}% | - |")
    return "\n".join(rows)


def main() -> None:
    runs = load()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.9))
    big = str(next(iter(runs.values()))["plot_batch"])

    for t, r in runs.items():
        c = COLOR[t]
        bs = sorted(int(b) for b in r["batches"])
        for k, ls, mk in ((4, "-", "o"), (2, "--", "s")):
            arm = f"placement_{k}"
            pts = [(b, up(r["batches"][str(b)], arm, "ms_disp")) for b in bs
                   if arm in r["batches"][str(b)]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=mk, color=c,
                    lw=1.8, ms=6, mec=c, mfc=c if k == 4 else "white")
        a = r["batches"][big]
        f0 = a["baseline"]["fanout"]
        for k in KEEPS:
            arm, rnd = f"placement_{k}", f"random_{k}"
            if arm not in a:
                continue
            bx.plot((f0 / a[arm]["fanout"] - 1) * 100, up(a, arm, "ms_disp"), marker="o",
                    color=c, ms=7, mec=c)
            if rnd in a:
                bx.plot((f0 / a[rnd]["fanout"] - 1) * 100, up(a, rnd, "ms_disp"), marker="x",
                        color=c, ms=7, mew=1.6)

    band = control_band(runs, big)
    ax.axhspan(-band, band, color="0.8", zorder=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1024, 4096, 16384], ["1024", "4096", "16384"])
    ax.set_xlabel("tokens per step")
    ax.set_ylabel("MoE dispatch speedup (%)")
    ax.set_title("The saving appears once dispatch is bandwidth-bound", fontsize=10)
    ax.grid(alpha=0.25)

    lim = max(1, *(abs(x) for x in bx.get_xlim()))
    bx.plot([0, lim], [0, lim], color="black", lw=0.7, ls=":")
    bx.annotate("all of the byte saving", (lim, lim), fontsize=7, ha="right", va="bottom")
    bx.axhline(0, color="black", lw=0.6)
    bx.axvline(0, color="black", lw=0.6)
    bx.set_xlabel("speedup predicted by the byte ratio alone (%)\n"
                  "mean fan-out before / mean fan-out after")
    bx.set_ylabel("measured dispatch speedup (%)")
    bx.set_title(f"...and tracks the bytes it removes  ({big} tokens/step)", fontsize=10)
    bx.grid(alpha=0.25)

    handles = [plt.Line2D([], [], color=COLOR[t], marker="o", label=LABEL[t]) for t in runs] + [
        plt.Line2D([], [], color="black", marker="o", ls="-", label="keep 4 of 8"),
        plt.Line2D([], [], color="black", marker="s", ls="--", mfc="white", label="keep 2 of 8"),
        plt.Line2D([], [], color="black", marker="x", ls="none",
                   label="random control (no fan-out cut)"),
        plt.Line2D([], [], color="0.8", lw=8, label=f"control envelope (+/-{band:.1f}%)")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.95)
    fig.suptitle("Confining the bottom experts to GPUs the token already visits, "
                 "4x H100, real NCCL all-to-all")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)

    md = table(runs)
    (RUNS / "speedup.md").write_text(
        f"Dispatch and layer speedup at {big} tokens per step, four ranks, "
        f"capacity fixed per batch so the expert GEMM is identical in every arm.\n\n{md}\n")
    print(md)
    print(f"\ncontrol envelope at {big} tokens/step: +/-{band:.2f}%")
    # The effect has to clear every arm that must not move, or the figure is measuring the
    # harness. Checked here rather than asserted in prose.
    worst = min(up(r["batches"][big], "placement_4", "ms_disp") for r in runs.values())
    assert worst > 5 * band, f"keep=4 dispatch {worst:.1f}% vs control band {band:.1f}%"
    print(f"smallest keep=4 dispatch gain {worst:.2f}% = {worst / band:.0f}x the band")
    print(f"WROTE {OUT} and {RUNS / 'speedup.md'}")


if __name__ == "__main__":
    main()
