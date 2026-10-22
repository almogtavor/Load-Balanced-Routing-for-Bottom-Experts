"""What the load term buys over the plain rule at the same `keep`, paired within launch.

    PYTHONPATH=experiments python experiments/lbgain.py

Both settings ran in the same launch, so `plain / balanced - 1` cancels the launch-to-launch
drift the floor in `moeplot.py` has to absorb. Mean, s.e. and a two-sided one-sample t-test
over launches; a launch count of one gets no test. Writes `runs/lbgain.md`.
"""

from __future__ import annotations

import importlib
import os
import statistics as st

from scipy.stats import ttest_1samp

PHASES = {"disp": "dispatch", "comb": "combine", "moe": "MoE layer"}
KEEPS = ("6", "4")


def gains(phase: str) -> dict:
    """`{tag: {keep: [per-launch % faster with the load term]}}`."""
    os.environ["PHASE"] = phase
    M = importlib.reload(importlib.import_module("moeplot"))
    out = {t: {k: [] for k in KEEPS} for t in M.LABEL}
    for (tag, _), settings in M.load().items():
        for k in KEEPS:
            if tag in out and k in settings and k + "p1" in settings:
                a, b = settings[k], settings[k + "p1"]
                if shared := set(a) & set(b):
                    x = max(shared)
                    out[tag][k].append((a[x] / b[x] - 1) * 100)
    return out, M.LABEL


def cell(v: list[float]) -> str:
    if len(v) < 2:
        return f"{v[0]:+.1f} x1" if v else "-"
    p = ttest_1samp(v, 0).pvalue
    s = f"{st.mean(v):+.1f} +/-{st.stdev(v) / len(v) ** .5:.1f} x{len(v)}"
    return f"**{s}**" if p < 0.05 and st.mean(v) > 0 else s


def main() -> None:
    res = {ph: gains(ph) for ph in PHASES}
    label = res["moe"][1]
    head = " | ".join(f"{n} keep={k} ↑" for n in PHASES.values() for k in KEEPS)
    rows = [f"| model | {head} |", "|---" * (1 + len(PHASES) * len(KEEPS)) + "|"]
    for t in label:
        rows.append(f"| {label[t]} | " + " | ".join(
            cell(res[ph][0][t][k]) for ph in PHASES for k in KEEPS) + " |")
    md = "\n".join(rows)
    open("runs/lbgain.md", "w").write(
        "% faster with the load term than the plain rule at the same keep, paired within "
        "launch: mean +/- s.e., x launches. Bold: faster at p < 0.05, two-sided t-test over launches.\n\n"
        f"{md}\n")
    print(md)


if __name__ == "__main__":
    main()
