"""Every setting of one tag against a reference setting, paired, from `runs/`.

    python experiments/paired.py TAG [REF]                 # REF defaults to base

Throughput and wall-clock, from `serve_<tag>_<setting>_r<rep>.json`: each launch is the median
of its timed batches, and the change is the mean log ratio against REF's launch in the same
sweep, with a 95% t-interval over sweeps. A launch is the unit because two launches of the same
engine differ by more than the batches inside one launch do.

Accuracy, from `eval_<tag>_<setting>_r<rep>/`: each question's score averaged over runs (GSM8K
strict match, HellaSwag length-normalized, MMLU over all subjects), then the mean difference
against REF in points and its standard error, paired by question.
"""

from __future__ import annotations

import glob
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict

# Two-sided 95% t quantiles by degrees of freedom.
# ponytail: table to 15, normal quantile beyond; use scipy.stats.t if runs grow past that.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131}
METRIC = {"gsm8k": ("exact_match", "strict-match"), "hellaswag": ("acc_norm", None),
          "mmlu": ("acc", None)}


def interval(logs: list[float]) -> str:
    m = st.mean(logs)
    h = T975.get(len(logs) - 1, 1.96) * st.stdev(logs) / math.sqrt(len(logs))
    pct = lambda x: f"{100 * math.expm1(x):+.1f}"
    return f"{pct(m)}% [{pct(m - h)}, {pct(m + h)}]"


def throughput(tag: str, ref: str) -> None:
    med = {}  # (setting, rep) -> (tok/s, seconds per batch)
    for f in glob.glob(f"runs/serve_{tag}_*_r*.json"):
        setting, rep = re.fullmatch(rf"serve_{re.escape(tag)}_(.+)_r(\d+)\.json", f.split("/")[-1]).groups()
        tp = json.load(open(f))["throughput"]
        med[setting, rep] = (st.median(r["tot_tok_per_s"] for r in tp), st.median(r["seconds"] for r in tp))
    for s in sorted({s for s, _ in med} - {ref}):
        reps = [r for x, r in med if x == s and (ref, r) in med]
        if len(reps) < 2:
            continue
        tp, wall = ([math.log(med[s, r][i] / med[ref, r][i]) for r in reps] for i in (0, 1))
        print(f"{s:40s} throughput {interval(tp)}  wall-clock {interval(wall)}  n={len(reps)}")


def accuracy(tag: str, ref: str) -> None:
    q = defaultdict(lambda: defaultdict(list))  # (setting, bench) -> question -> scores over runs
    for f in glob.glob(f"runs/eval_{tag}_*_r*/**/samples_*.jsonl", recursive=True):
        setting = re.fullmatch(rf"eval_{re.escape(tag)}_(.+)_r\d+", f.split("/")[1]).group(1)
        task = re.search(r"samples_(.+?)_\d{4}-\d\d-\d\dT", f).group(1)
        bench = "mmlu" if task.startswith("mmlu") else task
        metric, filt = METRIC[bench]
        for line in open(f):
            x = json.loads(line)
            if not filt or x.get("filter") == filt:
                q[setting, bench][task, x["doc_id"]].append(float(x[metric]))
    for s in sorted({s for s, _ in q}, key=lambda s: (s != ref, s)):
        row = []
        for b in METRIC:
            a, base = q.get((s, b)), q.get((ref, b))
            if not a or not base:
                continue
            if s == ref:
                row.append(f"{b} {100 * st.mean(st.mean(v) for v in a.values()):.1f} (n={len(a)})")
                continue
            d = [st.mean(a[k]) - st.mean(base[k]) for k in a.keys() & base.keys()]
            row.append(f"{b} {100 * st.mean(d):+.1f}+-{100 * st.stdev(d) / math.sqrt(len(d)):.1f} (n={len(d)})")
        print(f"{s:40s} " + " | ".join(row))


if __name__ == "__main__":
    tag, ref = sys.argv[1], (sys.argv[2:] or ["base"])[0]
    throughput(tag, ref)
    accuracy(tag, ref)
