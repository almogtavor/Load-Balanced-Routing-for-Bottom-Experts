#!/usr/bin/env python3
"""Format a complete accepted report into tables; no statistical computation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ARMS = ("stock", "keep6p8", "keep4p8")
ENDPOINTS = ("prefix_raw", "prefix_norm", "teacher_raw", "teacher_norm")


def number(value, decimals=2):
    assert math.isfinite(value)
    return f"{value:+.{decimals}f}"


def score(value):
    assert math.isfinite(value)
    if value == 0:
        return "0"
    if abs(value) >= 0.001:
        return f"{value:+.4f}"
    mantissa, exponent = f"{value:+.2e}".split("e")
    return mantissa + r"\!\times\!10^{" + str(int(exponent)) + "}"


def contrast_rows(report):
    rows = []
    for arm, label in (("keep6p8", "K6--stock"), ("keep4p8", "K4--stock")):
        for endpoint in ENDPOINTS:
            policy, convention = endpoint.split("_")
            key = arm + "-minus-stock/" + endpoint
            rows.append((key, label, ("P" if policy == "prefix" else "T") + "/" + convention))
    for arm, label in (("stock", "Stock: T--P"), ("keep6p8", "K6: T--P"), ("keep4p8", "K4: T--P")):
        for convention in ("norm", "raw"):
            key = arm + "/teacher-minus-prefix/" + convention
            rows.append((key, label, convention))
    assert len(rows) == 14 and {r[0] for r in rows} == set(report["contrasts"])
    return rows


def render(report):
    assert report["complete"] is True and report["schema"] == "fresh-causal-paired-report-v1"
    assert report["n"] == 256 and set(report["per_arm"]) == set(ARMS)
    rows = contrast_rows(report)
    text = [r"% Presentation only: values copied from the accepted frozen statistics report.",
            r"\begin{table*}[t]", r"\centering\small",
            r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
            r"Router & Prefix raw & Prefix normalized & Teacher raw & Teacher normalized \\", r"\midrule"]
    cells = []
    for arm, label in zip(ARMS, ("Stock", "Keep=6", "Keep=4")):
        assert set(report["per_arm"][arm]) == set(ENDPOINTS)
        values = []
        for endpoint in ENDPOINTS:
            item = report["per_arm"][arm][endpoint]
            assert item["n"] == 256 and type(item["correct"]) is int
            assert math.isclose(item["accuracy_percent"], 100 * item["correct"] / 256, abs_tol=1e-12)
            value = f'{item["correct"]} ({item["accuracy_percent"]:.2f})'
            values.append(value)
            cells.append({"table": "arm_accuracy", "arm": arm, "endpoint": endpoint, "display": value})
        text.append(label + " & " + " & ".join(values) + r" \\")
    text += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Chosen-answer accuracy on the fixed 256-item HellaSwag subset. Cells show correct count and percent; every arm uses all 256 items. Prefix scores use serial full-prefix re-prefills containing only positions preceding the scored token. Teacher scores retain the separate candidate-conditioned diagnostic. Normalization divides by the canonical ending's character length.}",
             r"\label{tab:followup-causal-accuracy-20261018}", r"\end{table*}", "",
             r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{6pt}",
             r"\begin{tabular}{@{}llcrr@{}}", r"\toprule",
             r"Contrast & Policy/scoring & $\Delta$ accuracy pp [95\% item interval] & Gain/loss & Answer flips \\",
             r"\midrule"]
    for key, label, policy in rows:
        item = report["contrasts"][key]["accuracy"]
        assert item["n"] == 256
        interval = item["conditional_item_sensitivity_95"]
        displayed = number(item["difference_pp"]) + r"\;[" + number(interval["lower"]) + "," + number(interval["upper"]) + "]"
        gainloss = str(item["paired_gain_items"]) + "/" + str(item["paired_loss_items"])
        flips = str(item["changed_chosen_answer_items"])
        text.append(f"{label} & {policy} & ${displayed}$ & {gainloss} & {flips}" + r" \\")
        cells.append({"table": "accuracy_contrasts", "contrast": key, "effect_interval": displayed, "gain_loss": gainloss, "flips": flips})
    text += [r"\bottomrule", r"\end{tabular}",
             r"\caption{All 14 prespecified accuracy contrasts and changed chosen-answer counts. K6/K4 denote keep=6/4 with $\lambda=8$; P is serial prefix and T is teacher scoring. Norm is character normalization. Intervals use 10,000 common paired item resamples (seed zero, nearest-rank 2.5/97.5 percentiles). They describe conditional item sensitivity from one launch per arm; they are unadjusted, descriptive, and do not estimate launch/host variation or establish equivalence.}",
             r"\label{tab:followup-causal-contrasts-20261018}", r"\end{table*}", "",
             r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{8pt}",
             r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Contrast & Policy/scoring & Mean $\Delta$ candidate log-score & 95\% item interval \\", r"\midrule"]
    for key, label, policy in rows:
        item = report["contrasts"][key]["candidate_score_difference"]
        assert item["n"] == 256
        interval = item["conditional_item_sensitivity_95"]
        mean = score(item["mean"])
        displayed = "[" + score(interval["lower"]) + "," + score(interval["upper"]) + "]"
        text.append(f"{label} & {policy} & ${mean}$ & ${displayed}$" + r" \\")
        cells.append({"table": "score_contrasts", "contrast": key, "effect": mean, "interval": displayed})
    text += [r"\bottomrule", r"\end{tabular}",
             r"\caption{All prespecified candidate log-score contrasts, with the same definitions and conditional item intervals as Table~\ref{tab:followup-causal-contrasts-20261018}. Raw units are nats per ending; norm units are nats per normalization character. Each contrast averages the four candidate shifts within each item, then across the 256 paired items. A positive log-score shift does not by itself imply improved chosen-answer accuracy. Teacher-minus-prefix shifts include cohort packing and numerical execution; stock is the required control.}",
             r"\label{tab:followup-causal-scores-20261018}", r"\end{table*}"]
    return "\n".join(text) + "\n", cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.report_sha256
    table, cells = render(json.loads(raw))
    with args.output.open("x") as target:
        target.write(table)
    manifest = {"scope": "Presentation only; no new estimates or model rows", "report_sha256": args.report_sha256,
                "formatter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "tex_sha256": hashlib.sha256(table.encode()).hexdigest(), "cells": cells}
    with args.manifest.open("x") as target:
        json.dump(manifest, target, indent=2, sort_keys=True)
        target.write("\n")
