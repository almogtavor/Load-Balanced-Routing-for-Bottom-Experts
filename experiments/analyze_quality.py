#!/usr/bin/env python3
"""Summarize completed or checkpointed accuracy.py runs, using only the stdlib.

  python3 experiments/analyze_quality.py runs/waldorf-20261017/quality

Writes quality_analysis.json and quality_analysis.md beside the input directory.
Stored paired-window confidence intervals are retained; nothing is re-estimated.
Logical EP describes the simulated contiguous expert-to-rank map for quality, not
distributed execution or a throughput measurement. Missing controls stay unknown.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


DEFAULT_INPUT = Path("runs/waldorf-20261017/quality")
REQUIRED_ARMS = ("baseline", "baseline2", "noop", "placement_6", "placement_6p1", "random_6")
ARM_LABELS = {"baseline": "baseline", "baseline2": "repeat", "noop": "no-op",
              "placement_6": "plain6", "placement_6p1": "balanced6p1", "random_6": "random6"}
SCOPE_NOTE = (
    "Quality-only model inference. Logical EP=4/8 means expert IDs are mapped to "
    "4/8 simulated contiguous ranks using floor(expert_id * logical_EP / n_experts). "
    "When visible CUDA devices=2, those logical ranks do not mean 4/8 physical GPUs: "
    "the model is placed across the two visible devices by its physical device map. "
    "Fan-out is a routing statistic; these runs do not measure distributed dispatch "
    "or serving throughput."
)
PARITY_NOTE = (
    "Parity is sampled reference-versus-production selector agreement, separate from "
    "the harness quality controls. Weight errors are measured only on matching "
    "selections; mismatched rows are excluded. Missing or zero-row samples are "
    "unsampled, not passed. No pass threshold for repeat NLL or weight error is "
    "invented by this report."
)


def status(value):
    return "passed" if value is True else "failed" if value is False else "unknown"


def parity_summary(arm):
    layers = arm.get("production_parity_by_layer")
    if not isinstance(layers, dict) or not layers:
        return {"status": "unsampled", "layers": 0, "rows": 0,
                "selection_mismatch_rows": None,
                "weight_max_abs_on_matching_selection": None}
    required = ("rows", "selection_mismatch_rows", "weight_max_abs_on_matching_selection")
    valid = all(isinstance(record, dict) and all(k in record for k in required)
                for record in layers.values())
    if not valid:
        return {"status": "incomplete", "layers": len(layers), "rows": None,
                "selection_mismatch_rows": None,
                "weight_max_abs_on_matching_selection": None}
    rows = sum(r["rows"] for r in layers.values())
    mismatches = sum(r["selection_mismatch_rows"] for r in layers.values())
    matching = rows - mismatches
    return {
        "status": ("unsampled" if not rows else "selection_mismatches" if mismatches
                   else "selection_exact"),
        "layers": len(layers), "rows": rows, "selection_mismatch_rows": mismatches,
        "matching_selection_rows": matching,
        "weight_max_abs_on_matching_selection": (
            max(r["weight_max_abs_on_matching_selection"] for r in layers.values()
                if r["rows"] > r["selection_mismatch_rows"]) if matching else None),
    }


def summarize(payload, source):
    arms = payload.get("arms", {})
    if not isinstance(arms, dict):
        raise ValueError("arms must be an object")
    validation = payload.get("validation") or {}
    expected = list(dict.fromkeys([*REQUIRED_ARMS, *payload.get("arm_order", [])]))
    missing = [name for name in expected if name not in arms]
    interventions = validation.get("interventions_fired")
    checks = {key: status(validation.get(key)) for key in (
        "noop_selection_exact", "noop_nll_within_resolution", "plain_placement_monotone")}
    checks["interventions_fired"] = (
        {name: status(interventions.get(name)) for name, arm in arms.items()
         if arm.get("arm") in ("placement", "random") and name != "noop"}
        if isinstance(interventions, dict) else None)
    required_check_states = [checks[k] for k in (
        "noop_selection_exact", "noop_nll_within_resolution", "plain_placement_monotone")]
    required_check_states += list((checks["interventions_fired"] or {}).values())
    declared = status(validation.get("passed"))
    if declared == "failed" or "failed" in required_check_states:
        control_status = "failed"
    elif (declared == "passed" and not missing and checks["interventions_fired"]
          and all(x == "passed" for x in required_check_states)):
        control_status = "passed"
    else:
        control_status = "unknown"
    complete = payload.get("complete") is True
    if missing or not complete:
        run_status = "incomplete"
    elif control_status == "failed":
        run_status = "complete_controls_failed"
    elif control_status == "unknown":
        run_status = "complete_controls_unknown"
    else:
        run_status = "complete"
    summary_arms = {}
    for name in dict.fromkeys([*expected, *arms]):
        arm = arms.get(name)
        if arm is None:
            summary_arms[name] = {"label": ARM_LABELS.get(name, name), "status": "missing"}
            continue
        summary_arms[name] = {
            "label": ARM_LABELS.get(name, name), "status": "present",
            **{k: arm.get(k) for k in ("nll", "dnll", "dnll_se", "dnll_ci95", "fanout",
                                      "fanout_ratio", "changed_frac", "router_rows")},
            "production_parity": parity_summary(arm),
        }
    repeat = arms.get("baseline2")
    return {
        "source": str(source), "tag": payload.get("tag"), "model": payload.get("model"),
        "logical_ep_size": payload.get("logical_ep_size", payload.get("n_gpus")),
        "visible_cuda_devices": payload.get("visible_cuda_devices"),
        "physical_device_map": payload.get("physical_device_map"),
        "n_experts": payload.get("n_experts"), "n_windows": payload.get("n_windows"),
        "seq_len": payload.get("seq_len"), "scored_tokens_per_arm": payload.get("scored_tokens_per_arm"),
        "input_token_sha256": payload.get("data", {}).get("input_token_sha256"),
        "ci_method": payload.get("ci_method"), "bootstrap_samples": payload.get("bootstrap_samples"),
        "complete": complete, "status": run_status, "missing_arms": missing,
        "baseline_nll": arms.get("baseline", {}).get("nll"),
        "repeat": {"status": "measured" if repeat else "missing", "pass_status": "not_defined",
                   "noise_floor": payload.get("noise_floor") or (
                       {k: repeat.get(k) for k in ("dnll", "dnll_se", "dnll_ci95", "t")}
                       if repeat else None)},
        "noop": {"status": ("passed" if checks["noop_selection_exact"] == "passed"
                            and checks["noop_nll_within_resolution"] == "passed"
                            else "failed" if "failed" in (checks["noop_selection_exact"],
                                                           checks["noop_nll_within_resolution"])
                            else "unknown"),
                 "vs_repeat": validation.get("noop_vs_repeat")},
        "controls": {"status": control_status, "declared_passed": validation.get("passed"),
                     "checks": checks},
        "arms": summary_arms,
    }


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.7g}"
    if isinstance(value, list):
        return "[" + ", ".join(display(x) for x in value) + "]"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    lines = ["# Quality analysis", "", SCOPE_NOTE, "", PARITY_NOTE, "",
             f"Input: `{report['input']}`. Runs: {len(report['runs'])}. "
             f"Read errors: {len(report['errors'])}.", ""]
    if not report["runs"]:
        lines += ["No accuracy checkpoints found; no experiments are claimed complete.", ""]
    for run in report["runs"]:
        lines += [f"## {display(run['tag'] or Path(run['source']).stem)}", "",
                  f"Model: `{display(run['model'])}`. Logical EP: {display(run['logical_ep_size'])}; "
                  f"visible physical GPUs: {display(run['visible_cuda_devices'])}. "
                  f"Status: **{run['status']}**. Controls: **{run['controls']['status']}**; "
                  f"no-op: **{run['noop']['status']}**; repeat: **{run['repeat']['status']}** "
                  "(pass criterion not defined by harness).", "",
                  f"Baseline NLL: **{display(run['baseline_nll'])}**. "
                  f"Windows: {display(run['n_windows'])} × {display(run['seq_len'])}; "
                  f"scored tokens/arm: {display(run['scored_tokens_per_arm'])}. "
                  f"CI: {display(run['ci_method'])}.", ""]
        if run["missing_arms"]:
            lines += ["Missing arms: " + ", ".join(f"`{a}`" for a in run["missing_arms"]) + ".", ""]
        lines += ["| Arm | NLL | Paired ΔNLL | Existing 95% CI | Fan-out ratio | Changed fraction | Selector mismatches / sampled rows | Max weight error on matches |",
                  "|---|---:|---:|---|---:|---:|---:|---:|"]
        for arm in run["arms"].values():
            if arm["status"] == "missing":
                lines.append(f"| {arm['label']} (missing) | — | — | — | — | — | — | — |")
                continue
            parity = arm["production_parity"]
            pair = (f"{display(parity['selection_mismatch_rows'])} / {display(parity['rows'])}"
                    if parity["status"] not in ("unsampled", "incomplete") else parity["status"])
            values = [arm["label"], *[display(arm[k]) for k in (
                "nll", "dnll", "dnll_ci95", "fanout_ratio", "changed_frac")],
                      pair, display(parity["weight_max_abs_on_matching_selection"])]
            lines.append("| " + " | ".join(values) + " |")
        lines += ["", f"Source: `{run['source']}`.", ""]
    if report["errors"]:
        lines += ["## Read errors", ""]
        lines += [f"- `{item['source']}`: {display(item['error'])}" for item in report["errors"]]
        lines += [""]
    return "\n".join(lines)


def reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def analyze(input_path):
    paths = ([input_path] if input_path.is_file() else
             sorted(input_path.rglob("acc_*.json")) if input_path.is_dir() else [])
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "input": str(input_path),
              "scope": SCOPE_NOTE, "parity_note": PARITY_NOTE, "runs": [], "errors": []}
    if not input_path.exists():
        report["errors"].append({"source": str(input_path), "error": "input does not exist"})
    for path in paths:
        try:
            payload = json.loads(path.read_text(), parse_constant=reject_constant)
            if not isinstance(payload, dict):
                raise ValueError("accuracy payload must be an object")
            report["runs"].append(summarize(payload, path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report["errors"].append({"source": str(path), "error": str(exc)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT,
                        help="directory of acc_*.json checkpoints, or one accuracy JSON")
    parser.add_argument("--output-prefix", type=Path,
                        help="output path without suffix (default: input parent/quality_analysis)")
    args = parser.parse_args(argv)
    prefix = args.output_prefix or args.input.parent / "quality_analysis"
    report = analyze(args.input)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = prefix.with_suffix(".json"), prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    md_path.write_text(markdown(report))
    print(f"{len(report['runs'])} runs; {len(report['errors'])} read errors; "
          f"wrote {json_path} and {md_path}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
