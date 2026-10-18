"""Stdlib-only synthetic checks for the quality result report."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import analyze_quality as quality


def payload():
    arms = {}
    for name in quality.REQUIRED_ARMS:
        arms[name] = {"arm": "baseline" if name.startswith("baseline") else
                      "random" if name.startswith("random") else "placement",
                      "nll": 2.0, "dnll": 0.002, "dnll_se": 0.001,
                      "dnll_ci95": [0.001, 0.003], "fanout_ratio": 0.9,
                      "changed_frac": 0.0 if name == "noop" else 0.2,
                      "production_parity_by_layer": {}}
    arms["placement_6p1"]["production_parity_by_layer"] = {
        "0": {"rows": 128, "selection_mismatch_rows": 1,
              "weight_max_abs_on_matching_selection": 0.001},
        "1": {"rows": 64, "selection_mismatch_rows": 2,
              "weight_max_abs_on_matching_selection": 0.003},
    }
    return {"tag": "synthetic_ep8", "model": "synthetic", "logical_ep_size": 8,
            "visible_cuda_devices": 2, "complete": True, "arms": arms,
            "validation": {"passed": True, "noop_selection_exact": True,
                           "noop_nll_within_resolution": True, "plain_placement_monotone": True,
                           "interventions_fired": {"placement_6": True,
                                                   "placement_6p1": True, "random_6": True}}}


class QualityAnalysisTests(unittest.TestCase):
    def test_complete_uses_existing_ci_and_sums_parity(self):
        run = quality.summarize(payload(), Path("acc_test.json"))
        self.assertEqual(run["status"], "complete")
        self.assertEqual(run["logical_ep_size"], 8)
        self.assertEqual(run["visible_cuda_devices"], 2)
        self.assertEqual(run["arms"]["placement_6p1"]["dnll_ci95"], [0.001, 0.003])
        parity = run["arms"]["placement_6p1"]["production_parity"]
        self.assertEqual(parity["rows"], 192)
        self.assertEqual(parity["selection_mismatch_rows"], 3)
        self.assertEqual(parity["status"], "selection_mismatches")
        self.assertEqual(parity["weight_max_abs_on_matching_selection"], 0.003)
        self.assertEqual(run["arms"]["random_6"]["production_parity"]["status"], "unsampled")

    def test_failed_control_is_not_overridden_by_declared_pass(self):
        data = payload()
        data["validation"]["interventions_fired"]["random_6"] = False
        run = quality.summarize(data, "synthetic")
        self.assertEqual(run["controls"]["status"], "failed")
        self.assertEqual(run["status"], "complete_controls_failed")

    def test_incomplete_keeps_missing_arms_and_unknown_controls(self):
        data = payload()
        data["complete"] = False
        del data["validation"]
        del data["arms"]["random_6"]
        run = quality.summarize(data, "synthetic")
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual(run["missing_arms"], ["random_6"])
        self.assertEqual(run["arms"]["random_6"]["status"], "missing")
        self.assertEqual(run["controls"]["status"], "unknown")
        self.assertEqual(run["noop"]["status"], "unknown")
        self.assertEqual(run["repeat"]["pass_status"], "not_defined")

    def test_missing_validation_check_is_unknown(self):
        data = payload()
        del data["validation"]["noop_nll_within_resolution"]
        run = quality.summarize(data, "synthetic")
        self.assertEqual(run["status"], "complete_controls_unknown")
        self.assertEqual(run["noop"]["status"], "unknown")

    def test_all_mismatched_rows_have_no_measured_weight_error(self):
        arm = {"production_parity_by_layer": {
            "0": {"rows": 2, "selection_mismatch_rows": 2,
                  "weight_max_abs_on_matching_selection": 0.0}}}
        self.assertIsNone(quality.parity_summary(arm)["weight_max_abs_on_matching_selection"])

    def test_cli_writes_reports_from_temporary_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "quality"
            input_path.mkdir()
            (input_path / "acc_test.json").write_text(json.dumps(payload()))
            partial = copy.deepcopy(payload())
            partial["complete"] = False
            del partial["arms"]["placement_6p1"]
            (input_path / "acc_partial.json").write_text(json.dumps(partial))
            self.assertEqual(quality.main([str(input_path)]), 0)
            result = json.loads((root / "quality_analysis.json").read_text())
            self.assertEqual(len(result["runs"]), 2)
            text = (root / "quality_analysis.md").read_text()
            self.assertIn("balanced6p1 (missing)", text)
            self.assertIn("3 / 192", text)
            self.assertIn("do not measure distributed dispatch", text)

    def test_malformed_and_nonfinite_input_stays_a_read_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "acc_bad.json"
            path.write_text('{"arms": {}, "bad": NaN}')
            report = quality.analyze(path)
            self.assertEqual(len(report["errors"]), 1)
            self.assertFalse(report["runs"])


if __name__ == "__main__":
    unittest.main()
