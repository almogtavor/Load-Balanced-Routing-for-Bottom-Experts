"""CPU ownership/prior-campaign/failed-attempt guards for secondary controller."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from common_v2 import read, write_new
import causal_driver_v1 as driver


class DriverTests(unittest.TestCase):
    def prior(self, root, **fields):
        rows = [{"replicate": r, "arm": a, "ep": 4, "complete": True}
                for r in (1, 2, 3) for a in ("stock", "keep6p8", "keep4p8")]
        value = {"complete": True, "execution_complete": True, "cleanup_complete": True, "cleanup_errors": [], "launches": rows, **fields}
        path = root / "quality.json"
        write_new(path, value)
        return path

    def test_prior_nine_complete_and_original_controller_exited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prior(Path(directory))
            lib = SimpleNamespace(process_record=lambda pid: None)
            self.assertTrue(driver.prior_gate(path, lib, {"pid": 167, "start_ticks": "5555001"})["complete"])
            lib.process_record = lambda pid: {"pid": 167, "start_ticks": "5555001", "state": "R"}
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                driver.prior_gate(path, lib, {"pid": 167, "start_ticks": "5555001"})

    def test_prior_cleanup_or_cell_missing_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.prior(Path(directory), cleanup_complete=False)
            with self.assertRaises(RuntimeError):
                driver.prior_gate(path, SimpleNamespace(process_record=lambda pid: None), {"pid": 167, "start_ticks": "5555001"})

    def test_failed_attempt_has_immutable_receipt_with_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            instance = object.__new__(driver.CausalDriver)
            instance.root, instance.path, instance.mode = Path(directory), Path(directory) / "plan.json", "training"
            write_new(instance.path, {})
            instance.state, instance.plan = {"launches": []}, {}
            instance.save, instance.validate = lambda: None, lambda: None
            instance.start_server = lambda *args: (_ for _ in ()).throw(RuntimeError("native startup failure"))
            instance.close_pair = lambda: {"owned_groups_closed": True}
            with self.assertRaises(RuntimeError):
                instance.launch_cell(0, "stock", 16384)
            receipt = read(instance.root / "cell00-stock-budget16384" / "launch.json")
            self.assertFalse(receipt["complete"])
            self.assertIn("native startup failure", receipt["workload_error"])
            self.assertFalse(instance.state["launches"][0]["complete"])

    def test_heldout_freeze_refuses_active_model_before_any_query(self):
        instance = object.__new__(driver.CausalDriver)
        instance.validate = lambda: None
        instance.training_gate = lambda: (Path("training-receipt"), [])
        instance.children = {"causal": object()}
        with self.assertRaisesRegex(RuntimeError, "No model may be active"):
            instance.freeze_heldout()


if __name__ == "__main__":
    unittest.main()
