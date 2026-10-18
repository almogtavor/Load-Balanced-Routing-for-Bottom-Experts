"""CPU-only fresh-owner/reference-policy fixtures; native GPU hashes mocked only in tests."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import causal_driver_v1 as old
import causal_driver_v3 as new
import fresh_owner as guard
from common_v2 import file_sha, read, write_new

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT.parent / "prior-inputs"
ARCHIVE = ROOT.parents[2] / "paper-followup-20261018/remote-capture/quality-final/archive-captures/20261018T104441Z-e991bdb1ff22/extracted-20261018T110353Z-29b9a973d411/files"


class FreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = {"namespace": "itay", "pod": guard.POD, "uid": "11111111-2222-4333-8444-555555555555",
                         "node": "synthetic-h100-node", "image": guard.IMAGE}
        self.allocation = {**self.identity, "verified_live": True, "gpu_count": 4, "gpu_name": "NVIDIA H100 80GB HBM3",
                           "gpu_inventory": [f"{i}, GPU-test-{i}, NVIDIA H100 80GB HBM3, 81559 MiB" for i in range(4)]}
        self.ap = self.root / "allocation.json"
        self.sp = self.root / "source.json"
        self.op = self.root / "owner.json"
        self.native = INPUTS / "native-runtime19.json"
        self.native_files = read(self.native)["files"]
        self.source = {"pod_identity": self.identity, "source_commit": new.SOURCE_COMMIT}
        self.save(self.ap, self.allocation)
        self.save(self.sp, self.source)
        self.owner = {"schema": "fresh-four-h100-causal-owner-v1", "complete": True,
                      "campaign": guard.CAMPAIGN, "old_partial_results_reused": False,
                      "pod_identity": self.identity, "allocation_proof_sha256": file_sha(self.ap),
                      "source_proof_sha256": file_sha(self.sp), "native_manifest_sha256": guard.NATIVE_MANIFEST_SHA,
                      "prior_inputs": {"quality_receipt": str(INPUTS / "prior-quality9.json"),
                                       "archive_marker": str(INPUTS / "prior-archive-FINAL_VERIFIED.json"),
                                       "release_receipt": str(INPUTS / "prior-release-quality.json")}}
        self.save(self.op, self.owner)

    def save(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True) + "\n")

    def actual(self, path):
        return self.native_files.get(str(path), None) or file_sha(path)

    def gate(self):
        with patch.object(guard, "file_sha", side_effect=self.actual):
            return guard.fresh_owner_gate(self.op, self.ap, self.sp, self.native, INPUTS / "prior-quality9.json")

    def test_actual_prior_closure_and_nineteen_file_manifest(self):
        self.assertEqual(self.gate(), self.native_files)

    def test_old_linux_pid_namespace_is_never_queried(self):
        lib = SimpleNamespace(process_record=lambda pid: self.fail("Old pod PID queried in fresh namespace"))
        self.assertTrue(guard.prior_gate(INPUTS / "prior-quality9.json", lib, {"owner_proof": str(self.op)})["complete"])

    def test_wrong_current_owner_or_prior_reuse_rejected(self):
        for fields in ({"campaign": "other"}, {"old_partial_results_reused": True}, {"complete": False}):
            with self.subTest(fields=fields):
                self.save(self.op, {**self.owner, **fields})
                with self.assertRaises(RuntimeError): self.gate()

    def test_old_uid_wrong_image_namespace_or_pod_rejected(self):
        for field, value in (("uid", guard.OLD_UID), ("image", "other-image"), ("namespace", "default"), ("pod", "other")):
            with self.subTest(field=field):
                identity = {**self.identity, field: value}
                self.save(self.op, {**self.owner, "pod_identity": identity})
                with self.assertRaises(RuntimeError): self.gate()

    def test_eight_gpu_or_duplicate_inventory_rejected(self):
        for allocation in ({**self.allocation, "gpu_count": 8},
                           {**self.allocation, "gpu_inventory": [self.allocation["gpu_inventory"][0]] * 4}):
            self.save(self.ap, allocation)
            self.save(self.op, {**self.owner, "allocation_proof_sha256": file_sha(self.ap)})
            with self.assertRaises(RuntimeError): self.gate()

    def test_current_source_owner_or_proof_byte_tamper_rejected(self):
        self.save(self.sp, {**self.source, "pod_identity": {**self.identity, "node": "wrong"}})
        with self.assertRaises(RuntimeError): self.gate()

    def test_missing_or_changed_native_source_rejected(self):
        bad = next(iter(self.native_files))
        with patch.object(guard, "file_sha", side_effect=lambda p: "0" * 64 if str(p) == bad else self.actual(p)):
            with self.assertRaisesRegex(RuntimeError, "native source differs"):
                guard.fresh_owner_gate(self.op, self.ap, self.sp, self.native, INPUTS / "prior-quality9.json")

    def test_changed_archived_quality_bytes_rejected(self):
        p = self.root / "changed-quality.json"
        p.write_bytes((INPUTS / "prior-quality9.json").read_bytes() + b" ")
        self.save(self.op, {**self.owner, "prior_inputs": {**self.owner["prior_inputs"], "quality_receipt": str(p)}})
        with self.assertRaisesRegex(RuntimeError, "immutable provenance changed"):
            guard.prior_gate(p, None, {"owner_proof": str(self.op)})

    def test_real_frozen256_subset_and_request_counts(self):
        prepared = ARCHIVE / "causal-v2/heldout-prepared"
        guard.selected_gate(prepared)
        wrong = self.root / "prepared"
        wrong.mkdir()
        (wrong / "selected-requests.json").write_bytes((prepared / "selected-requests.json").read_bytes())
        value = read(prepared / "protocol.json")
        for field, changed in (("total_requests", 29614), ("prefix_concurrency", 16), ("items", 255)):
            self.save(wrong / "protocol.json", {**value, field: changed})
            with self.assertRaises(RuntimeError): guard.selected_gate(wrong)

    def test_server_and_scoring_lifecycle_functions_unchanged(self):
        a = ast.parse((ROOT / "causal_driver_v1.py").read_text())
        b = ast.parse((ROOT / "causal_driver_v3.py").read_text())
        def fn(tree, name):
            return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        for name in ("command", "start_server", "training_gate", "launch_cell", "run", "validate"):
            self.assertEqual(ast.dump(fn(a, name)), ast.dump(fn(b, name)), name)
        for budget in (1024, 16384):
            plan = {"vllm": "/usr/local/bin/vllm", "model": "/model"}
            self.assertEqual(old.command(plan, budget), new.command(plan, budget))

    def test_actual_plan_constructor_and_training_preparation_cpu_integration(self):
        checkpoint = self.root / "checkpoint.json"
        checkpoint.write_bytes((ARCHIVE / "checkpoint-proof.json").read_bytes())
        lifecycle = ROOT.parents[2] / "paper-followup-20261018/performance/prefill_e2e.py"
        vllm = self.root / "vllm"
        vllm.write_bytes(b"synthetic executable; no GPU runtime invoked\n")
        args = SimpleNamespace(owner_proof=self.op, native_manifest=self.native, quality_receipt=INPUTS / "prior-quality9.json",
            allocation_proof=self.ap, source_proof=self.sp, checkpoint_proof=checkpoint,
            production_root=ARCHIVE / "provenance/final-inputs-v1/task-router", site_root=self.root / "site",
            preflight_inputs=INPUTS / "preflight.json", full_inputs=INPUTS / "requests.json", lifecycle_helper=lifecycle,
            output=self.root / "campaign", model=self.root / "model", vllm=vllm,
            heldout_helper=ROOT / "heldout_causal_v2.py", heldout_review=ROOT / "heldout-causal-v2-independent-review-v1.json",
            stop_unix=time.time() + 18000)
        lib = SimpleNamespace(source_identity=lambda *a: {"source": "CPU integration fixture"},
                              checkpoint_identity=lambda *a: {"model": "GPU weights not opened"})
        with patch.object(new.hybrid, "lifecycle", return_value=lib), patch.object(guard, "file_sha", side_effect=self.actual):
            new.make_plan(args)
        plan = read(args.output / "plan.json")
        self.assertEqual(plan["schema"], "fresh-four-h100-causal-v3")
        self.assertEqual(plan["allocation"]["gpu_count"], 4)
        self.assertEqual(plan["training_requests_per_cell"], 493)
        self.assertEqual(plan["training_cells"], [[a, b] for a, b in new.TRAINING_CELLS])
        self.assertTrue(set(self.native_files) <= set(plan["files"]))
        self.assertEqual(read(args.output / "training-prepared/protocol.json")["independent_training_items"], 4)


if __name__ == "__main__":
    unittest.main()
