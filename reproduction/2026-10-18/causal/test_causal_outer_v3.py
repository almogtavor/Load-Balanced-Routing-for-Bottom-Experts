"""CPU request/lease/admission/complete-stage/failure-retention integration."""
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import causal_outer_v3 as outer
import fresh_owner as guard
import test_fresh_causal_v3 as fixtures
from common_v2 import file_sha, read, write_new


class OuterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.FreshTests("test_actual_prior_closure_and_nineteen_file_manifest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        self.created = time.time() - 60
        self.stop = self.created + 18000
        f.owner.update(pod_created_unix=self.created, pod_active_deadline_seconds=21600)
        f.save(f.op, f.owner)
        checkpoint = f.root / "checkpoint.json"
        checkpoint.write_bytes((fixtures.ARCHIVE / "checkpoint-proof.json").read_bytes())
        values = {"TOOLS": fixtures.ROOT, "OUTPUT": f.root / "campaign", "OUTER": f.root / "outer",
                  "OWNER": f.op, "NATIVE": f.native, "QUALITY": fixtures.INPUTS / "prior-quality9.json",
                  "SOURCE": f.sp, "ALLOCATION": f.ap, "CHECKPOINT": checkpoint,
                  "PREFLIGHT": fixtures.INPUTS / "preflight.json", "FULL": fixtures.INPUTS / "requests.json",
                  "LIFECYCLE": fixtures.ROOT.parent / "performance/prefill_e2e.py"}
        for name, value in values.items():
            self.enterContext(patch.object(outer, name, value))
        self.enterContext(patch.object(outer, "file_sha", side_effect=f.actual))
        self.enterContext(patch.object(guard, "file_sha", side_effect=f.actual))
        self.request = f.root / "request.json"
        outer.freeze(self.request, self.stop, self.created)

    def test_cpu_freeze_and_validate_actual_frozen_assets(self):
        value = outer.validate(self.request, file_sha(self.request))
        self.assertEqual(len(value["commands"]), 4)
        self.assertEqual([v[0] for v in value["commands"]], ["plan", "training", "freeze-heldout", "heldout"])
        self.assertTrue(set(self.fixture.native_files) <= set(value["files"]))

    def test_cutoff_reserves_one_hour_and_creation_binding(self):
        for stop, created in ((self.created + 18001, self.created), (self.stop, self.created + 1)):
            with self.subTest(stop=stop, created=created):
                with self.assertRaises(RuntimeError): outer.freeze(self.fixture.root / "bad.json", stop, created)

    def test_changed_command_or_inventory_rejected_even_with_new_request_hash(self):
        baseline = read(self.request)
        for mutate in (lambda r: r["commands"].pop(), lambda r: r["files"].pop(next(iter(r["files"]))),
                       lambda r: r.update(old_partial_results_reused=True)):
            r = json.loads(json.dumps(baseline))
            mutate(r)
            self.fixture.save(self.request, r)
            with self.assertRaises(RuntimeError): outer.validate(self.request, file_sha(self.request))

    def harness(self, fail_heldout=False):
        calls = []
        plan = {"schema": "fresh-four-h100-causal-v3", "output": str(outer.OUTPUT), "stop_unix": self.stop,
                "model": outer.MODEL, "owner_proof": str(outer.OWNER), "native_manifest": str(outer.NATIVE),
                "allocation": self.fixture.allocation}
        frozen = outer.load(fixtures.ROOT / "frozen_causal_outer_v1.py", "fixture_frozen_lifetime", outer.FROZEN_OUTER_SHA)
        def closed_mode(mode, count):
            root = outer.OUTPUT / mode
            root.mkdir()
            rows = []
            matrix = [(a, b) for b in ((16384, 1024) if mode == "training" else (16384,))
                      for a in ("stock", "keep6p8", "keep4p8")]
            for i, (arm, budget) in enumerate(matrix):
                p = root / (str(i) + ".json")
                write_new(p, {"complete": True, "workload_complete": True, "finalization_errors": [],
                    "arm": arm, "budget": budget, "plan_sha256": file_sha(outer.OUTPUT / "plan.json"),
                    "closure": {"causal": {"remaining_members": []}}})
                rows.append({"complete": True, "arm": arm, "budget": budget, "receipt": str(p), "sha256": file_sha(p)})
            write_new(root / "receipt.json", {"complete": True, "cleanup_complete": True, "mode": mode,
                "plan_sha256": file_sha(outer.OUTPUT / "plan.json"), "launches": rows})
        def popen(command, **kwargs):
            stage = [s for s, c in outer.commands(self.stop) if c == command][0]
            calls.append(stage)
            if stage == "plan":
                outer.OUTPUT.mkdir()
                write_new(outer.OUTPUT / "plan.json", plan)
            elif stage == "training":
                closed_mode(stage, 6)
            elif stage == "freeze-heldout":
                p = outer.OUTPUT / "heldout-prepared"
                p.mkdir()
                for name in ("protocol.json", "selected-requests.json"):
                    (p / name).write_bytes((fixtures.ARCHIVE / "causal-v2/heldout-prepared" / name).read_bytes())
                admin = outer.OUTPUT / "freeze-heldout/receipt.json"
                write_new(admin, {"complete": True, "cleanup_complete": True, "plan_sha256": file_sha(outer.OUTPUT / "plan.json")})
                write_new(outer.OUTPUT / "heldout-freeze-receipt.json", {"complete": True,
                    "plan_sha256": file_sha(outer.OUTPUT / "plan.json"),
                    "training_receipt_sha256": file_sha(outer.OUTPUT / "training/receipt.json"),
                    "protocol_sha256": file_sha(p / "protocol.json"), "selected_requests_sha256": file_sha(p / "selected-requests.json"),
                    "review_sha256": file_sha(fixtures.ROOT / "heldout-causal-v2-independent-review-v1.json"),
                    "no_heldout_model_queries_before_freeze": True})
            elif not fail_heldout:
                closed_mode(stage, 3)
            rc = 9 if fail_heldout and stage == "heldout" else 0
            return SimpleNamespace(pid=500 + len(calls), returncode=rc, poll=lambda: rc)
        cleanups = []
        def cleanup(child, identity, lib, grace):
            cleanups.append((child.pid, grace))
            return {"remaining_members": [], "identity": identity}
        native = SimpleNamespace(mode_receipt=frozen.mode_receipt, cleanup_owned=cleanup)
        lib = SimpleNamespace(process_record=lambda pid: {"pid": pid, "pgrp": pid, "start_ticks": "123"},
                              save=lambda p, v: p.write_text(json.dumps(v)), gpu_idle=lambda **kw: {"complete": True})
        self.enterContext(patch.object(outer, "load", side_effect=lambda path, *args: lib if path == outer.LIFECYCLE else native))
        self.enterContext(patch.object(outer.subprocess, "Popen", side_effect=popen))
        self.enterContext(patch.object(outer.signal, "signal"))
        return calls, cleanups

    def test_full_four_stage_control_roundtrip_and_owned_closure(self):
        calls, cleanups = self.harness()
        outer.run(self.request, file_sha(self.request))
        r = read(outer.OUTER / "receipt.json")
        self.assertTrue(r["complete"] and r["cleanup_complete"])
        self.assertEqual(calls, ["plan", "training", "freeze-heldout", "heldout"])
        self.assertEqual(len(cleanups), 4)
        self.assertTrue(all(grace == 720 for _, grace in cleanups))

    def test_failed_heldout_is_retained_without_replacement(self):
        calls, cleanups = self.harness(fail_heldout=True)
        with self.assertRaisesRegex(RuntimeError, "stage failed"):
            outer.run(self.request, file_sha(self.request))
        r = read(outer.OUTER / "receipt.json")
        self.assertFalse(r["complete"])
        self.assertTrue(r["cleanup_complete"])
        self.assertEqual(calls, ["plan", "training", "freeze-heldout", "heldout"])
        self.assertFalse(r["stages"][-1]["complete"])
        self.assertEqual(r["stages"][-1]["returncode"], 9)
        self.assertEqual(len(cleanups), 4)


if __name__ == "__main__":
    unittest.main()
