"""CPU-only safeguards for the focused serving protocol; no model or GPU needed."""
import contextlib
import dataclasses
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import analyze_serving
import serve_focus


class FocusProtocolTests(unittest.TestCase):
    def setUp(self):
        self.original_helpers = {name: getattr(serve_focus.common, name)
                                 for name in ("ARMS", "server_command", "arm_environment", "check_active")}
        self.original_definitions = serve_focus.ARM_DEFINITIONS
        self.original_environment = serve_focus.CONTROLLED_SERVER_ENVIRONMENT
        serve_focus.configure_helpers("warp1")

    def tearDown(self):
        for name, value in self.original_helpers.items():
            setattr(serve_focus.common, name, value)
        serve_focus.ARM_DEFINITIONS = self.original_definitions
        serve_focus.CONTROLLED_SERVER_ENVIRONMENT = self.original_environment

    def test_arm_environments_cannot_inherit_a_router_or_probe(self):
        with patch.dict(os.environ, {"PLACEMENT_KEEP": "4", "PLACEMENT_PROBE": "bad.py",
                                    "PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4", "PLACEMENT_NUM_WARPS": "8"}):
            stock = serve_focus.arm_environment("stock")
            self.assertTrue(all(key not in stock for key in ("PLACEMENT_KEEP", "PLACEMENT_PROBE",
                                                            "PLACEMENT_EXPERIMENTAL_SELECTOR", "PLACEMENT_NUM_WARPS")))
            self.assertEqual(stock["NVSHMEM_QP_DEPTH"], "4096")
            self.assertEqual(serve_focus.arm_environment("balanced6p1")["PLACEMENT_EXPERIMENTAL_SELECTOR"], "baseline")
            noop = serve_focus.arm_environment("noop8")
            self.assertEqual(noop["PLACEMENT_KEEP"], "8")
            self.assertEqual(noop["PLACEMENT_EXPERIMENTAL_SELECTOR"], "warp1")

    def test_four_pairs_balance_positions_and_screen_reverses_order(self):
        schedule = serve_focus.counterbalanced_schedule(["stock", "tuned6p1"], 4, 1729)
        for arm in ("stock", "tuned6p1"):
            for position in (0, 1):
                self.assertEqual(sum(row["arms"][position] == arm for row in schedule), 2)
        second = serve_focus.counterbalanced_schedule(["stock", "tuned6p1"], 1, 1729, 1)
        self.assertEqual(second[0]["arms"], list(reversed(schedule[0]["arms"])))

    def test_ll_uses_small_matched_budget_linear_experts_and_actual_graph_evidence(self):
        cells = serve_focus.cells_for("decode-backend", "screen")
        self.assertEqual({cell.batch_tokens for cell in cells}, {1024})
        args = SimpleNamespace(vllm="vllm", model="model", port=8077, dp=4, max_model_len=4608,
            seed=1, batch_tokens=1024, max_num_seqs=128, gpu_util=.88, revision=None,
            trust_remote_code=False, server_arg=[], backend="deepep_low_latency", jit_verbose=True)
        command = serve_focus.server_command(args, None)
        self.assertEqual(command[command.index("--all2all-backend") + 1], "deepep_low_latency")
        self.assertEqual(command[command.index("--expert-placement-strategy") + 1], "linear")
        self.assertIn("--enable-chunked-prefill", command)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            path.write_text("'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: 3>\nCapturing CUDA graphs\n")
            self.assertEqual(serve_focus.runtime_evidence(path)["observed_cudagraph_modes"], ["FULL_AND_PIECEWISE"])

    def test_full_warmups_and_all_rounds_retained_when_measured_jit_is_observed(self):
        shape = serve_focus.cells_for("prefill-concurrency", "screen")[0].shape
        calls = []

        @contextlib.contextmanager
        def fake_serve(args, arm, directory, trace):
            (directory / "server.json").write_text(json.dumps({"placement_environment": {}}))
            definition = serve_focus.ARM_DEFINITIONS[arm]
            text = "'cudagraph_mode': <CUDAGraphMode.NONE: 0>\n"
            if definition["keep"] is not None:
                text += f"placement-aware routing active, keep={definition['keep']} of top_k=8, ep_size=4\n"
                text += "placement-router: experimental selector=warp1\n"
            (directory / "server.log").write_text(text)
            yield

        def fake_workload(args, shape, prompts):
            calls.append(shape.requests)
            if len(calls) % 6 == 4:  # First measured round, after three full warmups.
                with (args.output / f"r00_{active_arm}" / "server.log").open("a") as log:
                    log.write("WARNING 10-17 09:00:00 [jit_monitor.py:140] Triton kernel JIT compilation during inference: example\n")
            latency = {"count": shape.requests, "p50": 1, "p95": 2, "p99": 3}
            now = time.time_ns()
            return {"started_unix_ns": now, "finished_unix_ns": now + 1000000000,
                "elapsed_seconds": 1, "requests": shape.requests, "concurrency": shape.concurrency,
                "prompt_tokens": shape.requests * shape.prompt_tokens, "completion_tokens": shape.requests,
                "requests_per_second": shape.requests, "output_tokens_per_second": shape.requests,
                "total_tokens_per_second": shape.requests * (shape.prompt_tokens + 1),
                "ttft_ms": latency, "request_latency_ms": latency, "tpot_ms": None, "itl_ms": None,
                "requests_with_single_token_chunks": shape.requests}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(output=root, phase="screen", warmup_rounds=3, rounds=3, dp=4)
            manifest = {"identity": {"config": {"arms": ["stock", "tuned6p1"], "shapes": ["prefill"],
                "repeats": 1, "rounds": 3, "model": "model", "dp": 4}, "workloads": [dataclasses.asdict(shape)]}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with patch.object(serve_focus.common, "serve", fake_serve), patch.object(serve_focus.common, "run_workload", fake_workload):
                for active_arm in ("stock", "tuned6p1"):
                    serve_focus.measure_launch(args, active_arm, 0, shape, [])
                    result = json.loads((root / f"r00_{active_arm}" / "result.json").read_text())
                    self.assertTrue(result["complete"])
                    self.assertFalse(result["jit_clean_measurements_observed"])
                    self.assertEqual(len(result["shapes"]["prefill"]["rounds"]), 3)
            report = analyze_serving.analyze(root)
            self.assertTrue(report["completion"]["complete"], report["issues"])
            self.assertEqual(report["completion"]["measured_rounds_in_valid_launches"], 6)
            self.assertEqual(report["completion"]["measured_requests_in_valid_launches"], 768)
            self.assertEqual(calls, [128] * 12)
            self.assertTrue(any("observed JIT" in issue["message"] for issue in report["issues"]))
            self.assertTrue(all(not item["required_for_completion"] for item in report["comparisons"]
                                if item["metric"].startswith("tpot_ms")))


if __name__ == "__main__":
    unittest.main()
