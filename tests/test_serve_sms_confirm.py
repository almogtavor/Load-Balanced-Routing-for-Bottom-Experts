"""CPU-only safeguards for the four-arm SM/routing serving factorial."""
from collections import Counter
import copy
import dataclasses
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import serve_sms_confirm as runner


def evidence(arm):
    definition = runner.ARMS[arm]
    return [{"kind": "deepep_ht_sm_startup", "ep_rank": rank, "ep_world_size": 4,
        "requested_sms": definition["num_sms"], "manager_sms_ceiling": definition["num_sms"],
        "buffer_num_sms": definition["num_sms"], "internode": False, "deep_ep": "2.0.0",
        "rdma_allocated_bytes": 0, "num_qps_per_rank": 1, "enable_dbo": False,
        "use_ubatching": False, "ubatch_size": 0, "nvl_requested_bytes": runner.BUFFER_BYTES,
        "nvl_allocated_bytes": runner.BUFFER_BYTES, "nvl_auto_expanded": False,
        "nvl_dispatch_size_hint_bytes": 201591296, "nvl_combine_size_hint_bytes": 201591296,
        "buffer_allocation_has_not_run_yet": True, "gpu_physical_sms": 132,
        "placement_keep": None if definition["keep"] is None else "6",
        "placement_selector": definition["selector"], "hidden_size": 2048,
        "model_dtype": "torch.bfloat16", "padded_hidden_bytes": 4096} for rank in range(4)]


class SMConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.common = {name: getattr(runner.common, name) for name in
                       ("ARMS", "server_command", "arm_environment", "serve")}
        self.focus = {name: getattr(runner.focus, name) for name in
                      ("ARM_DEFINITIONS", "CONTROLLED_SERVER_ENVIRONMENT", "check_active")}
        self.active = runner.ACTIVE_EVIDENCE_DIRECTORY
        runner.configure_helpers("/work/router/experiments")
        runner.ACTIVE_EVIDENCE_DIRECTORY = Path("/results/unique-launch-evidence")

    def tearDown(self):
        for name, value in self.common.items():
            setattr(runner.common, name, value)
        for name, value in self.focus.items():
            setattr(runner.focus, name, value)
        runner.ACTIVE_EVIDENCE_DIRECTORY = self.active

    def test_williams_design_balances_positions_and_all_predecessors(self):
        rows = runner.schedule(1729)
        for arm in runner.ARMS:
            self.assertEqual([sum(row["arms"][position] == arm for row in rows) for position in range(4)], [1] * 4)
        predecessors = Counter(pair for row in rows for pair in zip(row["arms"], row["arms"][1:]))
        self.assertEqual(len(predecessors), 12)
        self.assertEqual(set(predecessors.values()), {1})

    def test_exact_budget_bytes_scope_and_routing_identity_required(self):
        for arm in runner.ARMS:
            self.assertTrue(runner.validate_evidence(evidence(arm), arm)["valid_all_ep_ranks"])
        for field, value in (("requested_sms", 20), ("manager_sms_ceiling", 20),
            ("buffer_num_sms", 20), ("nvl_allocated_bytes", 2048), ("nvl_requested_bytes", 2048),
            ("use_ubatching", True), ("enable_dbo", True), ("placement_selector", "rows4")):
            with self.subTest(field=field):
                rows = evidence("tuned_s64")
                rows[2][field] = value
                with self.assertRaises(RuntimeError):
                    runner.validate_evidence(rows, "tuned_s64")
        with self.assertRaisesRegex(RuntimeError, "every EP rank"):
            runner.validate_evidence(evidence("stock_s20")[:3], "stock_s20")

    def test_stock_and_tuned_share_extension_buffer_and_cpu_environment(self):
        with patch.dict(os.environ, {"PLACEMENT_KEEP": "4", "PLACEMENT_PROBE": "bad",
            "PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4", "PLACEMENT_DEEPEP_NUM_SMS": "40"}):
            for arm, definition in runner.ARMS.items():
                env = runner.common.arm_environment(arm)
                self.assertEqual(env["PLACEMENT_DEEPEP_NUM_SMS"], str(definition["num_sms"]))
                self.assertEqual(env["VLLM_DEEPEP_BUFFER_SIZE_MB"], "1024")
                self.assertEqual(env["OMP_NUM_THREADS"], "1")
                self.assertTrue(env["PYTHONPATH"].startswith("/work/router/experiments"))
                self.assertNotIn("PLACEMENT_PROBE", env)
                if definition["keep"] is None:
                    self.assertNotIn("PLACEMENT_KEEP", env)
                    self.assertNotIn("PLACEMENT_EXPERIMENTAL_SELECTOR", env)
                else:
                    self.assertEqual(env["PLACEMENT_EXPERIMENTAL_SELECTOR"], "rows4_direct")
        args = SimpleNamespace(vllm="vllm", model="model", port=1, dp=4, max_model_len=4608,
            seed=1, batch_tokens=16384, max_num_seqs=128, gpu_util=.88, revision=None,
            trust_remote_code=False, server_arg=[], backend="deepep_high_throughput", jit_verbose=True)
        command = runner.common.server_command(args, None)
        self.assertIn("--worker-extension-cls=" + runner.EXTENSION, command)
        self.assertIn('--compilation-config={"cudagraph_mode":"NONE"}', command)
        with self.assertRaisesRegex(RuntimeError, "profiling"):
            runner.common.server_command(args, Path("trace"))

    def test_offline_factorial_analysis_uses_four_launches_not_requests(self):
        speeds = {"stock_s20": 100., "tuned_s20": 105., "stock_s64": 140., "tuned_s64": 154.}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identity = {"config": {"rounds": 3, "warmup_rounds": 3}, "schedule": runner.schedule(1729),
                        "source_sha256": {"test": "frozen"}}
            (output / "manifest.json").write_text(json.dumps({"identity": identity}))
            for sweep in identity["schedule"]:
                for arm in sweep["arms"]:
                    shape, speed = runner.SHAPE, speeds[arm]
                    elapsed = shape.requests / speed
                    latency = {"count": shape.requests, "p50": 100, "p95": 110, "p99": 120}
                    row = {"requests": shape.requests, "concurrency": shape.concurrency,
                        "prompt_tokens": shape.requests * shape.prompt_tokens, "completion_tokens": shape.requests,
                        "elapsed_seconds": elapsed, "requests_per_second": speed, "output_tokens_per_second": speed,
                        "total_tokens_per_second": speed * (shape.prompt_tokens + 1),
                        "ttft_ms": latency, "request_latency_ms": latency, "tpot_ms": None, "itl_ms": None,
                        "requests_with_single_token_chunks": shape.requests}
                    value = {"arm": arm, "repeat": sweep["repeat"], "complete": True,
                        "measurement_type": "uninstrumented_streaming_http", "benchmark_source_sha256": identity["source_sha256"],
                        "sms_evidence": {"rank_records": evidence(arm)}, "shapes": {shape.name: {
                            "configuration": dataclasses.asdict(shape), "warmups": [row] * 3, "rounds": [row] * 3}}}
                    directory = output / f"r{sweep['repeat']:02d}_{arm}"
                    directory.mkdir()
                    (directory / "result.json").write_text(json.dumps(value))
            report = runner.write_analysis(output)
            self.assertTrue(report["complete"], report["issues"])
            self.assertEqual(report["counts"], {"expected_launches": 16, "valid_launches": 16,
                                               "measured_rounds": 48, "measured_requests": 6144})
            expected = {"deployment_sm_gain": 1.4, "placement_at_64": 1.1, "placement_at_20": 1.05, "combined": 1.54}
            for comparison in report["comparisons"]:
                if comparison["metric"] == "output_tokens_per_second":
                    self.assertEqual(comparison["n_paired_launches"], 4)
                    self.assertEqual(comparison["mean_log_ratio"]["df"], 3)
                    self.assertAlmostEqual(comparison["geometric_mean_ratio"], expected[comparison["contrast"]])


if __name__ == "__main__":
    unittest.main()
