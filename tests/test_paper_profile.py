"""CPU checks for profile isolation, original-source identity and trace coverage."""
from __future__ import annotations

import contextlib
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import paper_profile as runner


class PaperProfileTests(unittest.TestCase):
    def args(self, output, dp=4):
        return runner.arguments(["--dp", str(dp), "--model", "/model", "--output", str(output)])

    def test_exact_paper_server_configuration_adds_only_profiler(self):
        for dp in (4, 8):
            args = self.args("/profiles", dp)
            command = runner.server_command(args, Path("/profiles/traces"))
            self.assertEqual(command[:-2], runner.paper.server_command(args))
            self.assertEqual(command[command.index("--data-parallel-size") + 1], str(dp))
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
            self.assertIn('--compilation-config={"cudagraph_mode":"NONE"}', command)
            self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "16384")
            profiler = json.loads(command[-1])
            self.assertEqual(profiler["profiler"], "torch")
            self.assertTrue(profiler["torch_profiler_record_shapes"])
            self.assertTrue(profiler["torch_profiler_use_gzip"])
        with self.assertRaisesRegex(RuntimeError, "trace directory"):
            runner.server_command(args, None)

    def test_all_probe_and_experimental_controls_are_removed(self):
        polluted = {"PLACEMENT_KEEP": "4", "PLACEMENT_PROBE": "paper_probe.py",
            "PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4", "PLACEMENT_DEEPEP_SMS": "64",
            "PLACEMENT_FUTURE_CONTROL": "on", "PAPER_PROBE_DIR": "/stale",
            "PAPER_PROBE_CONTROL": "/control", "VLLM_TORCH_PROFILER_DIR": "/old",
            "VLLM_DEEPEP_BUFFER_SIZE_MB": "8", "VLLM_PLUGINS": "other_hook",
            "OMP_NUM_THREADS": "4", "NVSHMEM_QP_DEPTH": "1024", "PYTHONPATH": "/old"}
        with patch.dict(os.environ, polluted):
            stock, balanced = (runner.profile_environment(a) for a in runner.ARMS)
        self.assertFalse(any(k.startswith(("PLACEMENT_", "PAPER_PROBE_", "VLLM_TORCH_PROFILER_")) for k in stock))
        self.assertEqual({k: v for k, v in balanced.items() if k.startswith("PLACEMENT_")},
                         {"PLACEMENT_KEEP": "6", "PLACEMENT_LOAD_PENALTY": "1"})
        for env in (stock, balanced):
            self.assertNotIn("VLLM_DEEPEP_BUFFER_SIZE_MB", env)
            self.assertEqual(env["OMP_NUM_THREADS"], "1")
            self.assertEqual(env["NVSHMEM_QP_DEPTH"], "4096")
            self.assertEqual(env["VLLM_PLUGINS"], "placement_aware_router")
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[:2],
                             [str(runner.ROOT / "src"), str(runner.ROOT / "experiments")])

    def test_sm_contract_handles_real_annotated_default_and_fails_changed_runtime(self):
        manager = "class DeepEPAll2AllManagerBase:\n def __init__(self):\n  self.num_sms = 20\n"
        legacy = "class Buffer:\n num_sms: int = 20\n"
        proof = runner.sm20_source_contract(manager, legacy)
        self.assertEqual(proof["legacy_Buffer_num_sms_default"], 20)
        for a, b in ((manager.replace("20", "64"), legacy), (manager, legacy.replace("20", "40"))):
            with self.assertRaisesRegex(RuntimeError, "Unreviewed SM20"):
                runner.sm20_source_contract(a, b)

    def trace(self, directory, rank, arm="stock", T=16384, cpu=True, router=True,
              expert=True, experimental=False, graph=False):
        events = []
        for _ in range(48):
            if cpu:
                events.append({"cat": "cpu_op", "ph": "X", "name": "vllm::moe_forward",
                    "args": {"Input Dims": [[T, 2048], [T, 2048], [], [], [], []]}, "dur": 100})
            if router:
                name = "_par_select" if arm == "balanced6p1" else "void vllm::moe::topkGating<8,128>()"
                events.append({"cat": "kernel", "ph": "X", "name": name, "dur": 3})
            if expert:
                events.append({"cat": "kernel", "ph": "X", "name": "fused_moe_kernel", "dur": 10})
        if experimental:
            events.append({"cat": "kernel", "ph": "X", "name": "_par_select_rows_gl", "dur": 1})
        if graph:
            events.append({"cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch_v10000", "dur": 2})
        path = directory / f"dp{rank}_pp0_tp0_ep{rank}_rank0.test.pt.trace.json.gz"
        with gzip.open(path, "wt") as stream:
            json.dump({"traceEvents": events}, stream)
        return path

    def test_positive_original_router_moe_expert_and_exact_T_on_every_rank(self):
        for dp in (4, 8):
            for arm in runner.ARMS:
                with tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp)
                    for rank in range(dp):
                        self.trace(directory, rank, arm)
                    summary = runner.summarize_traces(directory, dp, arm)
                    runner.require_trace_evidence(summary, arm)
                    self.assertEqual(len(summary["traces"]), dp)
                    for rank in range(dp):
                        row = summary["by_dp_rank"][str(rank)]
                        self.assertEqual(row["moe_hidden_T_counts"], {"16384": 48})
                        self.assertEqual(row["counts"]["expert_kernel_calls"], 48)

    def test_missing_rank_cpu_router_expert_wrong_T_graph_and_tuned_kernel_fail(self):
        cases = (({"T": 32767}, "exact-T"), ({"cpu": False}, "CPU\\+CUDA"),
                 ({"router": False}, "router identity"), ({"expert": False}, "expert-kernel"),
                 ({"graph": True}, "CUDA graph"), ({"experimental": True}, "experimental selector"))
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for rank in range(4):
                    self.trace(directory, rank, "balanced6p1", **(kwargs if rank == 3 else {}))
                summary = runner.summarize_traces(directory, 4, "balanced6p1")
                with self.assertRaisesRegex(RuntimeError, message):
                    runner.require_trace_evidence(summary, "balanced6p1")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for rank in range(3):
                self.trace(directory, rank)
            with self.assertRaisesRegex(RuntimeError, "Missing/unexpected"):
                runner.require_trace_evidence(runner.summarize_traces(directory, 4, "stock"), "stock")

    def test_pristine_source_guard_cannot_accept_experimental_kernel(self):
        identity = {"source_sha256": dict(runner.paper.ORIGINAL_SOURCE)}
        runner.paper.require_original_source(identity)
        identity["source_sha256"]["src/placement_router/kernel.py"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            runner.paper.require_original_source(identity)

    def test_resume_rechecks_raw_trace_hashes_instead_of_trusting_complete_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "manifest.json").write_text("{}")
            directory = output / "profile_stock"
            traces = directory / "traces_first"
            traces.mkdir(parents=True)
            for rank in range(4):
                self.trace(traces, rank)
            saved = {"complete": True, "identity_sha256": runner.paper.digest(output / "manifest.json"),
                "trace_directory": str(traces), "trace_summary": runner.summarize_traces(traces, 4, "stock")}
            runner.common.atomic_json(directory / "result.json", saved)
            with patch("builtins.print"):
                runner.launch(self.args(output), "stock", [], {})
            path = next(traces.glob("*dp3_*.gz"))
            with gzip.open(path, "rt") as source:
                trace = json.load(source)
            trace["traceEvents"][0]["dur"] += 1
            with gzip.open(path, "wt") as source:
                json.dump(trace, source)
            with self.assertRaisesRegex(RuntimeError, "raw traces changed"):
                runner.launch(self.args(output), "stock", [], {})

    def test_only_bounded_workload_is_profiled_after_three_full_warmups(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "manifest.json").write_text("{}")
            args, calls, active = self.args(output), [], [False]
            @contextlib.contextmanager
            def serve(*_args):
                yield None
            def workload(args, shape, prompts, directory):
                calls.append((shape.prompt_tokens, shape.output_tokens, shape.requests,
                              shape.concurrency, len(prompts), active[0]))
                return {"configuration": shape.name}
            def control(args, endpoint):
                active[0] = endpoint == "/start_profile"
            with patch.object(runner, "serve", serve), patch.object(runner, "check_log", return_value={}), \
                 patch.object(runner, "workload", workload), patch.object(runner.common, "post_control", control), \
                 patch.object(runner, "summarize_traces", return_value={}), patch.object(runner, "require_trace_evidence"):
                result = runner.launch(args, "stock", [{"token_ids": [1]}] * 256, {})
            self.assertEqual(calls, [(4096, 1, 256, 128, 256, False)] * 3 + [(4096, 1, 128, 128, 128, True)])
            self.assertTrue(result["complete"])
            self.assertFalse(active[0])

    def test_profile_failure_stops_profiler_and_preserves_incomplete_result(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "manifest.json").write_text("{}")
            endpoints = []
            @contextlib.contextmanager
            def serve(*_args):
                yield None
            def workload(args, shape, prompts, directory):
                if shape == runner.PROFILE:
                    raise RuntimeError("Request failure")
                return {}
            with patch.object(runner, "serve", serve), patch.object(runner, "check_log", return_value={}), \
                 patch.object(runner, "workload", workload), \
                 patch.object(runner.common, "post_control", side_effect=lambda a, endpoint: endpoints.append(endpoint)):
                with self.assertRaisesRegex(RuntimeError, "Request failure"):
                    runner.launch(self.args(output), "stock", [], {})
            self.assertEqual(endpoints, ["/start_profile", "/stop_profile"])
            result = json.loads((output / "profile_stock/result.json").read_text())
            self.assertFalse(result["complete"])
            self.assertIn("Request failure", result["error"])

    def test_plan_does_not_touch_model_hardware_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "uncreated"
            with patch.object(runner, "ensure_manifest", side_effect=AssertionError("model access")), \
                 patch.object(runner.common, "metadata", side_effect=AssertionError("hardware query")), \
                 patch("builtins.print"):
                self.assertEqual(runner.main(["--dp", "8", "--output", str(output)]), 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
