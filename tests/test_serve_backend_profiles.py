"""CPU protocol checks for the separate backend diagnostic profiles."""
import contextlib
import gzip
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import serve_backend_profiles as runner


class BackendProfileTests(unittest.TestCase):
    def setUp(self):
        self.helpers = {name: getattr(runner.common, name)
                        for name in ("ARMS", "server_command", "arm_environment", "check_active")}
        self.definitions = runner.focus.ARM_DEFINITIONS
        self.environment = runner.focus.CONTROLLED_SERVER_ENVIRONMENT
        runner.configure_helpers()

    def tearDown(self):
        for name, value in self.helpers.items():
            setattr(runner.common, name, value)
        runner.focus.ARM_DEFINITIONS = self.definitions
        runner.focus.CONTROLLED_SERVER_ENVIRONMENT = self.environment

    def args(self, output=None):
        return SimpleNamespace(vllm="vllm", model="model", port=8077, dp=4, max_model_len=4608,
            seed=1729, batch_tokens=1024, max_num_seqs=64, gpu_util=.88, revision=None,
            trust_remote_code=False, server_arg=[], jit_verbose=True, output=output)

    def test_commands_retain_matched_backend_graph_and_profiler_configuration(self):
        for backend, compilation in runner.backend.COMPILATION_CONFIGS.items():
            args = self.args()
            args.backend = backend
            command = runner.server_command(args, Path("new_trace"))
            self.assertEqual(command[command.index("--all2all-backend") + 1], backend)
            self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "1024")
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")
            self.assertEqual(command[command.index("--expert-placement-strategy") + 1], "linear")
            self.assertIn("--enable-chunked-prefill", command)
            self.assertEqual(json.loads(command[-1].split("=", 1)[1]), compilation)
            profiler = json.loads(command[command.index("--profiler-config") + 1])
            self.assertEqual(profiler["torch_profiler_dir"], "new_trace")
            self.assertTrue(profiler["torch_profiler_record_shapes"])
        with self.assertRaisesRegex(RuntimeError, "trace directory"):
            runner.server_command(args, None)

    def test_inherited_placement_probe_and_profiler_environment_are_cleared(self):
        with patch.dict(os.environ, {"PLACEMENT_KEEP": "6", "PLACEMENT_PROBE": "bad.py",
                "PLACEMENT_EXPERIMENTAL_SELECTOR": "direct", "PLACEMENT_UNKNOWN_OPTION": "yes",
                "VLLM_TORCH_PROFILER_DIR": "/stale", "NVSHMEM_QP_DEPTH": "1024", "OMP_NUM_THREADS": "4"}):
            for arm in runner.backend.ARMS:
                env = runner.stock_environment(arm)
                self.assertFalse(any(key.startswith("PLACEMENT_") for key in env))
                self.assertNotIn("VLLM_TORCH_PROFILER_DIR", env)
                self.assertEqual(env["NVSHMEM_QP_DEPTH"], "4096")
                self.assertEqual(env["OMP_NUM_THREADS"], "1")

    def write_trace(self, directory, rank, graph=True, cpu=True, cuda=True):
        events = []
        if cpu:
            events.append({"cat": "cpu_op", "ph": "X", "name": "aten::linear", "dur": 2})
        if cuda:
            events.append({"cat": "kernel", "ph": "X", "name": "fused_moe_kernel", "dur": 10})
        if graph:
            events.extend([{"cat": "cuda_runtime", "ph": "X", "name": "cudaGraphLaunch_v10000", "dur": 3},
                           {"cat": "cuda_driver", "ph": "X", "name": "cuGraphLaunch", "dur": 2}])
        path = directory / f"dp{rank}_pp0_tp0_ep{rank}_rank0.123.pt.trace.json.gz"
        with gzip.open(path, "wt") as output:
            json.dump({"traceEvents": events}, output)

    def test_actual_graph_api_events_are_required_on_all_ll_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(4):
                self.write_trace(directory, rank, graph=rank != 3)
            summary = runner.summarize_traces(directory)
            self.assertTrue(summary["valid_all_dp_ranks_cpu_and_cuda"])
            with self.assertRaisesRegex(RuntimeError, "every DP rank"):
                runner.require_trace_evidence(summary, "stock_ll")
            self.write_trace(directory, 3)
            summary = runner.summarize_traces(directory)
            runner.require_trace_evidence(summary, "stock_ll")
            self.assertEqual(summary["cuda_graph_launch_api_events_by_dp_rank"], {0: 2, 1: 2, 2: 2, 3: 2})
            self.assertEqual(summary["traces"][0]["cuda_graph_launch_api_counts"]["cuGraphLaunch"], 1)
            with self.assertRaisesRegex(RuntimeError, "unexpectedly"):
                runner.require_trace_evidence(summary, "stock")

    def test_cpu_only_rank_cannot_pass_trace_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for rank in range(4):
                self.write_trace(directory, rank, graph=False, cuda=rank != 2)
            with self.assertRaisesRegex(RuntimeError, "CPU\\+CUDA"):
                runner.require_trace_evidence(runner.summarize_traces(directory), "stock")

    def test_three_full_warmups_precede_the_only_profile_workload(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, calls = self.args(Path(temporary)), []
            active = [False]
            @contextlib.contextmanager
            def serve(args, arm, directory, traces):
                (directory / "server.json").write_text("{}")
                (directory / "server.log").write_text("stock")
                yield
            def workload(args, shape, prompts, directory):
                calls.append((shape.prompt_tokens, shape.output_tokens, shape.concurrency,
                              shape.requests, active[0]))
                return {"configuration": shape.name}
            def control(args, endpoint):
                active[0] = endpoint == "/start_profile"
            summary = {"valid_all_dp_ranks_cpu_and_cuda": True, "no_placement_kernels": True,
                       "cuda_graph_launch_api_events_by_dp_rank": {rank: 0 for rank in range(4)}}
            with patch.object(runner.common, "serve", serve), patch.object(runner.focus, "workload", workload), \
                 patch.object(runner.common, "post_control", control), \
                 patch.object(runner.backend, "require_graph_evidence", return_value={}), \
                 patch.object(runner, "summarize_traces", return_value=summary):
                runner.launch(args, "stock", [{"token_ids": [1]}])
            self.assertEqual(calls, [(256, 128, 4, 16, False)] * 3 + [(256, 16, 4, 16, True)])
            self.assertFalse(active[0])
            result = json.loads((args.output / "profile_stock_ht/result.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["warmups"]), 3)


if __name__ == "__main__":
    unittest.main()
