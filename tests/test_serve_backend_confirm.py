"""CPU-only checks for independent stock backend confirmation."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import serve_backend_confirm as runner
import serve_focus


class BackendConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.helpers = {name: getattr(runner.common, name)
                        for name in ("ARMS", "server_command", "arm_environment", "check_active")}
        self.definitions = serve_focus.ARM_DEFINITIONS
        self.environment = serve_focus.CONTROLLED_SERVER_ENVIRONMENT
        runner.configure_helpers()

    def tearDown(self):
        for name, value in self.helpers.items():
            setattr(runner.common, name, value)
        serve_focus.ARM_DEFINITIONS = self.definitions
        serve_focus.CONTROLLED_SERVER_ENVIRONMENT = self.environment

    def test_every_backend_arm_clears_every_placement_selector(self):
        with patch.dict(os.environ, {"PLACEMENT_KEEP": "6", "PLACEMENT_LOAD_PENALTY": "1",
                                    "PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4", "PLACEMENT_PROBE": "bad.py"}):
            for arm in runner.ARMS:
                env = runner.common.arm_environment(arm)
                self.assertTrue(all(name not in env for name in ("PLACEMENT_KEEP", "PLACEMENT_LOAD_PENALTY",
                    "PLACEMENT_EXPERIMENTAL_SELECTOR", "PLACEMENT_PROBE")))
                self.assertEqual(env["NVSHMEM_QP_DEPTH"], "4096")

    def test_actual_completed_capture_is_required_on_every_dp_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "server.log"
            header = "'cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>\n'cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32, 64]\nCapturing CUDA graphs\n"
            path.write_text(header)
            with self.assertRaisesRegex(RuntimeError, "every DP worker"):
                runner.require_graph_evidence("stock_ll", path, 4)
            completed = [f"(Worker_DP{rank}_EP{rank} pid=1) INFO Graph capturing finished in 2 secs, took .1 GiB\n"
                         for rank in range(4)]
            path.write_text(header + "".join(completed[:3]))
            with self.assertRaisesRegex(RuntimeError, "every DP worker"):
                runner.require_graph_evidence("stock_ll", path, 4)
            path.write_text(header + "".join(completed))
            self.assertEqual(runner.require_graph_evidence("stock_ll", path, 4)["graph_capture_completed_dp_ranks"], [0, 1, 2, 3])
            path.write_text("'cudagraph_mode': <CUDAGraphMode.NONE: 0>\n")
            self.assertEqual(runner.require_graph_evidence("stock", path, 4)["observed_cudagraph_modes"], ["NONE"])
            with self.assertRaisesRegex(RuntimeError, "FULL_DECODE_ONLY"):
                runner.require_graph_evidence("stock_ll", path, 4)

    def test_explicit_backend_graph_commands_match_the_passed_smoke(self):
        args = SimpleNamespace(vllm="vllm", model="model", port=8077, dp=4, max_model_len=4608,
            seed=1, batch_tokens=1024, max_num_seqs=64, gpu_util=.88, revision=None,
            trust_remote_code=False, server_arg=[], backend="deepep_high_throughput", jit_verbose=True)
        for backend, expected in runner.COMPILATION_CONFIGS.items():
            args.backend = backend
            command = runner.common.server_command(args, None)
            configs = [json.loads(value.split("=", 1)[1]) for value in command if value.startswith("--compilation-config=")]
            self.assertEqual(configs, [expected])
            self.assertEqual(command[command.index("--all2all-backend") + 1], backend)
            self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "1024")
        with self.assertRaisesRegex(RuntimeError, "profiling"):
            runner.common.server_command(args, Path("traces"))

    def test_fixed_shapes_and_counterbalancing(self):
        self.assertEqual([(s.prompt_tokens, s.output_tokens, s.concurrency, s.requests) for s in runner.SHAPES],
                         [(256, 128, 4, 16), (256, 128, 32, 64)])
        schedule = serve_focus.counterbalanced_schedule(list(runner.ARMS), 4, 1729)
        for arm in runner.ARMS:
            self.assertEqual([sum(row["arms"][position] == arm for row in schedule) for position in (0, 1)], [2, 2])


if __name__ == "__main__":
    unittest.main()
