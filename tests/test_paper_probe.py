"""CPU tests of the actual paper-probe wrappers, event order and drain protocol."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import types
import unittest
from unittest import mock

import paper_probe as probe_module


class FakeCUDA:
    def __init__(self):
        self.events, self.operations = [], []
        self.capturing, self.stream_id = False, 7

    def Event(self, enable_timing=False):
        assert enable_timing
        event = FakeEvent(self, len(self.events))
        self.events.append(event)
        return event

    def current_stream(self):
        return SimpleNamespace(cuda_stream=self.stream_id)

    def is_current_stream_capturing(self):
        return self.capturing

    def set_device(self, device):
        self.device = device


class FakeEvent:
    def __init__(self, cuda, identifier):
        self.cuda, self.identifier = cuda, identifier
        self.done = False

    def record(self, stream):
        self.cuda.operations.append(("record", self.identifier, stream.cuda_stream))

    def synchronize(self):
        self.cuda.operations.append(("synchronize", self.identifier))
        self.done = True

    def query(self):
        self.cuda.operations.append(("query", self.identifier))
        return self.done

    def elapsed_time(self, end):
        assert end.done, "Elapsed time read before the end completed"
        self.cuda.operations.append(("elapsed", self.identifier, end.identifier))
        return 1.25


class PaperProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.control = self.directory / "control.json"
        self.control.write_text(json.dumps({"phase": "warmup", "epoch": 0, "round": 0}))
        self.cuda = FakeCUDA()
        self.probe = probe_module.PaperProbe(SimpleNamespace(cuda=self.cuda), self.directory / "probe",
            self.control, lambda runner, hidden: {"ep_rank": 2, "device_index": 2, "selector": "baseline"},
            start_thread=False)
        self.prep_calls = self.comb_calls = 0

        def prepare(a1=None, topk_ids=None):
            def receiver(value="tokens"):
                self.prep_calls += 1
                return value
            return receiver

        def combine(do_async=True):
            def receiver():
                self.comb_calls += 1
                return None
            return receiver if do_async else receiver()
        self.prepare = self.probe.wrap_half("dispatch", prepare)
        self.combine = self.probe.wrap_half("combine", combine)

        def inner(runner, hidden, do_async=True):
            self.assertEqual(self.prepare()(value="tokens"), "tokens")
            receiver = self.combine(do_async)
            if receiver is not None:
                self.assertIsNone(receiver())
            return "output"
        self.forward = self.probe.wrap_forward(inner)
        self.runners = [SimpleNamespace(layer_name=f"model.layers.{i}.mlp.experts", layer_id=i) for i in range(48)]

    def tearDown(self):
        self.probe.close()
        self.temp.cleanup()

    def call(self, index, tokens=16384, **kwargs):
        runner = self.runners[index % 48]
        return self.forward(runner, SimpleNamespace(shape=(tokens, 2048)), **kwargs)

    def change(self, phase, epoch, round_index=0):
        self.control.write_text(json.dumps({"phase": phase, "epoch": epoch, "round": round_index}))
        self.probe.refresh_control()

    def test_original_global96_warmup_and_chunk64_event_flush_order(self):
        for index in range(96):
            self.call(index)
        self.assertFalse(self.cuda.events)
        self.assertEqual(self.probe.skips["first96"], 96)
        for index in range(96, 159):
            self.call(index)
        self.assertFalse(any(op[0] == "synchronize" for op in self.cuda.operations))
        self.call(159)
        sync = [op[1] for op in self.cuda.operations if op[0] == "synchronize"]
        self.assertEqual(sync, [offset + end for offset in range(0, 64 * 6, 6) for end in (3, 5, 1)])
        elapsed = [op for op in self.cuda.operations if op[0] == "elapsed"]
        self.assertEqual([(op[1], op[2]) for op in elapsed[:3]], [(2, 3), (4, 5), (0, 1)])
        self.assertEqual(self.probe.since_flush, 0)
        self.assertEqual(self.probe.pending, [])
        data = self.probe.snapshot()
        self.assertEqual(len(data["forward_records"]), 64)
        self.assertTrue(all(r["valid"] and r["flush_origin"] == "chunk64" for r in data["forward_records"]))

    def test_async_receiver_end_is_after_callback_sync_combine_is_supported(self):
        self.probe.seen = 96
        self.call(0, do_async=False)
        self.assertEqual([span["phase"] for span in self.probe.pending], ["dispatch", "combine", "moe"])
        self.assertEqual(self.prep_calls, 1)
        self.assertEqual(self.comb_calls, 1)
        self.assertEqual([op[1] for op in self.cuda.operations if op[0] == "record"], [0, 2, 3, 4, 5, 1])

    def test_terminal_drain_never_synchronizes_unfinished_work(self):
        self.probe.seen = 96
        self.call(0)
        self.change("drain", 1)
        self.probe.drain_completed_tail()
        self.assertEqual(len(self.probe.pending), 3)
        self.assertFalse(self.probe.snapshot()["drained"])
        self.assertFalse(any(op[0] in ("synchronize", "elapsed") for op in self.cuda.operations))
        for event in self.cuda.events:
            event.done = True
        self.probe.drain_completed_tail()
        data = self.probe.snapshot()
        self.assertTrue(data["drained"])
        self.assertEqual(data["pending_spans"], 0)
        self.assertEqual(data["forward_records"][0]["flush_origin"], "drain_tail")
        self.assertTrue(data["forward_records"][0]["valid"])
        self.assertFalse(any(op[0] == "synchronize" for op in self.cuda.operations))
        self.probe.write(force=True)
        saved = json.loads(self.probe.path.read_text())
        self.assertEqual(saved["metadata"]["ep_rank"], 2)
        self.assertIn("rank2_pid", self.probe.path.name)
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.change("measure", 2)

    def test_actualT_not_bucket_controls_complete48_layer_step_coverage(self):
        self.probe.seen = 96
        self.change("measure", 1, 0)
        for index in range(48):
            self.call(index, tokens=16384)
        # Same legacy bucket, but not the exact T. It must never add an exact step.
        self.change("measure", 2, 1)
        for index in range(48):
            self.call(index, tokens=32767)
        for event in self.cuda.events:
            event.done = True
        self.change("drain", 3)
        self.probe.drain_completed_tail()
        data = self.probe.snapshot()
        self.assertEqual(data["coverage"]["by_epoch"]["1"]["complete_step_ids"], [1])
        self.assertEqual(data["coverage"]["by_epoch"]["1"]["minimum_exact_T_ready_forwards_per_layer"], 1)
        self.assertEqual(data["coverage"]["by_epoch"]["2"]["complete_step_ids"], [])
        self.assertEqual({r["bucket"] for r in data["forward_records"]}, {16384})
        self.assertEqual({r["T"] for r in data["forward_records"]}, {16384, 32767})

    def test_capture_skip_has_no_events_no_runtime_metadata_or_file(self):
        self.probe.seen = 96
        self.cuda.capturing = True
        self.call(0)
        self.assertFalse(self.cuda.events)
        self.assertIsNone(self.probe.metadata)
        self.assertIsNone(self.probe.path)
        self.assertEqual(self.probe.skips["capture"], 1)

    def test_changed_epoch_required_and_metadata_latched_for_whole_step(self):
        self.probe.seen = 96
        self.change("measure", 1, 0)
        self.call(0)
        self.change("measure", 2, 1)
        self.call(1)
        self.assertEqual([r["epoch"] for r in self.probe.rows], [1, 1])
        with self.assertRaisesRegex(ValueError, "requires a new epoch"):
            self.change("warmup", 2, 0)

    def test_pristine_kernel_has_no_experimental_private_fields(self):
        source = self.directory / "kernel.py"
        source.write_text("# original source\n")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        kernel = SimpleNamespace(__file__=str(source))
        self.assertEqual(probe_module.original_kernel_evidence(kernel, digest), digest)
        source.write_text("# modified baseline source\n")
        with self.assertRaisesRegex(RuntimeError, "pristine"):
            probe_module.original_kernel_evidence(kernel, digest)

    def test_installed_probe_uses_construction_config_without_runtime_context(self):
        parallel = SimpleNamespace(tensor_parallel_size=1, data_parallel_size=4,
            pipeline_parallel_size=1, all2all_backend="deepep_high_throughput",
            enable_dbo=False, use_ubatching=False, ubatch_size=0)
        config = SimpleNamespace(parallel_config=parallel,
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"),
                static_all_moe_layers=[]),
            model_config=SimpleNamespace(quantization=None, model="/verified/model"))
        context, calls = {"config": config}, []
        def get_current_config():
            calls.append(context["config"])
            if context["config"] is None:
                raise AssertionError("Current vLLM config is not set")
            return context["config"]

        class NativeRunner:
            def __init__(self, layer_name):
                self.layer_name = layer_name
                get_current_config().compilation_config.static_all_moe_layers.append(layer_name)
            @property
            def layer_id(self):
                return probe_module.layer_index(self.layer_name)
            def _forward_impl(self, hidden):
                return "native output"
        class NativePrepareFinalize:
            def prepare_async(self):
                return lambda: None
            def _finalize(self):
                return None
        for method in (NativeRunner.__init__, NativeRunner._forward_impl,
                       NativePrepareFinalize.prepare_async, NativePrepareFinalize._finalize):
            method.__module__ = "vllm.fake_native"
        modules = {name: types.ModuleType(name) for name in (
            "vllm.config", "vllm.distributed.parallel_state",
            "vllm.model_executor.layers.fused_moe.layer",
            "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht",
            "placement_router")}
        modules["vllm.config"].get_current_vllm_config = get_current_config
        modules["vllm.distributed.parallel_state"].get_ep_group = lambda: SimpleNamespace(rank_in_group=0, world_size=4)
        modules["vllm.model_executor.layers.fused_moe.layer"].MoERunner = NativeRunner
        modules["vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht"].DeepEPHTPrepareAndFinalize = NativePrepareFinalize
        modules["placement_router"].kernel = SimpleNamespace()
        cuda = FakeCUDA()
        cuda.current_device = lambda: 0
        cuda.get_device_name = lambda device: "NVIDIA H100 80GB HBM3"
        modules["torch"] = SimpleNamespace(cuda=cuda, __version__="fake-test")
        environment = {"PAPER_PROBE_DIR": str(self.directory / "installed"),
            "PAPER_PROBE_CONTROL": str(self.control), "PAPER_PROBE_EXPECTED_EP": "4",
            "PAPER_PROBE_EXPECTED_LAYERS": "48", "PAPER_PROBE_TOKENS": "16384",
            "PLACEMENT_EXPERIMENTAL_SELECTOR": "baseline"}
        with mock.patch.dict("sys.modules", modules), mock.patch.dict("os.environ", environment), \
                mock.patch.object(probe_module, "_INSTANCE", None), \
                mock.patch.object(probe_module, "original_kernel_evidence", return_value=probe_module.ORIGINAL_KERNEL_SHA256):
            probe_module.install()
            installed = probe_module._INSTANCE
            installed.start_thread = False
            runners = [NativeRunner(f"model.layers.{i}.mlp.experts") for i in range(48)]
            self.assertIs(probe_module.cached_runner_config(runners[0]), config)
            self.assertEqual(len(calls), 96)  # Native constructor plus capture per layer.
            context["config"] = None
            with self.assertRaisesRegex(AssertionError, "not set"):
                get_current_config()
            count = len(calls)
            try:
                self.assertEqual(runners[0]._forward_impl(SimpleNamespace(
                    dtype="torch.bfloat16", shape=(16384, 2048))), "native output")
                self.assertEqual(len(calls), count)
                self.assertTrue(installed.snapshot()["ready"])
                metadata = installed.metadata
                self.assertEqual(metadata["model"], "/verified/model")
                self.assertEqual(len(metadata["registered_MoE_layer_names"]), 48)
                self.assertFalse(metadata["config_acquisition"]["runtime_global_context_lookup"])
                self.assertFalse(cuda.events)  # Original first96 warmup is unchanged.
            finally:
                installed.close()

    def test_missing_construction_config_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "construction-cached"):
            probe_module.cached_runner_config(SimpleNamespace())

    def test_layer_name_failfast(self):
        self.assertEqual(probe_module.layer_index("model.layers.47.mlp.experts"), 47)
        with self.assertRaisesRegex(ValueError, "decoder layer"):
            probe_module.layer_index("unidentified")

    def test_idle_control_ack_and_final_tail_drain_need_no_new_forward(self):
        self.probe.start_thread = True
        self.probe.poll_seconds = .01
        self.probe.seen = 96
        self.call(0)
        def atomic_control(value):
            temporary = self.control.with_suffix(".tmp")
            temporary.write_text(json.dumps(value))
            temporary.replace(self.control)
        atomic_control({"phase": "measure", "epoch": 1, "round": 1})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            data = json.loads(self.probe.path.read_text())
            if data["control_ack"]["epoch"] == 1:
                break
            time.sleep(.01)
        self.assertEqual(data["control_ack"]["epoch"], 1)
        self.assertFalse(any(op[0] in ("query", "synchronize") for op in self.cuda.operations))
        for event in self.cuda.events:
            event.done = True
        atomic_control({"phase": "drain", "epoch": 2, "round": 1})
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            data = json.loads(self.probe.path.read_text())
            if data["drained"]:
                break
            time.sleep(.01)
        self.assertTrue(data["drained"])
        self.assertEqual(data["control_ack"]["epoch"], 2)
        self.assertFalse(any(op[0] == "synchronize" for op in self.cuda.operations))

    def test_dispatch_actualT_and_native_layer_id_are_checked(self):
        self.probe.seen = 96
        ids = SimpleNamespace(dtype="torch.int64", shape=(16384, 8))
        def inner(runner, hidden):
            self.prepare(a1=hidden, topk_ids=ids)()
            self.combine()()
            return None
        forward = self.probe.wrap_forward(inner)
        forward(self.runners[0], SimpleNamespace(shape=(16384, 2048)))
        self.assertEqual(self.probe.rows[0]["dispatch_input_T"], 16384)
        self.assertEqual(self.probe.rows[0]["routing_ID_dtype"], "torch.int64")
        self.runners[1].layer_id = 5
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            forward(self.runners[1], SimpleNamespace(shape=(16384, 2048)))

    def test_uninvoked_dispatch_receiver_cannot_report_ready_or_drained(self):
        self.probe.seen = 96
        def inner(runner, hidden):
            self.prepare()  # Deliberately fail to consume its completion receiver.
            self.combine()()
        forward = self.probe.wrap_forward(inner)
        forward(self.runners[0], SimpleNamespace(shape=(16384, 2048)))
        self.assertTrue(any("completed dispatch" in message for message in self.probe.errors))
        self.change("drain", 1)
        for event in self.cuda.events:
            event.done = True
        self.probe.drain_completed_tail()
        self.assertFalse(self.probe.snapshot()["drained"])
        self.assertFalse(self.probe.snapshot()["ready"])

    def test_timed_chunk_writes_never_traverse_or_serialize_raw_history(self):
        class HistoryMustNotBeWalked(list):
            def __iter__(self):
                raise AssertionError("Timed flush traversed cumulative forward history")
        self.probe.seen = 96
        for index in range(64):
            self.call(index)
        self.probe.rows = HistoryMustNotBeWalked(self.probe.rows)
        for index in range(64, 128):
            self.call(index)
        data = json.loads(self.probe.path.read_text())
        self.assertFalse(data["records_included"])
        self.assertNotIn("forward_records", data)
        self.assertFalse(data["coverage"]["details_included"])
        self.assertEqual(data["coverage"]["by_epoch"], {})
        self.assertEqual(data["records_stored_in_memory"], 128)
        self.assertEqual(data["completed_forwards"], 128)
        self.assertEqual(data["original_summary"]["buckets"]["16384"], [1.25, 128])
        self.assertLess(self.probe.path.stat().st_size, 4000)


if __name__ == "__main__":
    unittest.main()
