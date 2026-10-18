"""CPU protocol tests; fake the external runtime without importing DeepEP/CUDA."""
from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import uuid

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "deepep_sms_extension.py"


@pytest.fixture
def runtime(monkeypatch):
    events = []

    class Hint:
        def __init__(self, label):
            self.label = label

        def get_nvl_buffer_size_hint(self, hidden_bytes, world):
            events.append(("hint", self.label, Buffer.num_sms, hidden_bytes, world))
            return 2000 if self.label == "dispatch" else 3000

    class Buffer:
        num_sms = 20

        @staticmethod
        def set_num_sms(value):
            events.append(("set_sms", value))
            Buffer.num_sms = value

        @staticmethod
        def get_dispatch_config(world):
            assert world == 4
            return Hint("dispatch")

        @staticmethod
        def get_combine_config(world):
            assert world == 4
            return Hint("combine")

    class Manager:
        def __init__(self):
            self.world_size, self.rank, self.internode, self.num_sms = 4, 0, False, 20
            self.cpu_group = object()
            self.handle_cache = NS(_cache={}, _lock=threading.RLock())
            self.make_calls = 0

        def _make_all2all_kwargs(self):
            self.make_calls += 1
            events.append(("make_kwargs", self.num_sms, Buffer.num_sms))
            return {"group": self.cpu_group, "num_nvl_bytes": 1000,
                    "num_rdma_bytes": 0, "num_qps_per_rank": 1,
                    "low_latency_mode": False, "explicitly_destroy": True}

    config = NS(parallel_config=NS(all2all_backend="deepep_high_throughput", enable_dbo=False,
                                  ubatch_size=0, use_ubatching=False),
                model_config=NS(quantization=None, hf_text_config=NS(hidden_size=2048),
                                dtype=NS(itemsize=2)))
    props = NS(name="CPU fake H200", multi_processor_count=132)
    memory = NS(free=10000, total=20000)

    def gather(out, value, group):
        events.append(("gather", value))
        out[:] = [value.copy() for _ in range(4)]

    torch = NS(__version__="fake", cuda=NS(current_device=lambda: 0,
               get_device_properties=lambda device: props,
               mem_get_info=lambda device: (memory.free, memory.total)),
               distributed=NS(all_gather_object=gather))
    deep_ep = NS(__version__="2.0.0", Buffer=Buffer)
    monkeypatch.setitem(sys.modules, "vllm.distributed.device_communicators.all2all",
                        NS(DeepEPHTAll2AllManager=Manager))
    monkeypatch.setitem(sys.modules, "vllm.config", NS(get_current_vllm_config=lambda: config))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    monkeypatch.delenv("PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR", raising=False)

    def load(budget="40"):
        if budget is None:
            monkeypatch.delenv("PLACEMENT_DEEPEP_NUM_SMS", raising=False)
        else:
            monkeypatch.setenv("PLACEMENT_DEEPEP_NUM_SMS", budget)
        name = "sms_protocol_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return NS(load=load, Manager=Manager, Buffer=Buffer, events=events,
              props=props, memory=memory, config=config, torch=torch, deep_ep=deep_ep)


@pytest.mark.parametrize("budget", [None, "", "0", "21", "32", "forty"])
def test_budget_must_be_explicit_and_supported(runtime, budget):
    with pytest.raises(RuntimeError, match="must explicitly be 20, 40, or 64"):
        runtime.load(budget)
    assert runtime.events == []


@pytest.mark.parametrize("budget", ["20", "40", "64"])
def test_import_is_gpu_inert_and_extension_class_is_empty(runtime, budget):
    module = runtime.load(budget)
    assert runtime.events == []
    assert not [key for key in vars(module.DeepEPSMSWorkerExtension) if not key.startswith("__")]


def test_configures_before_builder_expands_hint_caches_and_logs_once(runtime, caplog, monkeypatch, tmp_path):
    module = runtime.load()
    monkeypatch.setenv("PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR", str(tmp_path))
    manager = runtime.Manager()
    with caplog.at_level(logging.WARNING):
        first = manager._make_all2all_kwargs()
        manager.handle_cache._cache["allocated"] = object()
        first["num_nvl_bytes"] = 17  # caller cannot corrupt the cached startup config
        second = manager._make_all2all_kwargs()
        module.install()  # idempotent patch installation
    assert manager.num_sms == runtime.Buffer.num_sms == 40
    assert second["num_nvl_bytes"] == 3000
    assert manager.make_calls == 1
    assert runtime.events[:4] == [("set_sms", 40), ("make_kwargs", 40, 40),
                                ("hint", "dispatch", 40, 4096, 4),
                                ("hint", "combine", 40, 4096, 4)]
    messages = [record.message for record in caplog.records if "placement-deepep-sms " in record.message]
    assert len(messages) == 1
    evidence = json.loads(messages[0].split("placement-deepep-sms ", 1)[1])
    assert evidence["requested_sms"] == 40
    assert evidence["nvl_allocated_bytes"] == 3000
    assert evidence["nvl_auto_expanded"]
    assert evidence["buffer_allocation_has_not_run_yet"]
    assert (evidence["enable_dbo"], evidence["ubatch_size"], evidence["use_ubatching"]) == (False, 0, False)
    files = list(tmp_path.glob("rank0_pid*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text()) == evidence


@pytest.mark.parametrize("attribute,value,match", [
    ("world_size", 8, "only one-node EP4"),
    ("internode", True, "only one-node EP4"),
])
def test_rejects_unsupported_topology_before_configuration(runtime, attribute, value, match):
    runtime.load()
    manager = runtime.Manager()
    setattr(manager, attribute, value)
    with pytest.raises(RuntimeError, match=match):
        manager._make_all2all_kwargs()
    assert runtime.events == []


def test_rejects_preexisting_buffers(runtime):
    runtime.load()
    manager = runtime.Manager()
    manager.handle_cache._cache["buffer"] = object()
    with pytest.raises(RuntimeError, match="before any HT buffer allocation"):
        manager._make_all2all_kwargs()
    assert runtime.events == []


def test_rejects_budget_larger_than_gpu(runtime):
    runtime.load("64")
    runtime.props.multi_processor_count = 40
    manager = runtime.Manager()
    with pytest.raises(RuntimeError, match="physical SM count"):
        manager._make_all2all_kwargs()
    assert runtime.events == []


@pytest.mark.parametrize("case", ["dbo", "ubatch_size", "use_ubatching", "quantization", "version"])
def test_unsupported_runtime_is_explicit(runtime, case):
    runtime.load()
    if case == "dbo":
        runtime.config.parallel_config.enable_dbo = True
    elif case == "ubatch_size":
        runtime.config.parallel_config.ubatch_size = 2
    elif case == "use_ubatching":
        runtime.config.parallel_config.use_ubatching = True
    elif case == "quantization":
        runtime.config.model_config.quantization = "fp8"
    else:
        runtime.deep_ep.__version__ = "other"
    with pytest.raises(RuntimeError):
        runtime.Manager()._make_all2all_kwargs()
    assert runtime.events == []


def test_memory_guard_is_collective_and_restores_state(runtime):
    runtime.load()
    manager = runtime.Manager()
    runtime.memory.free = 5000  # 3000-byte hint exceeds half-free budget
    with pytest.raises(RuntimeError, match="exceeds half of free GPU memory"):
        manager._make_all2all_kwargs()
    assert any(event[0] == "gather" for event in runtime.events)
    assert manager.num_sms == runtime.Buffer.num_sms == 20
    assert not hasattr(manager, "_placement_deepep_sms_experiment")


def test_rank_configuration_mismatch_restores_state(runtime):
    runtime.load()
    manager = runtime.Manager()

    def mismatch(out, value, group):
        out[:] = [value.copy() for _ in range(4)]
        out[3] = {"signature": value["signature"] | {"num_sms": 64}, "error": None}

    runtime.torch.distributed.all_gather_object = mismatch
    with pytest.raises(RuntimeError, match="differs across EP ranks"):
        manager._make_all2all_kwargs()
    assert manager.num_sms == runtime.Buffer.num_sms == 20
    assert not hasattr(manager, "_placement_deepep_sms_experiment")


def test_cannot_install_another_budget_or_raise_ceiling_after_configuration(runtime, monkeypatch):
    module = runtime.load()
    manager = runtime.Manager()
    manager._make_all2all_kwargs()
    monkeypatch.setenv("PLACEMENT_DEEPEP_NUM_SMS", "64")
    with pytest.raises(RuntimeError, match="another budget"):
        module.install()
    manager.num_sms = 64
    with pytest.raises(RuntimeError, match="ceiling changed"):
        manager._make_all2all_kwargs()


def test_unwritable_evidence_configuration_rolls_back_before_allocation(runtime, monkeypatch):
    runtime.load()
    manager = runtime.Manager()
    monkeypatch.setenv("PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR", "relative-path")
    with pytest.raises(RuntimeError, match="absolute path"):
        manager._make_all2all_kwargs()
    assert manager.num_sms == runtime.Buffer.num_sms == 20
    assert not hasattr(manager, "_placement_deepep_sms_experiment")
