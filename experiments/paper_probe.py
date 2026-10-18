#!/usr/bin/env python3
"""Paper-section-3 in-engine CUDA-event probe; original selector and HT only.

Load through the existing plugin, without changing moetime.py:
  PLACEMENT_PROBE=/work/router/experiments/paper_probe.py
  PAPER_PROBE_DIR=/results/paper/launch00/stock/probe
  PAPER_PROBE_CONTROL=/results/paper/launch00/stock/control.json
  PAPER_PROBE_EXPECTED_EP=4 PAPER_PROBE_EXPECTED_LAYERS=48
  PLACEMENT_EXPERIMENTAL_SELECTOR=baseline

The control file is atomically replaced by the driver and contains
{"phase":"warmup", "epoch":1, "round":0}. Phases are off/warmup/measure/drain;
epochs increase; drain is terminal. Change labels only between HTTP rounds and
wait for every worker's control_ack before sending the next round.

PRIMARY preserves moetime.py: global first96 forwards skipped, graph captures
skipped, compute-stream events around _forward_impl, prepare_async through its
receiver, and _finalize through its receiver. Every64 timed MoE forwards, span
ends synchronize and elapsed times reduce in original insertion order. This
deliberately preserves the paper's synchronization/cadence, not a sync-free timer.
All post96 token shapes and off/warmup/measure labels are retained. Exact T is
stored alongside the legacy power-of-two bucket; analysis must select T=16384.

A daemon normally only polls control/progress. After the ONE final drain it
queries completed tail events; it never synchronizes unfinished tail work.
Rows identify flush_origin chunk64/drain_tail. Do not drain between HTTP rounds.

Atomic rank{EP}_pid{PID}.json schema_version1:
 metadata: actual EP/DP rank (DP=EP, TP1), device, backend/graph/config, original
           selector, source hashes, timed method signatures, timing semantics.
 control_ack: phase/epoch/round; ready; drained; pending_spans; open_forwards;
 flush_in_progress; records_included; forward_records (final drain only): rows
 {forward_id, seen_forward, T, bucket, layer_name, layer_index, step_id,
  phase,epoch,round, moe_ms,dispatch_ms,combine_ms,flush_origin,valid,errors}.
 coverage.by_epoch: exact-T ready counts/layer and complete48-layer step IDs.
Timed chunk writes contain compact progress and original bucket medians/counts;
raw records remain in memory and are serialized only at the final idle drain.

GPU spans include preparation/reduction/metadata work and completion waits,
not pure transport time. Constant expert slots do not imply constant GEMM time.
No CUDA profiler, extra collective, or routing/cast instrumentation is enabled.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps
import hashlib
import inspect
import math
import json
import os
from pathlib import Path
import re
import statistics
import sys
import threading
import time

CHUNK = 64
WARMUP = 96
PHASES = ("moe", "dispatch", "combine")
CONTROL_PHASES = {"off", "warmup", "measure", "drain"}
_INSTANCE = None
ORIGINAL_KERNEL_SHA256 = "46fe57767d6b4230f311bc40f529bf04e1608bab68d627efd5dcf08fc7a75ff2"


def read_control(path):
    value = json.loads(path.read_text())
    phase, epoch, round_index = value.get("phase"), value.get("epoch"), value.get("round")
    if phase not in CONTROL_PHASES or type(epoch) is not int or epoch < 0 or type(round_index) is not int:
        raise ValueError("Control requires phase off/warmup/measure/drain, nonnegative integer epoch and integer round")
    return {"phase": phase, "epoch": epoch, "round": round_index}


def layer_index(name):
    match = re.search(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)", name)
    if not match:
        raise ValueError(f"Cannot identify decoder layer from {name!r}")
    return int(match.group(1))


def source_info(function):
    filename = inspect.getsourcefile(function)
    return {"qualified_name": function.__qualname__, "signature": str(inspect.signature(function)),
            "file": filename, "source_sha256": hashlib.sha256(inspect.getsource(function).encode()).hexdigest()}


def original_kernel_evidence(kernel, expected_sha=ORIGINAL_KERNEL_SHA256):
    digest = hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError("Paper reproduction requires pristine kernel.py from commit2d8f5192")
    if (getattr(kernel, "_EXPERIMENTAL_SELECTOR", "baseline") != "baseline"
            or getattr(kernel, "_EXPERIMENTAL_IMPL", None) is not None):
        raise RuntimeError("Actual imported routing implementation is experimental")
    return digest


def cache_runner_config(inner, get_config):
    """Retain the native construction config for compiled custom-op forwards.

    MoERunner's native constructor already requires a current vLLM config. Its
    compiled runtime custom op need not run inside that context. Keep the actual
    shared object, whose registered layer list is populated by later constructors;
    neither look up a global config nor copy an incomplete list during forward.
    """
    @wraps(inner)
    def initialized(runner, *args, **kwargs):
        result = inner(runner, *args, **kwargs)
        config = get_config()
        if config is None:
            raise RuntimeError("MoERunner construction has no vLLM config to cache")
        runner._paper_probe_vllm_config = config
        return result
    return initialized


def cached_runner_config(runner):
    config = getattr(runner, "_paper_probe_vllm_config", None)
    if config is None:
        raise RuntimeError("MoERunner has no construction-cached vLLM config")
    return config


class PaperProbe:
    """Injectable event/control core; CPU tests use the exact same wrappers."""

    def __init__(self, torch, output, control, metadata_factory, *, expected_layers=48,
                 exact_T=16384, start_thread=True, poll_seconds=.25, write_seconds=1.0):
        self.torch, self.output, self.control_path = torch, Path(output), Path(control)
        self.metadata_factory = metadata_factory
        self.expected_layers, self.exact_T = expected_layers, exact_T
        self.start_thread, self.poll_seconds, self.write_seconds = start_thread, poll_seconds, write_seconds
        self.lock, self.local = threading.RLock(), threading.local()
        self.control = read_control(self.control_path)
        self.metadata = None
        self.path = None
        self.thread = None
        self.seen = self.timed = self.since_flush = self.step_id = 0
        self.completed_forwards = 0
        self.open_forwards = 0
        self.flushing = False
        self.pending, self.rows, self.errors = [], [], []
        self.original_ms = {}
        self.seen_T, self.skips = Counter(), Counter()
        self.last_write = 0.
        self.current_step_control = None
        self.previous_layer = None
        self.layer_labels = {}
        self.stop_event = threading.Event()

    def ensure_runtime(self, runner, hidden):
        if self.metadata is not None:
            return
        self.metadata = self.metadata_factory(runner, hidden)
        rank, pid = self.metadata["ep_rank"], os.getpid()
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / f"rank{rank}_pid{pid}.json"
        self.write(force=True)
        print(f"[paper-probe] worker ready ep_rank={rank} pid={pid} "
              f"layers={self.expected_layers} selector={self.metadata.get('selector')} -> {self.path}", flush=True)
        if self.start_thread:
            self.thread = threading.Thread(target=self._daemon, name="paper-probe-control", daemon=True)
            self.thread.start()

    def refresh_control(self):
        value = read_control(self.control_path)
        with self.lock:
            if value["epoch"] < self.control["epoch"]:
                raise ValueError("Control epoch moved backwards")
            if value["epoch"] == self.control["epoch"] and value != self.control:
                raise ValueError("A changed phase/round requires a new epoch")
            if self.control["phase"] == "drain" and value != self.control:
                raise ValueError("Final drain is terminal; launch a fresh server")
            self.control = value

    def _error(self, message):
        with self.lock:
            if message not in self.errors:
                self.errors.append(message)

    def _daemon(self):
        try:
            self.torch.cuda.set_device(self.metadata["device_index"])
            while not self.stop_event.is_set():
                old = self.control.copy()
                self.refresh_control()
                if self.control["phase"] == "drain":
                    self.drain_completed_tail()
                    # Full raw history is an idle final-drain operation only.
                    self.write(force=old != self.control, include_records=self._drained())
                elif old != self.control:
                    self.write(force=True)
                self.stop_event.wait(self.poll_seconds)
        except BaseException as error:
            self._error(f"Control/drain daemon failed: {type(error).__name__}: {error}")
            self.write(force=True)

    def close(self):
        """Stop the observer thread without waiting on GPU work (CPU tests)."""
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def _stream(self):
        return self.torch.cuda.current_stream()

    @staticmethod
    def _stream_id(stream):
        return (getattr(getattr(stream, "device", None), "index", None),
                getattr(stream, "cuda_stream", stream))

    def start_span(self, row, phase):
        start, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        stream = self._stream()
        start.record(stream)
        row["span_calls"][phase] += 1
        return {"start": start, "end": end, "stream": self._stream_id(stream),
                "phase": phase, "row": row, "ended": False}

    def end_span(self, span):
        if span["ended"]:
            raise RuntimeError("An asynchronous receiver was called more than once")
        stream = self._stream()
        if self._stream_id(stream) != span["stream"]:
            raise RuntimeError("Probe requires one compute stream; callback changed streams")
        span["end"].record(stream)
        span["ended"] = True
        span["row"]["span_ends"][span["phase"]] += 1
        with self.lock:
            # Dispatch receiver, combine receiver, then MoE: same order as the
            # original probe's append/flush order, not sorted by phase.
            self.pending.append(span)

    def wrap_half(self, phase, inner):
        parameter_names = list(inspect.signature(inner).parameters)
        def argument(args, kwargs, name):
            if name in kwargs:
                return kwargs[name]
            position = parameter_names.index(name) if name in parameter_names else len(args)
            return args[position] if position < len(args) else None
        @wraps(inner)
        def wrapped(*args, **kwargs):
            row = getattr(self.local, "active", None)
            if row is None:
                return inner(*args, **kwargs)
            if phase == "dispatch":
                a1 = argument(args, kwargs, "a1")
                ids = argument(args, kwargs, "topk_ids")
                if a1 is not None:
                    row["dispatch_input_T"] = int(a1.shape[0])
                    if row["dispatch_input_T"] != row["T"]:
                        message = "Dispatch input T differs from _forward_impl hidden T"
                        row["errors"].append(message)
                        self._error(message)
                if ids is not None:
                    row["routing_ID_dtype"] = str(ids.dtype)
                    row["routing_ID_shape"] = list(ids.shape)
            span = self.start_span(row, phase)
            result = inner(*args, **kwargs)
            if not callable(result):
                self.end_span(span)
                return result

            @wraps(result)
            def receiver(*receiver_args, **receiver_kwargs):
                output = result(*receiver_args, **receiver_kwargs)
                self.end_span(span)
                return output
            return receiver
        return wrapped

    def wrap_forward(self, inner):
        @wraps(inner)
        def forward(runner, hidden_states, *args, **kwargs):
            # Both reads are metadata. Capture skip precedes runtime metadata,
            # daemon startup, output writing and any timing event allocation.
            capturing = self.torch.cuda.is_current_stream_capturing()
            T = int(hidden_states.shape[0])
            with self.lock:
                self.seen += 1
                seen = self.seen
                self.seen_T[str(T)] += 1
            if capturing:
                self.skips["capture"] += 1
                return inner(runner, hidden_states, *args, **kwargs)
            self.ensure_runtime(runner, hidden_states)
            key = id(runner)
            if key not in self.layer_labels:
                name = str(getattr(getattr(runner, "layer_name", ""), "value", getattr(runner, "layer_name", "")))
                parsed = layer_index(name)
                index = int(getattr(runner, "layer_id", parsed))
                if index != parsed:
                    raise RuntimeError("Native MoERunner.layer_id disagrees with its layer_name")
                self.layer_labels[key] = name, index
            name, index = self.layer_labels[key]
            if not 0 <= index < self.expected_layers:
                raise RuntimeError(f"Unexpected layer index {index}")
            with self.lock:
                if index == 0:
                    self.step_id += 1
                    self.current_step_control = self.control.copy()
                elif self.previous_layer != index - 1:
                    self._error(f"Nonsequential layer order: {self.previous_layer} -> {index}")
                self.previous_layer = index
                control = (self.current_step_control or self.control).copy()
                step = self.step_id
            if seen <= WARMUP:
                self.skips["first96"] += 1
                return inner(runner, hidden_states, *args, **kwargs)
            # Preserve every post96 shape, even outside the exact analysis T.
            # Drain never launches a new timed span; it is the final idle gate.
            if control["phase"] == "drain":
                self.skips["after_final_drain"] += 1
                return inner(runner, hidden_states, *args, **kwargs)
            row = {"forward_id": self.timed, "seen_forward": seen, "T": T,
                   "bucket": 1 << max(0, T.bit_length() - 1), "layer_name": name,
                   "layer_index": index, "step_id": step, **control,
                   "moe_ms": None, "dispatch_ms": None, "combine_ms": None,
                   "span_calls": {phase: 0 for phase in PHASES}, "errors": [],
                   "span_ends": {phase: 0 for phase in PHASES},
                   "closed": False, "valid": False, "flush_origin": None}
            with self.lock:
                self.timed += 1
                self.rows.append(row)
                self.open_forwards += 1
            previous = getattr(self.local, "active", None)
            self.local.active = row
            span = self.start_span(row, "moe")
            try:
                output = inner(runner, hidden_states, *args, **kwargs)
                if type(output).__name__ == "UnfinalizedMoEOutput" or (
                    isinstance(output, tuple) and any(type(v).__name__ == "UnfinalizedMoEOutput" for v in output)):
                    raise RuntimeError("Deferred MoE output is outside this timing contract")
            except BaseException as error:
                row["errors"].append(f"Forward failed: {type(error).__name__}: {error}")
                self._error(row["errors"][-1])
                with self.lock:
                    row["closed"] = True
                    self.open_forwards -= 1
                raise
            finally:
                self.local.active = previous
            self.end_span(span)
            with self.lock:
                row["closed"] = True
                self.open_forwards -= 1
            for phase in PHASES:
                if row["span_calls"][phase] != 1:
                    row["errors"].append(f"Expected one {phase} span, got {row['span_calls'][phase]}")
                    self._error(row["errors"][-1])
                if row["span_ends"][phase] != 1:
                    row["errors"].append(f"Expected one completed {phase} receiver/span, got {row['span_ends'][phase]}")
                    self._error(row["errors"][-1])
            self.since_flush += 1
            if self.since_flush >= CHUNK:
                self.since_flush = 0
                self.flush_primary()
            return output
        return forward

    def _reduce(self, spans, origin):
        for span in spans:
            value = float(span["start"].elapsed_time(span["end"]))
            row = span["row"]
            if not math.isfinite(value) or value < 0:
                message = f"Invalid {span['phase']} CUDA elapsed time: {value}"
                row["errors"].append(message)
                self._error(message)
                continue
            row[span["phase"] + "_ms"] = value
            row["flush_origin"] = origin
            self.original_ms.setdefault((span["phase"], row["bucket"]), []).append(value)
        for span in spans:
            row = span["row"]
            valid = row["closed"] and not row["errors"] and all(row[p + "_ms"] is not None for p in PHASES)
            if valid and not row["valid"]:
                self.completed_forwards += 1
            row["valid"] = valid

    def flush_primary(self):
        with self.lock:
            if self.flushing:
                raise RuntimeError("Overlapping probe flushes")
            self.flushing = True
            spans = self.pending
            self.pending = []
        try:
            # Intentionally identical to original moetime.py: synchronize each
            # recorded end in insertion order, then read that pair's elapsed.
            # Do not turn this into a different asynchronous timing protocol.
            for span in spans:
                span["end"].synchronize()
                self._reduce([span], "chunk64")
        finally:
            with self.lock:
                self.flushing = False
            self.write(force=True)

    def drain_completed_tail(self):
        with self.lock:
            if self.control["phase"] != "drain" or self.open_forwards or self.flushing:
                return
            ready, remaining = [], []
            for span in self.pending:
                (ready if span["end"].query() else remaining).append(span)
            self.pending = remaining
            self._reduce(ready, "drain_tail")

    def _drained(self):
        return (self.control["phase"] == "drain" and not self.pending and not self.open_forwards
                and not self.flushing and not self.errors and self.completed_forwards == self.timed)

    def snapshot(self, include_records=True):
        with self.lock:
            # Crucially, compact CHUNK64 snapshots never traverse raw history.
            rows = [dict(row) for row in self.rows] if include_records else None
            epochs = {}
            for row in rows or ():
                key = str(row["epoch"])
                group = epochs.setdefault(key, {"phase": row["phase"], "round": row["round"],
                    "exact_T_ready_forward_counts_by_layer": {}, "complete_step_ids": [], "_steps": {}})
                if row["valid"] and row["T"] == self.exact_T:
                    counts = group["exact_T_ready_forward_counts_by_layer"]
                    layer = str(row["layer_index"])
                    counts[layer] = counts.get(layer, 0) + 1
                    group["_steps"].setdefault(row["step_id"], set()).add(row["layer_index"])
            for group in epochs.values():
                group["complete_step_ids"] = sorted(step for step, layers in group.pop("_steps").items()
                                                     if layers == set(range(self.expected_layers)))
                counts = group["exact_T_ready_forward_counts_by_layer"]
                group["minimum_exact_T_ready_forwards_per_layer"] = min(counts.get(str(i), 0) for i in range(self.expected_layers))
            pending = len(self.pending)
            result = {"schema_version": 1, "metadata": self.metadata,
                "control_ack": self.control.copy(), "ready": self.metadata is not None and not self.errors,
                "drained": self._drained(), "records_included": include_records,
                "pending_spans": pending, "open_forwards": self.open_forwards,
                "flush_in_progress": self.flushing, "seen_forwards": self.seen,
                "timed_forwards": self.timed, "chunk_forward_remainder": self.since_flush,
                "completed_forwards": self.completed_forwards, "records_stored_in_memory": len(self.rows),
                "seen_T_counts": dict(self.seen_T), "skipped_forwards": dict(self.skips),
                "errors": self.errors.copy(), "coverage": {"exact_T": self.exact_T,
                    "expected_layers": self.expected_layers, "details_included": include_records, "by_epoch": epochs},
                "original_summary": {label: {str(bucket): [statistics.median(values), len(values)]
                    for (phase, bucket), values in sorted(self.original_ms.items()) if phase == name}
                    for name, label in (("moe", "buckets"), ("dispatch", "disp"), ("combine", "comb"))},
                "updated_unix_ns": time.time_ns()}
            if include_records:
                result["forward_records"] = rows
            return result

    def write(self, force=False, include_records=False):
        if self.path is None or not force and time.monotonic() - self.last_write < self.write_seconds:
            return
        # Only the daemon and primary flush write; serialize temp-file replace.
        with self.lock:
            deep_ep = sys.modules.get("deep_ep")
            if self.metadata is not None and deep_ep is not None:
                self.metadata["deep_ep_version"] = getattr(deep_ep, "__version__", None)
                self.metadata["deep_ep_buffer_num_sms"] = getattr(getattr(deep_ep, "Buffer", None), "num_sms", None)
            data = self.snapshot(include_records=include_records)
            temporary = self.path.with_suffix(f".json.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
            temporary.replace(self.path)
            self.last_write = time.monotonic()


def install():
    global _INSTANCE
    if _INSTANCE is not None:
        return
    selector = os.environ.get("PLACEMENT_EXPERIMENTAL_SELECTOR", "").strip().lower() or "baseline"
    if selector != "baseline":
        raise RuntimeError("Paper reproduction requires the original selector: baseline/unset")
    output = Path(os.environ["PAPER_PROBE_DIR"])
    control = Path(os.environ["PAPER_PROBE_CONTROL"])
    if not output.is_absolute() or not control.is_absolute():
        raise ValueError("PAPER_PROBE_DIR and PAPER_PROBE_CONTROL must be absolute")
    ep = int(os.environ.get("PAPER_PROBE_EXPECTED_EP", "4"))
    layers = int(os.environ.get("PAPER_PROBE_EXPECTED_LAYERS", "48"))
    exact_T = int(os.environ.get("PAPER_PROBE_TOKENS", "16384"))
    if ep not in (4, 8) or layers < 1 or exact_T != 16384:
        raise ValueError("Probe requires EP4/8, positive layer count and exact targetT16384")
    import torch
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize as P
    existing = getattr(MoERunner._forward_impl, "_paper_probe_instance", None)
    if existing is not None:
        if (existing.output != output or existing.control_path != control
                or getattr(MoERunner.__init__, "_paper_probe_instance", None) is not existing
                or any(getattr(getattr(P, name), "_paper_probe_instance", None) is not existing
                       for name in ("prepare_async", "_finalize"))):
            raise RuntimeError("A different/incomplete paper probe is already installed")
        _INSTANCE = existing
        return
    for cls, method in ((MoERunner, "__init__"), (MoERunner, "_forward_impl"),
                        (P, "prepare_async"), (P, "_finalize")):
        if method not in cls.__dict__ or getattr(getattr(cls, method), "_paper_probe", False):
            raise RuntimeError(f"Missing/already-wrapped {cls.__name__}.{method}")
        if not getattr(cls, method).__module__.startswith("vllm."):
            raise RuntimeError(f"Unexpected existing instrumentation on {cls.__name__}.{method}")
    methods = {"moe": MoERunner._forward_impl, "dispatch": P.prepare_async, "combine": P._finalize}
    sources = {name: source_info(method) for name, method in methods.items()}
    constructor = MoERunner.__init__
    config_source = source_info(constructor)

    def metadata_factory(runner, hidden):
        from vllm.distributed.parallel_state import get_ep_group
        from placement_router import kernel
        config, group = cached_runner_config(runner), get_ep_group()
        parallel = config.parallel_config
        if (group.world_size != ep or parallel.tensor_parallel_size != 1 or parallel.data_parallel_size != ep
                or parallel.pipeline_parallel_size != 1 or parallel.all2all_backend != "deepep_high_throughput"
                or bool(parallel.enable_dbo) or bool(parallel.use_ubatching) or int(parallel.ubatch_size) > 1):
            raise RuntimeError("Probe requires DP=EP4/8, TP1/PP1, HT, and all microbatching disabled")
        mode = getattr(config.compilation_config.cudagraph_mode, "name", str(config.compilation_config.cudagraph_mode))
        if mode != "NONE":
            raise RuntimeError(f"Paper HT reproduction requires actual CUDA graph mode NONE, got {mode}")
        kernel_sha = original_kernel_evidence(kernel)
        device = int(torch.cuda.current_device())
        model = config.model_config
        if model.quantization is not None or str(hidden.dtype) != "torch.bfloat16":
            raise RuntimeError("Paper reproduction scope is unquantized BF16")
        names = list(config.compilation_config.static_all_moe_layers)
        if len(names) != layers or {layer_index(str(name)) for name in names} != set(range(layers)):
            raise RuntimeError("Actual registered MoE layer coverage differs from expected decoder layers")
        if not hasattr(type(runner), "layer_id"):
            raise RuntimeError("Expected native MoERunner.layer_id property is absent")
        return {"pid": os.getpid(), "ep_rank": int(group.rank_in_group), "dp_rank": int(group.rank_in_group),
            "rank_basis": "get_ep_group.rank_in_group; assertedDP=EP andTP1", "ep_world_size": int(group.world_size),
            "device_index": device, "gpu": torch.cuda.get_device_name(device),
            "activation_dtype": str(hidden.dtype), "hidden_size": int(hidden.shape[-1]),
            "torch_version": str(torch.__version__),
            "backend": parallel.all2all_backend, "cudagraph_mode": mode, "model": model.model,
            "registered_MoE_layer_names": names, "selector": "baseline", "experimental_impl_is_none": True,
            "layer_index_basis": "Native MoERunner.layer_id, cached and crosschecked against layer_name",
            "arm": os.environ.get("PAPER_PROBE_ARM", os.environ.get("ARM", "stock")),
            "repeat": os.environ.get("PAPER_PROBE_REPEAT", os.environ.get("REP", "0")),
            "tag": os.environ.get("PAPER_PROBE_TAG", os.environ.get("TAG", "paper")),
            "placement_keep": os.environ.get("PLACEMENT_KEEP"), "load_penalty": os.environ.get("PLACEMENT_LOAD_PENALTY"),
            "timing_semantics": {"primary": "original CHUNK64 end.synchronize insertion-order reduction", "warmup_global_forwards": WARMUP,
                "flush_whole_forwards": CHUNK, "exact_target_T": exact_T, "all_actual_T_retained": True,
                "dispatch": "prepare_async entry through receiver completion, including prepare/metadata work and waits",
                "combine": "_finalize entry through receiver completion, including reduction/output copy and waits",
                "tail": "only terminal drain queries completed ends; flush_origin=drain_tail"},
            "timed_methods": sources, "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config_acquisition": {"basis": "native_MoERunner_construction_cached_object",
                "runtime_global_context_lookup": False, "constructor": config_source},
            "kernel_source_sha256": kernel_sha, "pristine_source_commit": "2d8f5192ba195ec1fd3e62a259ce67c43cafa4c3"}

    probe = PaperProbe(torch, output, control, metadata_factory, expected_layers=layers, exact_T=exact_T)
    MoERunner.__init__ = cache_runner_config(constructor, get_current_vllm_config)
    MoERunner._forward_impl = probe.wrap_forward(methods["moe"])
    P.prepare_async = probe.wrap_half("dispatch", methods["dispatch"])
    P._finalize = probe.wrap_half("combine", methods["combine"])
    for method in (MoERunner.__init__, MoERunner._forward_impl, P.prepare_async, P._finalize):
        method._paper_probe = True
        method._paper_probe_instance = probe
    _INSTANCE = probe
    print(f"[paper-probe] installed original selector, globalwarmup={WARMUP}, chunk={CHUNK}; "
          f"actualT=16384, EP={ep}, layers={layers}, control={control}", flush=True)
