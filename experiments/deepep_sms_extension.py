"""Standalone opt-in DeepEP HT SM-budget experiment for one-node EP4.

Load in BOTH matched arms, before worker/device initialization:
  PYTHONPATH=/work/router/experiments \
  PLACEMENT_DEEPEP_NUM_SMS=40 \
  PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR=/results/experiment/sms-evidence \
  vllm serve ... --worker-extension-cls deepep_sms_extension.DeepEPSMSWorkerExtension

Import only patches the HT kwargs builder. DeepEP/CUDA are first queried at
get_handle(), immediately before the original handle-cache buffer allocation.
No production files are changed. Current scope: DeepEP2.0 legacy HT, one node,
EP4, unquantized floating-point model, all microbatching disabled. The requested budget is a
process-wide DeepEP setting and must match on every rank. Every worker emits
one JSON evidence line (and an optional atomic JSON file) before allocation.
"""
from __future__ import annotations

from functools import wraps
import importlib
import json
import logging
import os
from pathlib import Path
import time

_LOG = logging.getLogger(__name__)
_ENV = "PLACEMENT_DEEPEP_NUM_SMS"
_STATE = "_placement_deepep_sms_experiment"


class DeepEPSMSWorkerExtension:
    """Empty extension: module import installs the experimental startup hook."""


def _requested_budget():
    raw = os.environ.get(_ENV)
    if raw not in ("20", "40", "64"):
        raise RuntimeError(f"{_ENV} must explicitly be 20, 40, or 64 when loading this extension")
    return int(raw)


def _emit(evidence):
    encoded = json.dumps(evidence, sort_keys=True, allow_nan=False)
    directory = os.environ.get("PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR")
    if directory:
        directory = Path(directory)
        if not directory.is_absolute():
            raise RuntimeError("PLACEMENT_DEEPEP_SMS_EVIDENCE_DIR must be an absolute path")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank{evidence['ep_rank']}_pid{evidence['pid']}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(encoded + "\n")
        temporary.replace(path)
    _LOG.warning("placement-deepep-sms %s", encoded)


def install():
    budget = _requested_budget()
    managers = importlib.import_module("vllm.distributed.device_communicators.all2all")
    cls = managers.DeepEPHTAll2AllManager
    original = cls._make_all2all_kwargs
    previous = getattr(original, "_placement_sms_extension_budget", None)
    if previous is not None:
        if previous != budget:
            raise RuntimeError("DeepEP SM extension is already installed with another budget")
        return

    @wraps(original)
    def configured_kwargs(self):
        # The original get_handle lazily imports DeepEP before reaching this
        # method. Do not import DeepEP/CUDA during extension registration.
        with self.handle_cache._lock:
            state = getattr(self, _STATE, None)
            if state is not None:
                if self.num_sms != budget:
                    raise RuntimeError("DeepEP manager SM ceiling changed after configuration")
                return state["kwargs"].copy()
            if self.world_size != 4 or self.internode:
                raise RuntimeError("DeepEP SM extension supports only one-node EP4")
            if self.handle_cache._cache:
                raise RuntimeError("DeepEP SM extension must run before any HT buffer allocation")

            deep_ep = importlib.import_module("deep_ep")
            torch = importlib.import_module("torch")
            config = importlib.import_module("vllm.config").get_current_vllm_config()
            if deep_ep.__version__ != "2.0.0":
                raise RuntimeError("DeepEP SM extension requires the verified DeepEP2.0.0 legacy API")
            if config.parallel_config.all2all_backend != "deepep_high_throughput":
                raise RuntimeError("DeepEP SM extension requires the HT backend")
            enable_dbo = bool(config.parallel_config.enable_dbo)
            ubatch_size = int(config.parallel_config.ubatch_size)
            use_ubatching = bool(config.parallel_config.use_ubatching)
            if enable_dbo or ubatch_size > 1 or use_ubatching:
                raise RuntimeError("DeepEP SM extension requires all microbatching disabled")
            model = config.model_config
            if model.quantization is not None:
                raise RuntimeError("DeepEP SM extension currently supports unquantized models")
            hidden = int(model.hf_text_config.hidden_size)
            itemsize = int(model.dtype.itemsize)
            if hidden <= 0 or itemsize not in (2, 4):
                raise RuntimeError("Unsupported model hidden size or activation dtype")
            # Same 512-byte alignment as HT.maybe_roundup_layer_hidden_size.
            hidden_bytes = ((hidden * itemsize + 511) // 512) * 512
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            if budget > properties.multi_processor_count:
                raise RuntimeError("DeepEP SM budget exceeds this GPU's physical SM count")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            previous_manager_sms, previous_buffer_sms = self.num_sms, deep_ep.Buffer.num_sms
            try:
                self.num_sms = budget
                deep_ep.Buffer.set_num_sms(budget)
                kwargs = original(self)
                if (kwargs["num_rdma_bytes"] != 0 or kwargs["num_qps_per_rank"] != 1
                        or kwargs["low_latency_mode"]):
                    raise RuntimeError("Unexpected non-intranode HT buffer configuration")
                dispatch_hint = int(deep_ep.Buffer.get_dispatch_config(4).get_nvl_buffer_size_hint(hidden_bytes, 4))
                combine_hint = int(deep_ep.Buffer.get_combine_config(4).get_nvl_buffer_size_hint(hidden_bytes, 4))
                requested_bytes = int(kwargs["num_nvl_bytes"])
                if min(requested_bytes, dispatch_hint, combine_hint) < 0:
                    raise RuntimeError("DeepEP returned an invalid buffer size")
                allocated_bytes = max(requested_bytes, dispatch_hint, combine_hint)
                if allocated_bytes <= 0:
                    raise RuntimeError("DeepEP HT buffer size must be positive")
                # Buffer construction is next. Leave half the currently free
                # memory for the model, KV cache and allocator transient work.
                kwargs["num_nvl_bytes"] = allocated_bytes
                signature = {"num_sms": budget, "hidden_bytes": hidden_bytes,
                             "num_nvl_bytes": allocated_bytes, "world_size": 4}
                memory_error = (f"DeepEP HT buffer {allocated_bytes} bytes exceeds half of free GPU memory {free_bytes}"
                                if allocated_bytes > free_bytes // 2 else None)
                preflight = {"signature": signature, "error": memory_error}
                gathered = [None] * 4
                torch.distributed.all_gather_object(gathered, preflight, group=self.cpu_group)
                errors = [value["error"] for value in gathered if value["error"]]
                if errors:
                    raise RuntimeError("; ".join(errors))
                if any(value["signature"] != signature for value in gathered):
                    raise RuntimeError("DeepEP SM/buffer configuration differs across EP ranks")
                evidence = {"kind": "deepep_ht_sm_startup", "unix_ns": time.time_ns(),
                    "pid": os.getpid(), "ep_rank": self.rank, "ep_world_size": 4,
                    "internode": False, "device_index": device, "gpu": properties.name,
                    "gpu_physical_sms": properties.multi_processor_count,
                    "requested_sms": budget, "manager_sms_ceiling": self.num_sms,
                    "buffer_num_sms": deep_ep.Buffer.num_sms,
                    "enable_dbo": enable_dbo, "ubatch_size": ubatch_size,
                    "use_ubatching": use_ubatching,
                    "hidden_size": hidden, "model_dtype": str(model.dtype),
                    "padded_hidden_bytes": hidden_bytes,
                    "nvl_requested_bytes": requested_bytes,
                    "nvl_dispatch_size_hint_bytes": dispatch_hint,
                    "nvl_combine_size_hint_bytes": combine_hint,
                    "nvl_allocated_bytes": allocated_bytes,
                    "nvl_auto_expanded": allocated_bytes > requested_bytes,
                    "rdma_allocated_bytes": 0, "num_qps_per_rank": 1,
                    "free_gpu_bytes_before_allocation": free_bytes, "total_gpu_bytes": total_bytes,
                    "deep_ep": deep_ep.__version__, "torch": torch.__version__,
                    "placement_keep": os.environ.get("PLACEMENT_KEEP"),
                    "placement_selector": os.environ.get("PLACEMENT_EXPERIMENTAL_SELECTOR"),
                    "buffer_allocation_has_not_run_yet": True}
                _emit(evidence)
                setattr(self, _STATE, {"kwargs": kwargs.copy(), "evidence": evidence})
                return kwargs
            except BaseException:
                # No HT allocation has happened: preserve prior configuration
                # if a preflight, size check, collective or evidence write fails.
                self.num_sms = previous_manager_sms
                deep_ep.Buffer.set_num_sms(previous_buffer_sms)
                raise

    configured_kwargs._placement_sms_extension_budget = budget
    cls._make_all2all_kwargs = configured_kwargs


install()
