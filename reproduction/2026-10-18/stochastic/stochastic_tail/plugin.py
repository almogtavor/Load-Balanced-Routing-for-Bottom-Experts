"""Separate experiment entrypoint for nine honest actual-callable contracts.

Public files are lineage, not evidence that the private selector is native PAR.
All quality arms, including stock/PAR controls, use the same optional CPU call
ledger. Timed experiments must explicitly disable that ledger and remain separate.
"""
from __future__ import annotations
import atexit
from collections import Counter
from contextvars import ContextVar
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time

from .reference import MASK, parameters

PUBLIC_COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
CONFIG_SHA = "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438"
PUBLIC = {
    "placement_router/__init__.py": "8425741d4c8d90b5ce2ce122cff4abb8a095b8e488ce9e18ce468a0916f5d0cf",
    "placement_router/kernel.py": "06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854",
    "placement_router/plugin.py": "c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8",
    "placement_router/vendor/__init__.py": "623bf62f4ae70782a32c1691d3648c120c67c5e4853725d3881265b7dfd96b25",
    "placement_router/vendor/fused_topk_router.py": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc"}
MODES = ("stock", "par", "uniform_hot", "weighted_hot", "uniform_range")
EXPECTED_LAYERS = {f"model.layers.{i}.mlp.experts" for i in range(48)}
NATIVE_FILES = {"vllm/model_executor/layers/fused_moe/router/fused_topk_router.py",
                "vllm/model_executor/layers/fused_moe/router/base_router.py",
                "vllm/model_executor/layers/fused_moe/runner/moe_runner.py", "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/worker/gpu/model_runner.py",
                "vllm/v1/worker/gpu_worker.py", "vllm/config/vllm.py", "vllm/v1/core/sched/output.py",
                "vllm/entrypoints/serve/dev/rlhf/api_router.py", "vllm/entrypoints/serve/dev/rpc/api_router.py"}
_CONFIG = None
_REGISTERED = False
_CONTROL = None
_STEP = ContextVar("stochastic_tail_experiment_step", default=None)


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_hashes():
    root = Path(__file__).resolve().parent
    return {"stochastic_tail/" + name: sha(root / name)
            for name in ["__init__.py", "reference.py", "kernel.py", "plugin.py", "native_hooks.py", "worker_extension.py"]}


def read_plan(path, expected_sha):
    need(sha(path) == expected_sha, "Experimental launch plan changed")
    value = json.loads(Path(path).read_text())
    p = value["parameters"]
    need(p["mode"] in MODES and p["seed"] in (0, 1, 2) and p["range_limit"] == 32
         and p["temperature"] == 1.0 and p["ranks"] == 4, "Not frozen nine-arm/three-seed protocol")
    need((p["mode"] == "stock" and p["keep"] is None) or (p["mode"] != "stock" and p["keep"] in (4, 6)), "Invalid arm head")
    need((p["mode"] == "par" and p["load_penalty"] in (0.0, 8.0))
         or (p["mode"] != "par" and p["load_penalty"] is None), "Wrong deterministic price or stochastic arm price")
    need(value["private_source_sha256"] == private_hashes(), "Private overlay source drift")
    need(set(value["native_source_files"]) == set(value["native_source_sha256"]) == NATIVE_FILES
         and all(sha(path) == value["native_source_sha256"][name] for name, path in value["native_source_files"].items()),
         "Native runtime source drift")
    need(Path(__file__).resolve().parent == (Path(value["private_site"]) / "stochastic_tail").resolve(),
         "Private entrypoint imported outside the frozen overlay site")
    need(value["model_id"] == MODEL_ID and value["revision"] == REVISION, "Wrong pinned Qwen checkpoint")
    lineage_path = Path(value["public_source_proof"]["path"])
    need(sha(lineage_path) == value["public_source_proof"]["sha256"], "Public lineage proof changed")
    lineage = json.loads(lineage_path.read_text())
    need(lineage["source_commit"] == PUBLIC_COMMIT
         and Path(lineage["site_root"]).resolve() == Path(value["public_site"]).resolve(), "Public lineage proof/site differs")
    checkpoint_path = Path(value["checkpoint_proof"]["path"])
    need(sha(checkpoint_path) == value["checkpoint_proof"]["sha256"], "Checkpoint proof changed")
    checkpoint = json.loads(checkpoint_path.read_text())
    need(checkpoint.get("canonical_identity_verified") is True and checkpoint["model_id"] == MODEL_ID
         and checkpoint["revision"] == REVISION and checkpoint["canonical_config_sha256"] == CONFIG_SHA
         and Path(checkpoint["local_model_path"]).resolve() == Path(value["model_path"]).resolve()
         and checkpoint["weights_verified"] == checkpoint["safetensors"]
         and len(checkpoint["safetensors"]) == len({f["filename"] for f in checkpoint["safetensors"]}) == 16
         and sha(Path(value["model_path"]) / "config.json") == CONFIG_SHA,
         "Fresh complete pinned checkpoint proof missing")
    need({f["filename"]: [(Path(value["model_path"]) / f["filename"]).stat().st_size,
                          (Path(value["model_path"]) / f["filename"]).stat().st_mtime_ns]
          for f in checkpoint["safetensors"]} == value["checkpoint_stat_signatures"]
         and all(value["checkpoint_stat_signatures"][f["filename"]][0] == f["bytes"] for f in checkpoint["safetensors"]),
         "Canonical model shard stat identity changed")
    gate_path = Path(value["gpu_gate"]["path"])
    need(sha(gate_path) == value["gpu_gate"]["sha256"], "GPU gate proof changed")
    gate = json.loads(gate_path.read_text())
    need(gate.get("complete") is True and gate.get("cuda_required") is True
         and gate.get("private_source_sha256") == private_hashes()
         and all(gate.get("checks", {}).get(name) is True for name in [
             "native_head", "distinct_experts", "dtype_and_sentinel", "original_weights",
             "ragged_and_padding", "rng_words_and_repeat", "fanout", "ties_and_range"]),
         "Fresh complete same-source GPU correctness gate required before any serving")
    need(value["record_calls"] is (value["purpose"] in ("quality", "preflight", "proof")),
         "Only explicitly separate timing may disable the CPU call ledger")
    need(value["purpose"] in ("quality", "preflight", "proof", "prefill_timing"), "Unknown experimental purpose")
    need(value["runtime"] == {"ep": 4, "dp": 4, "tp": 1, "pp": 1,
         "max_model_len": 8192, "max_num_seqs": 256, "max_num_batched_tokens": 16384,
         "graphs": "NONE", "prefix_caching": False, "all2all_backend": "deepep_high_throughput",
         "dtype": "torch.bfloat16", "expert_placement": "linear", "seed": 0,
         "logprobs_mode": "raw_logprobs", "use_v2_model_runner": True,
         "worker_extension_cls": "stochastic_tail.worker_extension.CheckpointExtension", "server_dev_mode": True}, "Runtime differs from the frozen quality protocol")
    return value


def callable_identity(function):
    code = function.__code__
    path = Path(code.co_filename).resolve()
    return {"module": function.__module__, "qualname": function.__qualname__,
            "code_filename": str(path), "source_sha256": sha(path),
            "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest()}


def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
    from vllm.distributed.parallel_state import get_ep_group
    from placement_router.plugin import _unsupported
    from placement_router.vendor.fused_topk_router import _get_padding_mask
    from .kernel import select
    config = _CONFIG["parameters"]
    if not self.__dict__.get("_stochastic_supported"):
        why = _unsupported(self, router_logits, config["keep"])
        need(why is None, "Stochastic selector unsupported here: " + str(why))
        self._stochastic_supported = True
    need(hasattr(self, "_stochastic_layer_index") and hasattr(self, "_stochastic_current_call"),
         "Native layer/call RNG identity missing")
    return select(router_logits, self.top_k, config["keep"], config["ranks"], mode=config["mode"],
                  seed=config["seed"], layer=self._stochastic_layer_index, call=self._stochastic_current_call,
                  range_limit=config["range_limit"], temperature=config["temperature"],
                  is_padding=_get_padding_mask(router_logits.shape[0]), indices_dtype=indices_type,
                  physical_rank=int(get_ep_group().rank_in_group))


def register():
    global _CONFIG, _REGISTERED, _CONTROL
    need(not _REGISTERED, "Experimental entrypoint must install once per process")
    _REGISTERED = True
    plan_path = Path(os.environ["STOCHASTIC_EXPERIMENT_PLAN"])
    plan_sha = os.environ["STOCHASTIC_EXPERIMENT_PLAN_SHA256"]
    config = _CONFIG = read_plan(plan_path, plan_sha)
    p = config["parameters"]
    need(os.environ.get("VLLM_PLUGINS") == "stochastic_tail_experiment", "Select only the isolated experiment entrypoint")
    need(not os.environ.get("PLACEMENT_PROBE"), "External routing instrumentation is not part of this experiment")
    for name, expected in PUBLIC.items():
        need(sha(Path(config["public_site"]) / name) == expected, "Pristine lineage file changed: " + name)
    from placement_router import plugin as public_plugin
    import placement_router
    import placement_router.kernel as public_kernel_module
    import placement_router.vendor as public_vendor
    import placement_router.vendor.fused_topk_router as public_router
    actual_public = {"placement_router/__init__.py": placement_router.__file__,
        "placement_router/plugin.py": public_plugin.__file__, "placement_router/kernel.py": public_kernel_module.__file__,
        "placement_router/vendor/__init__.py": public_vendor.__file__,
        "placement_router/vendor/fused_topk_router.py": public_router.__file__}
    need(all(Path(path).resolve() == (Path(config["public_site"]) / name).resolve()
             and sha(path) == PUBLIC[name] for name, path in actual_public.items()), "Actual pristine lineage imports differ")
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter
    need(not getattr(BaseRouter, "_placement_patched", False)
         and not getattr(BaseRouter, "_stochastic_experiment_patched", False), "Another selector already patched the process")
    need(FusedTopKRouter._compute_routing.__module__ == "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
         "Expected pristine native compute callable before private overlay")
    if p["mode"] == "par":
        need(os.environ.get("PLACEMENT_KEEP") == str(p["keep"])
             and float(os.environ.get("PLACEMENT_LOAD_PENALTY", "nan")) == p["load_penalty"], "PAR controls require explicit frozen price0/8")
        public_plugin.register()
    else:
        need(not os.environ.get("PLACEMENT_KEEP") and not os.environ.get("PLACEMENT_LOAD_PENALTY"),
             "Native/stochastic arms must not enable the public PAR patch")
        if p["mode"] != "stock":
            FusedTopKRouter._compute_routing = _compute_routing
    BaseRouter._stochastic_experiment_patched = True
    output = Path(config["evidence_directory"])
    output.mkdir(parents=True, exist_ok=True)
    routes, counters, proof_written = {}, {}, False
    lock = threading.Lock()

    def emit(kind, **fields):
        from vllm.distributed.parallel_state import get_ep_group
        try:
            rank = int(get_ep_group().rank_in_group)
        except AssertionError:
            rank = None
        row = {"kind": kind, "pid": os.getpid(), "ep_rank": rank, "unix_ns": time.time_ns(),
               "plan_sha256": plan_sha, "parameters": p, **fields}
        raw = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with lock:
            fd = os.open(output / f"pid{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                need(os.write(fd, raw) == len(raw), "Incomplete experimental evidence write")
            finally:
                os.close(fd)
        return {"row":row,"record_sha256":hashlib.sha256(raw).hexdigest(),
                "ledger_path":str(output / f"pid{os.getpid()}.jsonl")}

    constructor, route = MoERunner.__init__, BaseRouter._select_experts
    need(route.__module__.startswith("vllm."), "Unexpected native route wrapper")
    holder = {}

    @wraps(constructor)
    def constructed(self, *args, **kwargs):
        from vllm.config import get_current_vllm_config
        result = constructor(self, *args, **kwargs)
        need(self.layer_name in EXPECTED_LAYERS, "Expected named48-layer Qwen MoE")
        index = int(re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts", self.layer_name).group(1))
        need(index not in counters, "Native layer constructed twice; RNG identity would be ambiguous")
        self.router._stochastic_layer_index = index
        self.router._stochastic_layer_name = self.layer_name
        self.router._stochastic_call_count = 0
        counters[index] = self.router
        holder["config"] = get_current_vllm_config()
        return result

    def snapshot(reason, **metadata):
        if counters:
            emit("counter_snapshot", reason=reason,
                 counters={str(i): r._stochastic_call_count for i, r in sorted(counters.items())}, **metadata)

    def identity(hidden):
        import torch
        import vllm
        from vllm.distributed.parallel_state import get_ep_group
        from placement_router import kernel as public_kernel
        runtime = holder["config"]
        parallel = runtime.parallel_config
        actual = {"ep": int(get_ep_group().world_size), "dp": int(parallel.data_parallel_size),
                  "tp": int(parallel.tensor_parallel_size), "pp": int(parallel.pipeline_parallel_size),
                  "max_model_len": int(runtime.model_config.max_model_len),
                  "max_num_seqs": int(runtime.scheduler_config.max_num_seqs),
                  "max_num_batched_tokens": int(runtime.scheduler_config.max_num_batched_tokens),
                  "graphs": getattr(runtime.compilation_config.cudagraph_mode, "name", str(runtime.compilation_config.cudagraph_mode)),
                  "prefix_caching": bool(runtime.cache_config.enable_prefix_caching),
                  "all2all_backend": parallel.all2all_backend, "dtype": str(hidden.dtype),
                  "expert_placement": parallel.expert_placement_strategy, "seed": runtime.model_config.seed,
                  "logprobs_mode": runtime.model_config.logprobs_mode,
                  "use_v2_model_runner": bool(runtime.use_v2_model_runner),
                  "worker_extension_cls": parallel.worker_extension_cls,
                  "server_dev_mode": os.environ.get("VLLM_SERVER_DEV_MODE") == "1"}
        need(actual == config["runtime"] and not parallel.enable_dbo and not parallel.use_ubatching
             and parallel.enable_expert_parallel, "Actual experimental runtime differs")
        need(str(vllm.__version__).split("+")[0] == "0.31.0", "Wrong vLLM runtime")
        need(str(runtime.model_config.model) == config["model_path"], "Wrong model path")
        need(set(routes) == set(range(48)) and set(counters) == set(range(48)), "Actual48 native router contract incomplete")
        need(public_kernel.LOAD_TILE == 16, "Pristine lineage tile changed")
        emit("worker_identity", complete=True, source_lineage_commit=PUBLIC_COMMIT,
             public_source_sha256=PUBLIC, private_source_sha256=private_hashes(), runtime=actual,
             actual_runner=_CONTROL["runner_identity"], observer_revision=4,
             actual_compute_callable=callable_identity(FusedTopKRouter._compute_routing),
             actual_route_callable=callable_identity(BaseRouter._select_experts),
             actual_constructor_callable=callable_identity(MoERunner.__init__),
             expected_compute_module=("vllm.model_executor.layers.fused_moe.router.fused_topk_router" if p["mode"] == "stock"
                 else "placement_router.vendor.fused_topk_router" if p["mode"] == "par" else "stochastic_tail.plugin"),
             public_par_patched=bool(getattr(BaseRouter, "_placement_patched", False)),
             public_load_penalty=public_plugin._LOAD_PENALTY if p["mode"] == "par" else None,
             effective_load_tile=(16 if p["load_penalty"] else 1) if p["mode"] == "par" else None,
             experimental_overlay_active=True, record_calls=config["record_calls"],
             gpu_gate=config["gpu_gate"], native_layers=[routes[i] for i in range(48)],
             native_source_sha256=config["native_source_sha256"],
             checkpoint_proof=config["checkpoint_proof"], public_module_paths=actual_public,
             private_module_root=str(Path(__file__).resolve().parent),
             initial_rng_counters={str(i): 0 for i in sorted(counters)},
             counters_at_identity={str(i): r._stochastic_call_count for i, r in sorted(counters.items())},
             rng_contract="seed/physical-DP-rank/layer/per-router-call/local-row/expert; startup,dummy,scheduling consume calls; no request invariance",
             model_path=str(runtime.model_config.model), model_id=config["model_id"], revision=config["revision"],
             gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__), vllm_version=str(vllm.__version__))

    @wraps(route)
    def selected(self, hidden, logits, *args, **kwargs):
        nonlocal proof_written
        need(hasattr(self, "_stochastic_layer_index"), "Router missing native RNG layer mapping")
        with _CONTROL["route_lock"]:
            from .native_hooks import guard_route
            guard_route(_CONTROL,self,logits,config["record_calls"],_STEP,emit,need)
            call = self._stochastic_call_count
            need(call <= MASK, "RNG forward-call uint32 counter exhausted")
            self._stochastic_current_call = call
            self._stochastic_call_count += 1
            result = route(self, hidden, logits, *args, **kwargs)
            expected = ("vllm.model_executor.layers.fused_moe.router.fused_topk_router" if p["mode"] == "stock"
                        else "placement_router.vendor.fused_topk_router" if p["mode"] == "par" else "stochastic_tail.plugin")
            need(type(self)._compute_routing.__module__ == expected and self.top_k == 8
                 and self.global_num_experts == 128 and bool(self.renormalize)
                 and self.scoring_func == "softmax" and self.eplb_state is None, "Actual selector/router contract differs")
            if p["mode"] == "par":
                need(self.__dict__.get("_keep_for") == p["keep"] and public_plugin._LOAD_PENALTY == p["load_penalty"], "PAR control inactive/wrong price")
            if self._stochastic_layer_index not in routes:
                routes[self._stochastic_layer_index] = {"layer_name": self._stochastic_layer_name,
                    "layer_index": self._stochastic_layer_index, "router_id": id(self),
                    "top_k": self.top_k, "experts": self.global_num_experts, "renormalize": bool(self.renormalize),
                    "scoring_func": self.scoring_func, "eplb": self.eplb_state is not None,
                    "actual_public_keep": self.__dict__.get("_keep_for"),
                    "compute_callable": callable_identity(type(self)._compute_routing)}
            if config["record_calls"]:
                row = {"layer": self._stochastic_layer_index, "call": call, "local_T": int(logits.shape[0]), "experts": int(logits.shape[1]),
                       "logits_dtype": str(logits.dtype), "weight_dtype": str(result[0].dtype), "indices_dtype": str(result[1].dtype)}
                active = _STEP.get()
                if active is None:
                    emit("orphan_route_call", **row)
                else:
                    active["calls"].append(row)
            if not proof_written and len(routes) == 48:
                identity(hidden)
                proof_written = True
                print(f"[stochastic-tail] actual48router proof mode={p['mode']} keep={p['keep']} seed={p['seed']}", flush=True)
            return result

    _CONTROL = {"plan_sha256":plan_sha,"counters":counters,"routes":routes,"snapshot":snapshot,"emit":emit,
                "route_lock":threading.RLock(),"sealed":False,"seal_nonce":None,"checkpoint_nonces":set()}
    MoERunner.__init__, BaseRouter._select_experts = constructed, selected
    from .native_hooks import install
    install(config,_CONTROL,_STEP,emit,callable_identity,need)
    atexit.register(snapshot,"atexit")

    emit("entrypoint_installed", complete=True, actual_entrypoint="stochastic_tail_experiment=stochastic_tail.plugin:register",
         private_source_sha256=private_hashes(), pristine_public_files_are_lineage_only=True,
         record_calls=config["record_calls"], selection_mode=p["mode"])
