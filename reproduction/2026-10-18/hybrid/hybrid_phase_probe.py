"""Correctness-only native scheduler/worker/routing evidence for real NIXL PD.

Load through PLACEMENT_PROBE in both roles. It does not change selections,
scheduler outputs or transfer state. CPU JSONL writes instrument these launches;
never treat their timings as uninstrumented performance. No tensor readback,
CUDA synchronization, or new collective. Physical dummy ranks are distinguished
from workers carrying real scheduled requests by their scheduler request map.
"""
from __future__ import annotations
from collections import Counter
from functools import wraps
import hashlib
import inspect
import importlib.util
import json
import os
from pathlib import Path
import time
import sys

_runner_spec = importlib.util.spec_from_file_location("hybrid_v2_native_runner_contract",
                                                    Path(__file__).with_name("native_runner_contract.py"))
_runner = importlib.util.module_from_spec(_runner_spec)
_previous_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _runner_spec.loader.exec_module(_runner)
finally:
    sys.dont_write_bytecode = _previous_bytecode
actual_runner, execute_flags, installed_sources = _runner.actual_runner, _runner.execute_flags, _runner.installed_sources
_request_spec = importlib.util.spec_from_file_location("hybrid_v4_native_request_id", Path(__file__).with_name("native_request_id.py"))
request_ids = importlib.util.module_from_spec(_request_spec)
_previous_bytecode = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _request_spec.loader.exec_module(request_ids)
finally:
    sys.dont_write_bytecode = _previous_bytecode

COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
SOURCE = {
    "kernel": "06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854",
    "plugin": "c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8",
    "vendor": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc"}
_INSTALLED = False


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_prompt(row):
    prompt = row.prompt_token_ids
    require(isinstance(prompt, list) and prompt and all(type(t) is int and t >= 0 for t in prompt),
            "Phase proof requires actual native integer prompt IDs")
    prefill = row.prefill_token_ids
    require(prefill == prompt, "Fresh V2 request all_token_ids must be the exact prompt; resumed requests are outside this proof")
    prompt_hash = hashlib.sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":")).encode()).hexdigest()
    return {"prompt_len": len(prompt), "initial_computed_tokens": row.num_computed_tokens,
            "prompt_sha256": prompt_hash, "last_prompt_token_id": prompt[-1],
            "native_all_token_ids_sha256": prompt_hash, "native_all_token_ids_len": len(prefill),
            "native_last_prefill_token_id": prefill[-1]}


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    role, arm = os.environ["HYBRID_ROLE"], os.environ["HYBRID_ARM"]
    require("VLLM_USE_V2_MODEL_RUNNER" not in os.environ, "Observe native default runner; no explicit override")
    keep = {"stock": None, "keep8": 8, "plain6": 6, "keep6p8": 6, "keep4p8": 4}[arm]
    require(role in ("prefill", "decode") and (role != "decode" or arm == "stock"), "Decode must use stock")
    out = Path(os.environ["HYBRID_PROBE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    config_holder, active, identified, known_requests = {}, {}, set(), {}
    contracts = {}
    probe_sha256 = digest(__file__)
    runner_contract_sha256 = digest(Path(__file__).with_name("native_runner_contract.py"))
    request_id_contract_sha256 = digest(Path(__file__).with_name("native_request_id.py"))

    def emit(kind, **fields):
        value = {"kind": kind, "role": role, "arm": arm, "pid": os.getpid(),
                 "unix_ns": time.time_ns(), "probe_sha256": probe_sha256,
                 "runner_contract_sha256": runner_contract_sha256,
                 "request_id_contract_sha256": request_id_contract_sha256, **fields}
        data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(out / f"pid{os.getpid()}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            require(os.write(fd, data) == len(data), "Incomplete evidence write")
        finally:
            os.close(fd)

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    import vllm
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import NixlPullConnectorScheduler
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
    constructor, route, execute = MoERunner.__init__, BaseRouter._select_experts, GPUModelRunner.execute_model
    native_sources = installed_sources(Path(vllm.__file__).resolve().parent)
    request_id_sources = request_ids.installed_sources(Path(vllm.__file__).resolve().parent)
    from vllm.v1.engine.input_processor import InputProcessor
    from vllm import envs
    request_ids.install_assignment_hook(InputProcessor, envs, request_id_sources, emit)
    execute_signature = inspect.signature(execute)
    runner_proof = {}

    @wraps(constructor)
    def constructed(self, *args, **kwargs):
        result = constructor(self, *args, **kwargs)
        config_holder["config"] = get_current_vllm_config()
        return result

    @wraps(route)
    def routed(self, hidden_states, router_logits, *args, **kwargs):
        result = route(self, hidden_states, router_logits, *args, **kwargs)
        config = config_holder.get("config")
        if config is not None:
            if id(self) not in contracts:
                contract = {"router_id": id(self), "routing_module": type(self)._compute_routing.__module__,
                            "keep": self.__dict__.get("_keep_for"), "top_k": self.top_k,
                            "experts": self.global_num_experts, "renormalize": bool(getattr(self, "renormalize", False)),
                            "scoring_func": getattr(self, "scoring_func", None), "eplb": self.eplb_state is not None}
                expected_module = ("placement_router.vendor.fused_topk_router" if keep is not None else
                                   "vllm.model_executor.layers.fused_moe.router.fused_topk_router")
                require(contract["routing_module"] == expected_module and contract["keep"] == keep
                        and contract["top_k"] == 8 and contract["experts"] == 128 and contract["renormalize"]
                        and contract["scoring_func"] == "softmax" and not contract["eplb"], "Unexpected native router")
                contracts[id(self)] = contract
            if active:
                active["router_calls"] += 1
                active["router_ids"].add(id(self))
                active["local_token_shapes"][int(hidden_states.shape[0])] += 1
            else:
                emit("route_outside_execute", router_id=id(self), local_tokens=int(hidden_states.shape[0]))
            if len(contracts) == 48 and "worker" not in identified:
                import torch
                import vllm
                from vllm.distributed.parallel_state import get_ep_group
                from placement_router import kernel, plugin
                from placement_router.vendor import fused_topk_router
                source = {"kernel": digest(kernel.__file__), "plugin": digest(plugin.__file__),
                          "vendor": digest(fused_topk_router.__file__)}
                require(source == SOURCE and kernel.LOAD_TILE == 16 and plugin._KEEP == keep,
                        "Source/tile/plugin identity differs")
                require(plugin._LOAD_PENALTY == (8.0 if arm.endswith("p8") else 0.0), "Wrong penalty")
                p, k = config.parallel_config, config.kv_transfer_config
                require(p.data_parallel_size == 4 and p.tensor_parallel_size == p.pipeline_parallel_size == 1
                        and get_ep_group().world_size == 4 and p.enable_expert_parallel
                        and p.expert_placement_strategy == "linear" and p.all2all_backend == "deepep_high_throughput",
                        "Hybrid requires physical DP=EP4 TP1PP1 DeepEPHT linear")
                require(k.kv_connector == "NixlConnector" and k.kv_load_failure_policy == "fail"
                        and k.kv_role == ("kv_producer" if role == "prefill" else "kv_consumer"), "Wrong transfer contract")
                graph = getattr(config.compilation_config.cudagraph_mode, "name", str(config.compilation_config.cudagraph_mode))
                require(graph == "NONE" and config.model_config.seed == 0 and str(hidden_states.dtype) == "torch.bfloat16"
                        and not config.cache_config.enable_prefix_caching and config.model_config.max_model_len == 8192
                        and config.scheduler_config.max_num_seqs == 256 and config.scheduler_config.max_num_batched_tokens == 16384
                        and config.model_config.logprobs_mode == "raw_logprobs" and vllm.__version__.split("+")[0] == "0.31.0",
                        "Unexpected hybrid graph/dtype/model/scheduler/logprob contract")
                emit("worker_identity", ep_rank=int(get_ep_group().rank_in_group), ep_size=4,
                     device=torch.cuda.current_device(), gpu=torch.cuda.get_device_name(), source_sha256=source,
                     source_commit=COMMIT, vllm_version=vllm.__version__, model_path=str(config.model_config.model),
                     graphs=graph, seed=config.model_config.seed, max_model_len=config.model_config.max_model_len,
                     logprobs_mode=config.model_config.logprobs_mode,
                     dtype=str(hidden_states.dtype), contracts=list(contracts.values()), kv_role=k.kv_role,
                     kv_connector=k.kv_connector, kv_load_failure_policy=k.kv_load_failure_policy,
                     max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
                     max_num_seqs=config.scheduler_config.max_num_seqs,
                     prefix_caching=config.cache_config.enable_prefix_caching,
                     cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                     native_runner=runner_proof or None,
                     runner_contract_sha256=digest(Path(__file__).with_name("native_runner_contract.py")))
                identified.add("worker")
        return result

    @wraps(execute)
    def executed(self, scheduler_output, *args, **kwargs):
        require(not active, "Nested GPUModelRunner execution not supported in correctness probe")
        if not runner_proof:
            proof = actual_runner(self, GPUModelRunner, execute, native_sources)
            runner_proof.update(proof)
            from vllm.distributed.parallel_state import get_ep_group
            emit("runner_identity", ep_rank=int(get_ep_group().rank_in_group), native_runner=proof,
                 runner_contract_sha256=digest(Path(__file__).with_name("native_runner_contract.py")))
        require(type(self) is GPUModelRunner and self.vllm_config.use_v2_model_runner is True
                and GPUModelRunner.execute_model is executed,
                "Actual runner class/config/callable changed after first native execution")
        flags = execute_flags(execute_signature, self, scheduler_output, args, kwargs)
        scheduled = dict(scheduler_output.num_scheduled_tokens)
        if not flags["dummy_run"]:
            for row in scheduler_output.scheduled_new_reqs:
                known_requests[row.req_id] = native_prompt(row)
        active.update(router_calls=0, router_ids=set(), local_token_shapes=Counter())
        try:
            return execute(self, scheduler_output, *args, **kwargs)
        finally:
            from vllm.distributed.parallel_state import get_ep_group
            emit("worker_step", ep_rank=int(get_ep_group().rank_in_group), scheduled_tokens=scheduled,
                 request_contracts={key: known_requests.get(key) for key in scheduled},
                 router_calls=active["router_calls"], distinct_router_objects=len(active["router_ids"]),
                 local_token_shapes=dict(active["local_token_shapes"]),
                 dummy_run=flags["dummy_run"], is_profile=flags["is_profile"],
                 has_real_scheduled_requests=bool(scheduled) and not flags["dummy_run"],
                 total_scheduled_tokens=int(scheduler_output.total_num_scheduled_tokens))
            active.clear()
            if not flags["dummy_run"]:
                for key in scheduler_output.finished_req_ids:
                    known_requests.pop(key, None)

    matched = NixlPullConnectorScheduler.get_num_new_matched_tokens
    allocated = NixlPullConnectorScheduler.update_state_after_alloc
    finished = NixlPullConnectorScheduler.request_finished
    received = NixlBaseConnectorWorker.get_transfer_results
    stats = NixlBaseConnectorWorker.get_kv_connector_stats
    errors = NixlBaseConnectorWorker.get_block_ids_with_load_errors

    @wraps(matched)
    def got_matched(self, request, num_computed_tokens):
        params = dict(request.kv_transfer_params or {})
        result = matched(self, request, num_computed_tokens)
        emit("scheduler_matched", request_id=request.request_id, params=params, prompt_len=request.num_prompt_tokens,
             prior_computed_tokens=num_computed_tokens, external_tokens=int(result[0]), asynchronous=bool(result[1]),
             dp_rank=int(self.vllm_config.parallel_config.data_parallel_index))
        return result

    @wraps(allocated)
    def got_allocated(self, request, blocks, num_external_tokens):
        params = dict(request.kv_transfer_params or {})
        result = allocated(self, request, blocks, num_external_tokens)
        emit("scheduler_allocated", request_id=request.request_id, params=params,
             external_tokens=int(num_external_tokens), prompt_len=request.num_prompt_tokens,
             dp_rank=int(self.vllm_config.parallel_config.data_parallel_index))
        return result

    @wraps(finished)
    def got_finished(self, request, block_ids):
        result = finished(self, request, block_ids)
        emit("scheduler_finished", request_id=request.request_id, delay_free=bool(result[0]),
             emitted_transfer_params=result[1], request_status=str(request.status),
             dp_rank=int(self.vllm_config.parallel_config.data_parallel_index))
        return result

    @wraps(received)
    def got_received(self):
        result = received(self)
        if result.finished_sending or result.finished_recving or result.failed_recving:
            from vllm.distributed.parallel_state import get_ep_group
            emit("worker_transfer_finished", ep_rank=int(get_ep_group().rank_in_group),
                 done_sending=sorted(result.finished_sending), done_receiving=sorted(result.finished_recving),
                 failed_receiving=sorted(result.failed_recving))
        return result

    @wraps(stats)
    def got_stats(self):
        result = stats(self)
        if result is not None and not result.is_empty():
            from vllm.distributed.parallel_state import get_ep_group
            emit("worker_transfer_stats", ep_rank=int(get_ep_group().rank_in_group), data=result.data)
        return result

    @wraps(errors)
    def got_errors(self):
        result = errors(self)
        if result:
            emit("worker_transfer_errors", invalid_block_ids=sorted(result))
        return result

    MoERunner.__init__, BaseRouter._select_experts, GPUModelRunner.execute_model = constructed, routed, executed
    NixlPullConnectorScheduler.get_num_new_matched_tokens = got_matched
    NixlPullConnectorScheduler.update_state_after_alloc = got_allocated
    NixlPullConnectorScheduler.request_finished = got_finished
    NixlBaseConnectorWorker.get_transfer_results = got_received
    NixlBaseConnectorWorker.get_kv_connector_stats = got_stats
    NixlBaseConnectorWorker.get_block_ids_with_load_errors = got_errors
    emit("installed", request_id_sources=request_id_sources, instrumentation="CPU correctness evidence; exclude from performance estimates")
