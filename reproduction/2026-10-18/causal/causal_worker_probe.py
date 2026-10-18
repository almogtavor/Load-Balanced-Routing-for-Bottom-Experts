#!/usr/bin/env python3
"""Derived from frozen task_bench_probe.py for declared causal chunk sensitivity.

Loaded by PLACEMENT_PROBE. CAUSAL_PROBE_DIR, CAUSAL_PROBE_ARM,
CAUSAL_EXPECTED_BUDGET and CAUSAL_SOURCE_PROOF are required. The evidence is emitted
after 48 distinct native BaseRouter objects have actually routed. The wrapper
does not read tensor values, synchronize CUDA, modify choices or collect timings.
"""
from __future__ import annotations

from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time

COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
SOURCE = {
    "src/placement_router/kernel.py": "06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854",
    "src/placement_router/plugin.py": "c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8",
    "src/placement_router/vendor/fused_topk_router.py": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc"}
ARMS = {"stock": (None, None), "keep8": (8, 0.), "plain6": (6, 0.),
        "keep6p8": (6, 8.), "keep4p8": (4, 8.), "stock_repeat": (None, None)}
_INSTALLED = False


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def router_contract(router, keep):
    return {"class": type(router).__module__ + "." + type(router).__qualname__,
            "top_k": router.top_k, "experts": router.global_num_experts,
            "renormalize": bool(getattr(router, "renormalize", False)),
            "scoring_func": getattr(router, "scoring_func", None),
            "eplb": router.eplb_state is not None,
            "actual_keep": router.__dict__.get("_keep_for"),
            "routing_module": type(router)._compute_routing.__module__,
            "expected": router.top_k == 8 and router.global_num_experts == 128
                and bool(getattr(router, "renormalize", False))
                and getattr(router, "scoring_func", None) == "softmax"
                and router.eplb_state is None
                and router.__dict__.get("_keep_for") == keep
                and type(router)._compute_routing.__module__ == (
                    "placement_router.vendor.fused_topk_router" if keep is not None
                    else "vllm.model_executor.layers.fused_moe.router.fused_topk_router")}


def runtime_evidence(config, hidden, contracts, *, arm, ep, proof_path):
    import torch
    import vllm
    from vllm.distributed.parallel_state import get_ep_group
    from placement_router import kernel, plugin
    from placement_router.vendor import fused_topk_router
    group = get_ep_group()
    parallel = config.parallel_config
    keep, penalty = ARMS[arm]
    graph = getattr(config.compilation_config.cudagraph_mode, "name", str(config.compilation_config.cudagraph_mode))
    source_paths = {"src/placement_router/kernel.py": kernel.__file__,
                    "src/placement_router/plugin.py": plugin.__file__,
                    "src/placement_router/vendor/fused_topk_router.py": fused_topk_router.__file__}
    actual = {name: sha_file(path) for name, path in source_paths.items()}
    proof = json.loads(Path(proof_path).read_text())
    require(proof.get("source_commit", proof.get("base_commit")) == COMMIT, "Wrong deployment commit")
    require(actual == SOURCE, "Actual imported task routing source differs from frozen public tile16 commit")
    if "site_root" in proof:
        require(all(Path(path).resolve() == (Path(proof["site_root"]) / name.removeprefix("src/")).resolve()
                    for name, path in source_paths.items()), "Imported router modules are outside the deployed task-site")
    require(kernel.LOAD_TILE == 16 and plugin._KEEP == keep, "Wrong imported tile/keep")
    require((plugin._LOAD_PENALTY if keep is not None else None) == penalty, "Wrong imported load price")
    require(bool(getattr(__import__("vllm.model_executor.layers.fused_moe.router.base_router", fromlist=["BaseRouter"]).BaseRouter,
                         "_placement_patched", False)) == (keep is not None), "Stock/plugin activation mismatch")
    names = [str(x) for x in config.compilation_config.static_all_moe_layers]
    require(len(names) == 48 and len(contracts) == 48, "Require48 registered/actually routed layers")
    use_ubatching = bool(parallel.use_ubatching) or bool(parallel.enable_dbo) or int(parallel.ubatch_size) > 1
    expected_budget = int(os.environ["CAUSAL_EXPECTED_BUDGET"])
    require(expected_budget in (1024, 16384), "Chunk budget must be one of two preregistered values")
    value = {"complete": True, "pid": os.getpid(), "created_unix_ns": time.time_ns(), "arm": arm,
             "ep_rank": int(group.rank_in_group), "ep_world_size": int(group.world_size),
             "physical_dp": int(parallel.data_parallel_size), "tp": int(parallel.tensor_parallel_size),
             "pp": int(parallel.pipeline_parallel_size), "backend": parallel.all2all_backend,
             "enable_expert_parallel": bool(parallel.enable_expert_parallel),
             "expert_placement_strategy": parallel.expert_placement_strategy,
             "graphs": graph, "dtype": str(hidden.dtype), "hidden_size": int(hidden.shape[-1]),
             "prefix_caching": bool(config.cache_config.enable_prefix_caching), "ubatching": use_ubatching,
             "max_model_len": config.model_config.max_model_len,
             "max_num_seqs": config.scheduler_config.max_num_seqs,
             "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
             "logprobs_mode": config.model_config.logprobs_mode,
             "expected_budget": expected_budget,
             "model_path": str(config.model_config.model), "seed": config.model_config.seed,
             "placement_keep": keep, "load_penalty": penalty, "load_tile": kernel.LOAD_TILE,
             "registered_layers": len(names), "registered_layer_names": names,
             "observed_router_objects": len(contracts), "router_contracts": list(contracts.values()),
             "all_routers_expected": all(c["expected"] for c in contracts.values()),
             "source_sha256": actual, "source_paths": source_paths, "source_commit": COMMIT,
             "deployment_proof_sha256": sha_file(proof_path), "probe_sha256": sha_file(__file__),
             "device_index": torch.cuda.current_device(), "gpu_name": torch.cuda.get_device_name(),
             "torch_version": str(torch.__version__), "vllm_version": str(vllm.__version__),
             "env_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "method": "post-native-route first48distinctBaseRouterobjects; no tensor values/timing/synchronization"}
    require(value["ep_world_size"] == ep and value["physical_dp"] == ep and value["tp"] == 1 and value["pp"] == 1,
            "Task protocol requires physicalDP=EP, TP1/PP1")
    require(value["enable_expert_parallel"] and value["expert_placement_strategy"] == "linear",
            "Expert mapping must be contiguous linear physicalEP")
    require(value["backend"] == "deepep_high_throughput" and graph == "NONE" and not use_ubatching
            and value["dtype"] == "torch.bfloat16" and value["hidden_size"] == 2048 and not value["prefix_caching"]
            and value["all_routers_expected"] and value["max_model_len"] == 8192 and value["max_num_seqs"] == 256
            and value["max_num_batched_tokens"] == expected_budget and value["seed"] == 0
            and value["logprobs_mode"] == "raw_logprobs" and value["vllm_version"].split("+")[0] == "0.31.0",
            "Actual task server differs from frozen protocol")
    return value


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    output = Path(os.environ["CAUSAL_PROBE_DIR"])
    arm, ep = os.environ["CAUSAL_PROBE_ARM"], 4
    proof_path = Path(os.environ["CAUSAL_SOURCE_PROOF"])
    require(arm in ARMS, "Unknown arm")
    config_holder, contracts = {}, {}
    done = False
    constructor, route = MoERunner.__init__, BaseRouter._select_experts
    require(route.__module__.startswith("vllm."), "Unexpected existing task routing instrumentation")

    @wraps(constructor)
    def initialized(self, *args, **kwargs):
        result = constructor(self, *args, **kwargs)
        config = get_current_vllm_config()
        require(config is not None, "Native MoERunner construction has no config")
        config_holder["config"] = config
        return result

    @wraps(route)
    def routed(self, hidden_states, router_logits, *args, **kwargs):
        nonlocal done
        result = route(self, hidden_states, router_logits, *args, **kwargs)
        if not done and id(self) not in contracts:
            contracts[id(self)] = router_contract(self, ARMS[arm][0])
            if len(contracts) == 48:
                evidence = runtime_evidence(config_holder["config"], hidden_states, contracts,
                                            arm=arm, ep=ep, proof_path=proof_path)
                output.mkdir(parents=True, exist_ok=True)
                path = output / f"rank{evidence['ep_rank']}_pid{os.getpid()}.json"
                require(not path.exists(), "Worker evidence already exists")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
                os.replace(temporary, path)
                done = True
                print(f"[task-bench] ready arm={arm} ep_rank={evidence['ep_rank']} ep={ep} "
                      f"keep={evidence['placement_keep']} penalty={evidence['load_penalty']} tile=16 routers=48", flush=True)
        return result

    MoERunner.__init__, BaseRouter._select_experts = initialized, routed
    _INSTALLED = True
    print(f"[task-bench] identity probe installed arm={arm} physicalEP={ep}; no timers", flush=True)
