#!/usr/bin/env python3
"""Fixed-input DeepEP 2.0 legacy dispatch/combine mechanism replay (no expert GEMM).

  OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=4 \
      experiments/replay_deepep.py --output /results/deepep_replay.json

All ranks use the same source-token count in each case (default 1/512/16384).
Frozen seeded synthetic gate logits determine stock top8 versus balanced keep6,
penalty1 routes. Routing is precomputed and excluded from measurement. Actual
DeepEP legacy layout/dispatch, a weighted identity expert, and combine are timed
as one operation with CUDA events and CPU wall time. No per-phase synchronization
is inserted. Per-block rank-max times, per-rank spread, destination token/expert
loads, and ideal cross-rank activation payload bytes are retained. These bytes
are a model of BF16 activation copies, not measured NVLink/RDMA wire counters.
This isolates the communication mechanism; it is not a model-quality, full-MoE,
or end-to-end serving benchmark. APIs were checked against the installed DeepEP
2.0.0 buffers.legacy implementation used by vLLM 0.31 HT in this campaign.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import time


def stats(values):
    ordered = sorted(values)
    def q(p):
        pos = (len(ordered) - 1) * p
        lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {"count": len(values), "median": q(.5), "p95": q(.95), "min": ordered[0], "max": ordered[-1],
            "mean": statistics.mean(values)}


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 512, 16384])
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--buffer-mb", type=int, default=1024)
    parser.add_argument("--num-sms", type=int, choices=(20, 40, 64), default=20,
                        help="DeepEP communication SM budget; default20 matches the serving baseline")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    if min([args.experts, args.hidden, args.buffer_mb, args.warmup, args.blocks, args.iterations, *args.tokens]) < 1:
        parser.error("All numeric experiment settings must be positive")
    import torch
    import torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != 4 or args.experts % world or args.experts // world < 8:
        parser.error("This controlled replay requires world=4 and at least eight experts per rank")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180), device_id=device)
    cpu_group = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=180))
    # DeepEP checks the loaded NCCL runtime during import, so import it after
    # establishing the same NCCL process group used by the serving environment.
    import deep_ep
    from placement_router.kernel import rewrite_torch, gate_weights
    if deep_ep.__version__ != "2.0.0":
        raise RuntimeError(f"Expected verified DeepEP 2.0.0 legacy API, got {deep_ep.__version__}")
    if args.num_sms > torch.cuda.get_device_properties(device).multi_processor_count:
        raise ValueError("Requested DeepEP SM budget exceeds the GPU's SM count")
    deep_ep.Buffer.set_num_sms(args.num_sms)
    dispatch_config = deep_ep.Buffer.get_dispatch_config(world)
    combine_config = deep_ep.Buffer.get_combine_config(world)
    hidden_bytes = args.hidden * torch.tensor([], dtype=torch.bfloat16).element_size()
    dispatch_nvl_hint = int(dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, world))
    combine_nvl_hint = int(combine_config.get_nvl_buffer_size_hint(hidden_bytes, world))
    requested_nvl_bytes = args.buffer_mb * 1024 * 1024
    allocated_nvl_bytes = max(requested_nvl_bytes, dispatch_nvl_hint, combine_nvl_hint)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if allocated_nvl_bytes > free_bytes // 2:
        raise RuntimeError(f"Required NVLink communication buffer {allocated_nvl_bytes} bytes exceeds half available GPU memory {free_bytes}; refusing unsafe allocation")
    buffer = deep_ep.Buffer(group=cpu_group, num_nvl_bytes=allocated_nvl_bytes,
                            num_rdma_bytes=0, low_latency_mode=False, num_qps_per_rank=1,
                            explicitly_destroy=True)
    metadata = {"rank": rank, "local_rank": local_rank, "gpu": torch.cuda.get_device_name(device),
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
                "deep_ep": deep_ep.__version__, "torch": torch.__version__,
                "deep_ep_buffer_source_sha256": hashlib.sha256(inspect.getsource(deep_ep.Buffer).encode()).hexdigest()}
    all_metadata = [None] * world
    dist.all_gather_object(all_metadata, metadata, group=cpu_group)
    result = {"complete": False, "kind": "synthetic_fixed_input_communication_only",
              "configuration": vars(args) | {"output": str(args.output), "world": world, "k": 8,
                  "keep": 6, "load_penalty": 1, "dtype": "bfloat16", "gate_logits_dtype": "float32",
                  "num_sms": deep_ep.Buffer.num_sms,
                  "nvl_buffer_requested_bytes": requested_nvl_bytes,
                  "nvl_dispatch_size_hint_bytes": dispatch_nvl_hint,
                  "nvl_combine_size_hint_bytes": combine_nvl_hint,
                  "nvl_buffer_allocated_bytes": allocated_nvl_bytes,
                  "nvl_buffer_auto_expanded_to_hint": allocated_nvl_bytes > requested_nvl_bytes,
                  "expert_gemm": False,
                  "routing_precomputed_not_timed": True}, "rank_metadata": all_metadata,
              "limits": ["Synthetic frozen logits are reproducible and shape matched; they are not observed model router distributions.",
                         "Weighted identity replaces expert GEMM, so results cannot predict a full MoE layer or serving speedup.",
                         "Ideal activation bytes exclude indices, weights, protocol headers, padding, local copies and implementation overhead. No wire counters are measured.",
                         "CUDA and wall times include layout, uncached dispatch, identity weighting, combine and their waits/host launch overhead; there is no per-phase flush.",
                         "A CPU barrier aligns the start of each block and is outside timing; residual per-rank arrival skew remains part of the replay."],
              "cases": []}
    if rank == 0:
        save(args.output, result)

    def roundtrip(x, ids, weights):
        previous = buffer.capture()
        n_rank, n_rdma, n_expert, membership, _ = buffer.get_dispatch_layout(
            topk_idx=ids, num_experts=args.experts, previous_event=previous,
            async_finish=False, allocate_on_comm_stream=False)
        recv_x, recv_ids, recv_weights, recv_counts, handle, event = buffer.dispatch(
            x=x, handle=None, num_tokens_per_rank=n_rank, num_tokens_per_rdma_rank=n_rdma,
            is_token_in_rank=membership, num_tokens_per_expert=n_expert,
            topk_idx=ids, topk_weights=weights, expert_alignment=1, config=dispatch_config,
            previous_event=previous, async_finish=True, allocate_on_comm_stream=False)
        if event.event is not None:
            event.current_stream_wait()
        # Each token is received once per destination rank. Sum only expert
        # weights local to this rank; invalid slots are represented by -1.
        local_weight = torch.where(recv_ids >= 0, recv_weights, 0).sum(-1, keepdim=True)
        identity_out = (recv_x.float() * local_weight).to(torch.bfloat16)
        output, _, event = buffer.combine(x=identity_out, handle=handle,
            topk_weights=None, config=combine_config, previous_event=buffer.capture(),
            async_finish=True, allocate_on_comm_stream=False)
        if event.event is not None:
            event.current_stream_wait()
        return output, recv_counts, recv_x.shape[0]

    try:
        for tokens in args.tokens:
            generator = torch.Generator(device=device).manual_seed(args.seed + rank * 1009 + tokens)
            logits = torch.randn((tokens, args.experts), generator=generator, device=device, dtype=torch.float32)
            x = torch.randn((tokens, args.hidden), generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)
            stock = logits.topk(8, dim=-1).indices
            balanced = rewrite_torch(logits, stock, 6, world, load_penalty=1.0)
            routes = {name: (ids.to(deep_ep.topk_idx_t), gate_weights(logits, ids, renormalize=True).contiguous())
                      for name, ids in (("stock", stock), ("balanced6p1", balanced))}
            case = {"source_tokens_by_rank": [tokens] * world, "experts": args.experts,
                    "hidden": args.hidden, "arms": {}, "blocks": []}
            for arm, (ids, weights) in routes.items():
                destination = ids * world // args.experts
                membership = (destination.unsqueeze(-1) == torch.arange(world, device=device)).any(1)
                send_rows = membership.sum(0).cpu().tolist()
                expert_slots = torch.bincount(ids.flatten(), minlength=args.experts).cpu().tolist()
                local = {"rank": rank, "token_rank_copies_by_destination": send_rows,
                         "expert_slots_by_global_expert": expert_slots,
                         "selected_expert_slots": ids.numel(),
                         "expert_ids_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()}
                gathered = [None] * world
                dist.all_gather_object(gathered, local, group=cpu_group)
                recv_rows = [sum(s["token_rank_copies_by_destination"][d] for s in gathered) for d in range(world)]
                per_expert = [sum(s["expert_slots_by_global_expert"][e] for s in gathered) for e in range(args.experts)]
                slots_by_rank = [sum(per_expert[d * (args.experts // world):(d + 1) * (args.experts // world)]) for d in range(world)]
                cross = sum(sum(s["token_rank_copies_by_destination"]) - s["token_rank_copies_by_destination"][s["rank"]] for s in gathered)
                case["arms"][arm] = {"source_rank_routes": gathered, "destination_unique_token_rows": recv_rows,
                    "destination_expert_slots": slots_by_rank, "global_expert_slot_counts": per_expert,
                    "max_over_mean_destination_expert_slots": max(slots_by_rank) / statistics.mean(slots_by_rank),
                    "cross_rank_token_copies": cross,
                    "ideal_cross_rank_activation_bytes_one_direction": cross * args.hidden * 2,
                    "ideal_dispatch_plus_combine_activation_bytes": cross * args.hidden * 2 * 2}
                assert sum(per_expert) == tokens * world * 8
                dist.barrier(group=cpu_group)
                output, counts, received = roundtrip(x, ids, weights)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(output, x, atol=.04, rtol=.02)
                assert received == recv_rows[rank], (arm, rank, received, recv_rows)
                expected_counts = per_expert[rank * (args.experts // world):(rank + 1) * (args.experts // world)]
                assert list(counts) == expected_counts, (arm, rank, counts, expected_counts)
                local_check = {"rank": rank, "weighted_identity_max_abs_error": float((output.float() - x.float()).abs().max()),
                               "received_unique_token_rows": received, "expert_slot_counts_match": True}
                checks = [None] * world
                dist.all_gather_object(checks, local_check, group=cpu_group)
                case["arms"][arm]["correctness"] = checks
                for _ in range(args.warmup):
                    roundtrip(x, ids, weights)
                torch.cuda.synchronize(device)
            initial_order = list(routes)
            random.Random(args.seed + tokens).shuffle(initial_order)
            for block in range(args.blocks):
                order = initial_order.copy() if block % 2 == 0 else initial_order[::-1]
                block_result = {"block": block, "order": order, "arms": {}}
                for arm in order:
                    ids, weights = routes[arm]
                    samples = []
                    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                              for _ in range(args.iterations)]
                    # CUDA event handles initialize lazily at first record.
                    # Prime them before the block barrier/timer so allocation
                    # costs do not become communication or arrival-skew costs.
                    for start, finish in events:
                        start.record()
                        finish.record()
                    torch.cuda.synchronize(device)
                    dist.barrier(group=cpu_group)
                    block_began = time.perf_counter()
                    for start, finish in events:
                        began = time.perf_counter()
                        start.record()
                        roundtrip(x, ids, weights)
                        finish.record()
                        finish.synchronize()
                        samples.append({"wall_ms": (time.perf_counter() - began) * 1000,
                                        "cuda_ms": start.elapsed_time(finish)})
                    block_wall_ms = (time.perf_counter() - block_began) * 1000
                    block_cuda_ms = events[0][0].elapsed_time(events[-1][1])
                    gathered = [None] * world
                    dist.all_gather_object(gathered, {"rank": rank, "iterations": samples,
                        "block_wall_ms": block_wall_ms, "block_cuda_ms": block_cuda_ms}, group=cpu_group)
                    rank_max_wall = [max(g["iterations"][i]["wall_ms"] for g in gathered) for i in range(args.iterations)]
                    rank_max_cuda = [max(g["iterations"][i]["cuda_ms"] for g in gathered) for i in range(args.iterations)]
                    spread_wall = [max(g["iterations"][i]["wall_ms"] for g in gathered) - min(g["iterations"][i]["wall_ms"] for g in gathered) for i in range(args.iterations)]
                    block_result["arms"][arm] = {"per_rank": gathered,
                        "rankmax_block_wall_ms_per_iteration": max(g["block_wall_ms"] for g in gathered) / args.iterations,
                        "rankmax_block_cuda_ms_per_iteration": max(g["block_cuda_ms"] for g in gathered) / args.iterations,
                        "rank_max_wall_ms": rank_max_wall, "rank_max_cuda_ms": rank_max_cuda,
                        "rank_wall_spread_ms": spread_wall}
                case["blocks"].append(block_result)
                if rank == 0:
                    # Checkpoint each completed pair without losing earlier cases.
                    result["cases"] = [c for c in result["cases"] if c["source_tokens_by_rank"] != case["source_tokens_by_rank"]] + [case]
                    save(args.output, result)
                    print(f"T={tokens} block={block} " + " ".join(
                        f"{arm} rankmaxwall={block_result['arms'][arm]['rankmax_block_wall_ms_per_iteration']:.4f}ms"
                        for arm in order), flush=True)
            for arm in routes:
                walls = [v for b in case["blocks"] for v in b["arms"][arm]["rank_max_wall_ms"]]
                cuda = [v for b in case["blocks"] for v in b["arms"][arm]["rank_max_cuda_ms"]]
                block_walls = [b["arms"][arm]["rankmax_block_wall_ms_per_iteration"] for b in case["blocks"]]
                block_cuda = [b["arms"][arm]["rankmax_block_cuda_ms_per_iteration"] for b in case["blocks"]]
                case["arms"][arm]["rank_max_wall_ms_summary"] = stats(block_walls)
                case["arms"][arm]["rank_max_cuda_ms_summary"] = stats(block_cuda)
                case["arms"][arm]["pooled_iteration_rank_max_wall_ms_summary"] = stats(walls)
                case["arms"][arm]["pooled_iteration_rank_max_cuda_ms_summary"] = stats(cuda)
            before, after = case["arms"]["stock"], case["arms"]["balanced6p1"]
            paired_ratios = [b["arms"]["balanced6p1"]["rankmax_block_wall_ms_per_iteration"] /
                             b["arms"]["stock"]["rankmax_block_wall_ms_per_iteration"] for b in case["blocks"]]
            case["comparison"] = {"paired_block_wall_time_ratios_balanced_over_stock": paired_ratios,
                "paired_block_wall_ratio_summary": stats(paired_ratios),
                "repeat_unit": "Counterbalanced paired blocks; individual iterations within a block are not independent repeats.",
                "ideal_activation_payload_change_pct": (
                after["ideal_cross_rank_activation_bytes_one_direction"] / before["ideal_cross_rank_activation_bytes_one_direction"] - 1) * 100,
                "rankmax_wall_median_change_pct": (after["rank_max_wall_ms_summary"]["median"] / before["rank_max_wall_ms_summary"]["median"] - 1) * 100,
                "rankmax_cuda_median_change_pct": (after["rank_max_cuda_ms_summary"]["median"] / before["rank_max_cuda_ms_summary"]["median"] - 1) * 100}
            if rank == 0:
                save(args.output, result)
        result["complete"] = True
        if rank == 0:
            save(args.output, result)
    finally:
        torch.cuda.synchronize(device)
        buffer.destroy()
        dist.destroy_process_group(cpu_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
