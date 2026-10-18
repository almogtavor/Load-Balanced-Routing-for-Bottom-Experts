#!/usr/bin/env python3
"""Real two-leg NIXL producer/consumer generation and prefix-scoring pilot.

One HTTP prompt per leg is intentional: a completion response carries one
top-level kv_transfer_params, so batching several prompts under it is unsafe.
The client never falls back to local prefill. Positive matched-token allocation,
actual receive completion and stock native decode are audited separately from
hybrid_phase_probe JSONL. These nonstreaming diagnostics do not measure TTFT.
"""
from __future__ import annotations
from native_accounting import zero_cached_transfer, zero_cached_usage
from native_request_id import SOURCES as REQUEST_SOURCES, external_item_id, producer_internal_id
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time
from common_v2 import (MODEL_ID, REVISION, SOURCE_COMMIT, base_body, choices, encoded,
                       file_sha, next_lp, post, read, require, sha, write_new)

REMOTE_DECODE = {"do_remote_decode": True, "do_remote_prefill": False,
                 "remote_engine_id": None, "remote_block_ids": None,
                 "remote_host": None, "remote_port": None}


def transfer_params(response, prompt_len):
    params = response.get("kv_transfer_params")
    require(isinstance(params, dict) and params.get("do_remote_prefill") is True
            and params.get("do_remote_decode") is False, "Producer did not return remote-prefill transfer parameters")
    require(all(params.get(key) is not None for key in ("remote_engine_id", "remote_request_id", "remote_host", "remote_port")),
            "Incomplete native transfer identity")
    blocks = params.get("remote_block_ids")
    require(isinstance(blocks, list) and blocks and all(isinstance(group, list) for group in blocks)
            and any(len(group) for group in blocks), "No actual producer KV blocks")
    require(type(params.get("remote_num_tokens")) is int and params["remote_num_tokens"] == prompt_len,
            "Producer KV does not cover the prompt")
    require(params.get("tp_size") == 1 and params.get("pp_size") == 1
            and params.get("transfer_mode") == "pull", "Unexpected transfer layout/mode")
    zero_cached_transfer(params, REMOTE_DECODE, params)
    return params


def paired_body(body, params):
    producer = dict(body)
    producer.update(max_tokens=1, stream=False, kv_transfer_params=dict(REMOTE_DECODE))
    # Producer output text is discarded; consumer regenerates the first output
    # from transferred prompt KV according to native connector's boundary rules.
    for key in ("logprobs", "logprob_token_ids", "prompt_logprobs", "min_tokens", "stop"):
        producer.pop(key, None)
    consumer = dict(body)
    if params is not None:
        consumer["kv_transfer_params"] = params
    return producer, consumer


def handoff(body, *, prefill_url, decode_url, stem, dp_rank, timeout, stop_unix):
    require(time.time() < stop_unix, "Global handoff deadline reached")
    headers = {"X-data-parallel-rank": str(dp_rank), "X-Request-Id": body["request_id"]}
    producer, _ = paired_body(body, None)
    start = time.perf_counter_ns()
    p_response, p_transport = post(prefill_url, producer, Path(str(stem) + "-prefill"),
                                    timeout=min(timeout, max(1, stop_unix - time.time())), headers=headers)
    require(p_response["id"]=="cmpl-"+body["request_id"], "Producer response does not bind exact application request ID")
    choices(p_response, [body["prompt"]])
    params = transfer_params(p_response, len(body["prompt"]))
    require(producer_internal_id(p_response) == params["remote_request_id"], "Native producer full internal identity differs")
    _, consumer = paired_body(body, params)
    require(consumer["prompt"] == producer["prompt"] and consumer["kv_transfer_params"] == params,
            "Handoff must preserve exact producer prompt and returned params")
    require(time.time() < stop_unix, "Deadline reached between producer and consumer; no fallback")
    d_response, d_transport = post(decode_url, consumer, Path(str(stem) + "-decode"),
                                    timeout=min(timeout, max(1, stop_unix - time.time())), headers=headers)
    require(d_response["id"]=="cmpl-"+body["request_id"], "Consumer response does not bind exact application request ID")
    result = choices(d_response, [body["prompt"]])[0]
    zero_cached_transfer(params, producer["kv_transfer_params"], consumer["kv_transfer_params"])
    zero_cached_usage(d_response["usage"], len(body["prompt"]), len(result["token_ids"]))
    require(len(result["token_ids"]) <= body["max_tokens"], "Consumer exceeded frozen cap")
    return result, {"request_id": body["request_id"], "dp_rank": dp_rank,
                    "prompt_sha256": sha(encoded(body["prompt"])), "prompt_tokens": len(body["prompt"]),
                    "transfer_params": params, "producer_transport": p_transport,
                    "producer_response_id": p_response["id"], "consumer_response_id": d_response["id"],
                    "producer_native_request_id": params["remote_request_id"],
                    "consumer_external_request_id": external_item_id(d_response["id"]),
                    "consumer_transport": d_transport, "elapsed_two_leg_ns": time.perf_counter_ns() - start,
                    "ttft_note": "Not measured: diagnostic response is nonstreaming"}


def reference_difference(identifier, handoff, monolithic):
    prefix=0
    for a,b in zip(handoff['token_ids'],monolithic['token_ids']):
        if a!=b:break
        prefix+=1
    equal=handoff['token_ids']==monolithic['token_ids']
    fields=('token_ids','text','finish_reason')
    return {'id':identifier,'equal_token_ids':equal,'equal_finish_reason':handoff['finish_reason']==monolithic['finish_reason'],
            'equal_text':handoff['text']==monolithic['text'],'common_generated_prefix_tokens':prefix,
            'first_differing_generated_position':None if equal else prefix,
            'monolithic_choice':{k:monolithic[k] for k in fields},'handoff_choice':{k:handoff[k] for k in fields},
            'interpretation':'Diagnostic serving-mode difference only; no monolithic equivalence or noise claim'}


def contract_pair(args):
    p, d = read(args.prefill_contract), read(args.decode_contract)
    for role, c in (("prefill", p), ("decode", d)):
        require(c.get("complete") is True and c["role"] == role and c["ep"] == 4
                and c["source_commit"] == SOURCE_COMMIT and c["checkpoint_revision"] == REVISION
                and c["kv_connector"] == "NixlConnector" and c["kv_load_failure_policy"] == "fail"
                and c["prefix_caching"] is False
                and c["request_id_sources"]["sha256"]==REQUEST_SOURCES
                and c["request_id_contract_sha256"]==file_sha(Path(__file__).with_name("native_request_id.py")), "Missing/mismatched ready hybrid contract")
    require(d["arm"] == "stock" and p["arm"] == args.arm, "Decode must remain stock; wrong prefill arm")
    require(p["checkpoint_proof_sha256"] == d["checkpoint_proof_sha256"]
            and p["source_proof_sha256"] == d["source_proof_sha256"], "P/D model/source differ")
    return p, d


def run(args):
    require(file_sha(args.prepared) == args.prepared_sha256, "Frozen input file changed")
    p, d = contract_pair(args)
    rows = read(args.prepared)
    if args.mode == "smoke":
        rows = [r for r in rows if r["task"] == "gsm8k" and r["split"] == "train"][:4]
        require(len(rows) == 4, "Smoke requires four fixed GSM8K training prompts")
    elif args.mode == "generate":
        rows = [r for r in rows if r["task"] == "gsm8k"]
        require(rows and all(r["split"] == args.expected_split for r in rows), "Unexpected generation split")
    else:
        rows = [r for r in rows if r["task"] == "hellaswag"]
        require(rows and all(r["split"] == args.expected_split for r in rows), "Unexpected scoring split")
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate input IDs")
    require(args.mode != "smoke" or args.concurrency == 4, "Stock reference and handoff smoke both require concurrency4")
    jobs = []
    for row in rows:
        if args.mode != "prefix-score":
            jobs.append({"row": row, "prompt": row["token_ids"], "target": None, "position": None})
        else:
            for j in range(row["continuation_start"], len(row["token_ids"])):
                jobs.append({"row": row, "prompt": row["token_ids"][:j], "target": row["token_ids"][j], "position": j})
    require(0 < len(jobs) <= args.max_requests and args.stop_unix > time.time(), "Missing/insufficient global bound")
    require(args.mode != "prefix-score" or args.concurrency == 1,
            "Causal prefix probe must remain serial to prevent cross-prefix future leakage")
    args.output.mkdir(parents=True, exist_ok=False)
    start = {"complete": False, "mode": args.mode, "arm": args.arm, "prefill_contract": p, "decode_contract": d,
             "launch_id": args.launch_id, "served_model": args.served_model, "expected_split": args.expected_split,
             "input_file_sha256": file_sha(args.prepared), "jobs": len(jobs), "concurrency": args.concurrency,
             "max_tokens": args.max_tokens, "stop_unix": args.stop_unix, "max_requests": args.max_requests,
             "helper_sha256": file_sha(__file__), "common_sha256": file_sha(Path(__file__).with_name("common_v2.py")),
             "phase_probe_sha256": file_sha(Path(__file__).with_name("hybrid_phase_probe.py")),
             "model_id": MODEL_ID, "revision": REVISION, "source_commit": SOURCE_COMMIT,
             "phase_probe_required_before_publication": True, "performance_claim": False}
    start["stock_reference_policy"] = "V6 diagnostic only; stock-handoff is baseline; failedV5 preserved"
    start["stock_monolithic_reference_requests"] = len(jobs) if args.mode == "smoke" and args.arm == "stock" else 0
    write_new(args.output / "start.json", start)
    reference = {}
    # A separate, explicit correctness reference before handoffs. This is not
    # fallback after a failed transfer, and reference request IDs never overlap
    # the handoff IDs. Reference timings are not pooled with hybrid timings.
    if args.mode == "smoke" and args.arm == "stock":
        def reference_job(pair):
            index, job = pair
            require(time.time() < args.stop_unix, "Deadline before native stock reference")
            body = base_body(args.served_model, job["prompt"], args.launch_id + f"-reference-{index:06d}", args.max_tokens)
            body["stop"] = ["Q:", "</s>", "<|im_end|>"]
            raw, transport = post(args.decode_url, body, args.output / "raw" / f"reference-{index:06d}",
                                   timeout=min(args.timeout, max(1, args.stop_unix - time.time())),
                                   headers={"X-data-parallel-rank": str(index % 4)})
            return job["row"]["id"], {"choice": choices(raw, [job["prompt"]])[0], "transport": transport,
                                      "request_id": raw["id"], "raw_stem": f"reference-{index:06d}"}
        with ThreadPoolExecutor(max_workers=4) as pool:
            reference = dict(pool.map(reference_job, enumerate(jobs)))
        write_new(args.output / "stock-monolithic-reference.json", reference)
    def job_run(pair):
        index, job = pair
        request_id = args.launch_id + f"-{index:06d}"
        body = base_body(args.served_model, job["prompt"], request_id,
                         max_tokens=1 if args.mode == "prefix-score" else args.max_tokens)
        if args.mode != "prefix-score":
            body["stop"] = ["Q:", "</s>", "<|im_end|>"]
        else:
            body.update(logprobs=0, logprob_token_ids=[job["target"]], return_tokens_as_token_ids=True, ignore_eos=True)
        choice, evidence = handoff(body, prefill_url=args.prefill_url, decode_url=args.decode_url,
                                   stem=args.output / "raw" / f"{index:06d}", dp_rank=index % 4,
                                   timeout=args.timeout, stop_unix=args.stop_unix)
        record = {"id": job["row"]["id"], "position": job["position"], "choice": choice, **evidence}
        if job["target"] is not None:
            record.update(target_id=job["target"], target_logprob=next_lp(choice, job["target"]))
        write_new(args.output / "records" / f"{index:06d}.json", record)
        return record
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            records = []
            # Bound outstanding work and stop before submitting a later cohort
            # if any current request fails. Preserve all in-flight successes.
            indexed = list(enumerate(jobs))
            for offset in range(0, len(indexed), args.concurrency):
                require(time.time() < args.stop_unix, "Global campaign deadline reached")
                records.extend(pool.map(job_run, indexed[offset:offset + args.concurrency]))
        fidelity = None
        if reference:
            fidelity = [reference_difference(r["id"], r["choice"], reference[r["id"]]["choice"]) for r in records]
            write_new(args.output / "stock-handoff-fidelity.json", fidelity)
            # V6 is prospectively compared with stock-handoff. Retain every
            # monolithic difference; systematic mode differences are not noise
            # and do not establish exact output equivalence.
            pass
        write_new(args.output / "result.json", {"complete": True, "mode": args.mode, "requests": len(records),
                  "start_sha256": file_sha(args.output / "start.json"), "arm": args.arm,
                  "record_files_sha256": {f"records/{i:06d}.json": file_sha(args.output / "records" / f"{i:06d}.json")
                                          for i in range(len(records))},
                  "requires_phase_audit": True, "no_performance_claim": True,
                  "stock_reference_files_sha256": {name: file_sha(args.output / name) for name in
                    ("stock-monolithic-reference.json", "stock-handoff-fidelity.json") if (args.output / name).exists()},
                  "raw_files_sha256": {str(path.relative_to(args.output)): file_sha(path)
                                       for path in sorted((args.output / "raw").glob("*.json"))},
                  "finished_unix_ns": time.time_ns()})
    except BaseException as error:
        write_new(args.output / "failure.json", {"complete": False, "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "generate", "prefix-score"), required=True)
    for name in ("prepared", "prefill-contract", "decode-contract", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared-sha256", "prefill-url", "decode-url", "arm", "launch-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-split", default="train")
    parser.add_argument("--served-model", default="task-model")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, choices=(1, 4, 16, 64), default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--stop-unix", type=float, required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    run(parser.parse_args())
