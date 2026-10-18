#!/usr/bin/env python3
"""Audit a CLOSED instrumented two-leg campaign, including real transfer use."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from common_v2 import SOURCE_COMMIT, SOURCE, base_body, choices, encoded, file_sha, read, require, sha, write_new
from hybrid_client import paired_body, transfer_params


def load_events(folder):
    import json
    files = sorted(folder.glob("pid*.jsonl"))
    require(files, "Missing native phase evidence")
    rows, hashes = [], {}
    for path in files:
        before = file_sha(path)
        content = path.read_bytes()
        require(content.endswith(b"\n"), "Partial phase-event file")
        parsed = [json.loads(line) for line in content.splitlines()]
        require(all(r["pid"] == int(path.stem.removeprefix("pid")) for r in parsed), "Phase file/PID identity differs")
        rows.extend(parsed)
        require(file_sha(path) == before, "Phase evidence still being written; audit after owned server closure")
        hashes[str(path)] = before
    return rows, hashes


def audit(args):
    result, start = read(args.result / "result.json"), read(args.result / "start.json")
    require(result.get("complete") is True and result["start_sha256"] == file_sha(args.result / "start.json"), "Incomplete/changed run")
    require(file_sha(args.prepared) == start["input_file_sha256"], "Prepared IDs changed")
    require(start["helper_sha256"] == file_sha(Path(__file__).with_name("hybrid_client.py"))
            and start["common_sha256"] == file_sha(Path(__file__).with_name("common_v2.py"))
            and start["phase_probe_sha256"] == file_sha(Path(__file__).with_name("hybrid_phase_probe.py")),
            "Frozen execution helpers differ")
    raw_paths = sorted((args.result / "raw").glob("*.json"))
    require(set(result["raw_files_sha256"]) == {str(p.relative_to(args.result)) for p in raw_paths}, "Raw HTTP inventory differs")
    for path in raw_paths:
        require(file_sha(path) == result["raw_files_sha256"][str(path.relative_to(args.result))], "Raw HTTP bytes changed")
    paths = sorted((args.result / "records").glob("*.json"))
    require(len(paths) == result["requests"] == start["jobs"], "Missing or excess request records")
    require(set(result["record_files_sha256"]) == {"records/" + p.name for p in paths}, "Record inventory differs")
    records = []
    for path in paths:
        require(file_sha(path) == result["record_files_sha256"]["records/" + path.name], "Record bytes changed")
        records.append(read(path))
    inputs = read(args.prepared)
    if start["mode"] == "smoke":
        inputs = [r for r in inputs if r["task"] == "gsm8k" and r["split"] == "train"][:4]
    else:
        task = "hellaswag" if start["mode"] == "prefix-score" else "gsm8k"
        inputs = [r for r in inputs if r["task"] == task]
        require(all(r["split"] == start["expected_split"] for r in inputs), "Source split differs")
    jobs = []
    for row in inputs:
        if start["mode"] == "prefix-score":
            jobs.extend((row, j, row["token_ids"][:j]) for j in range(row["continuation_start"], len(row["token_ids"])))
        else:
            jobs.append((row, None, row["token_ids"]))
    require(len(jobs) == len(records) and len({r["id"] for r in inputs}) == len(inputs), "Frozen job inventory differs")
    p_events, p_hashes = load_events(args.prefill_probe)
    d_events, d_hashes = load_events(args.decode_probe)
    require(all(r["probe_sha256"] == start["phase_probe_sha256"] for r in p_events + d_events), "Phase probe identity differs")
    require(all(r["role"] == "prefill" and r["arm"] == start["arm"] for r in p_events), "Prefill role/arm differs")
    require(all(r["role"] == "decode" and r["arm"] == "stock" for r in d_events), "Decode must remain stock")
    worker_proofs = {}
    for role, events in (("prefill", p_events), ("decode", d_events)):
        identities = [r for r in events if r["kind"] == "worker_identity"]
        require(len(identities) == 4 and {r["ep_rank"] for r in identities} == set(range(4))
                and len({r["pid"] for r in identities}) == 4, "Missing/duplicate physical EP workers")
        for worker in identities:
            expected_source = dict(zip(("kernel", "plugin", "vendor"), SOURCE.values()))
            require(worker["source_commit"] == SOURCE_COMMIT and len(worker["contracts"]) == 48
                    and worker["ep_size"] == 4 and worker["kv_connector"] == "NixlConnector"
                    and worker["kv_load_failure_policy"] == "fail" and worker["prefix_caching"] is False
                    and worker["source_sha256"] == expected_source,
                    "Incomplete/wrong runtime router/transfer identity")
            contract = start[role + "_contract"]
            expected_keep = {"stock": None, "keep8": 8, "plain6": 6, "keep6p8": 6, "keep4p8": 4}[worker["arm"]]
            expected_module = ("placement_router.vendor.fused_topk_router" if expected_keep is not None else
                               "vllm.model_executor.layers.fused_moe.router.fused_topk_router")
            require(len({r["router_id"] for r in worker["contracts"]}) == 48 and all(
                r["keep"] == expected_keep and r["routing_module"] == expected_module and r["top_k"] == 8
                and r["experts"] == 128 and r["renormalize"] is True and r["scoring_func"] == "softmax" and r["eplb"] is False
                for r in worker["contracts"]), "Native routing contract differs")
            require(worker["graphs"] == "NONE" and worker["seed"] == 0 and worker["dtype"] == "torch.bfloat16"
                    and worker["vllm_version"].split("+")[0] == "0.31.0" and worker["model_path"] == contract["model_path"]
                    and worker["kv_role"] == ("kv_producer" if role == "prefill" else "kv_consumer")
                    and worker["max_model_len"] == 8192 and worker["max_num_seqs"] == 256
                    and worker["max_num_batched_tokens"] == 16384 and worker["logprobs_mode"] == "raw_logprobs",
                    "Actual worker configuration differs")
        worker_proofs[role] = [{"rank": r["ep_rank"], "pid": r["pid"], "source_sha256": r["source_sha256"]} for r in identities]
    require(not any(r["kind"] == "worker_transfer_errors" for r in p_events + d_events), "KV load error evidence present")
    require(not any(r.get("failed_receiving") for r in p_events + d_events), "Native failed-receive evidence present")
    all_stats = [r for r in p_events + d_events if r["kind"] == "worker_transfer_stats"]
    for row in all_stats:
        require(all(not values for key, values in row["data"].items()
                    if key in ("num_failed_transfers", "num_failed_notifications", "num_failed_handshakes", "num_kv_expired_reqs")),
                "NIXL transfer failure/expiry evidence present")
    transferred_bytes = sum(sum(r["data"].get("bytes_transferred", [])) for r in all_stats)
    require(transferred_bytes > 0, "No actual NIXL transfer telemetry bytes; params alone are not proof")
    received = set(x for r in d_events if r["kind"] == "worker_transfer_finished" for x in r["done_receiving"])
    allocations = defaultdict(list)
    steps = defaultdict(list)
    for row in d_events:
        if row["kind"] == "scheduler_allocated":
            allocations[row["request_id"]].append(row)
        elif row["kind"] == "worker_step":
            for request_id, n in row["scheduled_tokens"].items():
                steps[request_id].append({"tokens": n, "rank": row["ep_rank"],
                                          "contract": row["request_contracts"].get(request_id),
                                          "router_objects": row["distinct_router_objects"]})
    seen = set()
    for i, record in enumerate(records):
        row, position, prompt = jobs[i]
        require(record["id"] == row["id"] and record["position"] == position and record["dp_rank"] == i % 4
                and record["request_id"] == start["launch_id"] + f"-{i:06d}", "Request/frozen source identity differs")
        request_id = record["consumer_native_request_id"]
        require(request_id not in seen and request_id in received, "Missing/duplicate actual receive completion")
        seen.add(request_id)
        allocated = allocations[request_id]
        require(any(r["external_tokens"] >= record["prompt_tokens"] - 1 and r["external_tokens"] > 0
                    and r["params"].get("remote_request_id") == record["producer_native_request_id"] for r in allocated),
                "No matched external KV allocation bound to this producer request")
        request_steps = steps[request_id]
        require(request_steps and all(r["tokens"] == 1 and r["rank"] == record["dp_rank"]
                                      and r["router_objects"] == 48 for r in request_steps),
                "Decode recomputed multi-token prompt, used wrong rank, or missed routing phase coverage")
        initial = next((r["contract"] for r in request_steps if r["contract"] is not None), None)
        require(initial and initial["prompt_len"] == record["prompt_tokens"]
                and initial["initial_computed_tokens"] >= record["prompt_tokens"] - 1,
                "Decoder initial schedule lacks transferred-prefix coverage")
        # Re-read actual HTTP bodies: no invented params or changed prompt.
        prefix = args.result / "raw" / f"{i:06d}"
        p_envelope = read(Path(str(prefix) + "-prefill.request.json"))
        d_envelope = read(Path(str(prefix) + "-decode.request.json"))
        p_request, d_request = p_envelope["body"], d_envelope["body"]
        body = base_body(start["served_model"], prompt, record["request_id"],
                         max_tokens=1 if start["mode"] == "prefix-score" else start["max_tokens"])
        if start["mode"] == "prefix-score":
            body.update(logprobs=0, logprob_token_ids=[row["token_ids"][position]], return_tokens_as_token_ids=True, ignore_eos=True)
        else:
            body["stop"] = ["Q:", "</s>", "<|im_end|>"]
        expected_p, expected_d = paired_body(body, record["transfer_params"])
        require(p_request == expected_p and d_request == expected_d, "HTTP bodies differ from frozen generation/prefix protocol")
        expected_headers = {"X-data-parallel-rank": str(i % 4), "X-Request-Id": record["request_id"]}
        require(p_envelope["sha256"] == sha(encoded(p_request)) and d_envelope["sha256"] == sha(encoded(d_request))
                and p_envelope["headers"] == d_envelope["headers"] == expected_headers, "HTTP request hash/rank headers differ")
        for role_name, transport in (("prefill", record["producer_transport"]), ("decode", record["consumer_transport"])):
            raw_path = Path(str(prefix) + "-" + role_name + ".response.json")
            require(file_sha(raw_path) == transport["response_sha256"] and transport["http_status"] == 200
                    and transport["response_complete"] is True, "Incomplete/changed raw HTTP response")
            require(read(Path(str(prefix) + "-" + role_name + ".transport.json")) == transport, "Transport receipt differs")
            raw = read(raw_path)
            choices(raw, [prompt])
            if role_name == "decode":
                require(len(raw["choices"]) == 1 and raw["choices"][0] == record["choice"]
                        and raw["id"] == record["consumer_response_id"]
                        and raw["id"] + "-0" == record["consumer_native_request_id"], "Recorded output differs from preserved raw response")
            else:
                require(transfer_params(raw, len(prompt)) == record["transfer_params"]
                        and raw["id"] == record["producer_response_id"]
                        and raw["id"] + "-0" == record["producer_native_request_id"], "Actual producer transfer identity differs")
        require(p_request["prompt"] == d_request["prompt"]
                and sha(encoded(d_request["prompt"])) == record["prompt_sha256"]
                and d_request["kv_transfer_params"] == record["transfer_params"]
                and d_request["max_tokens"] == (1 if start["mode"] == "prefix-score" else start["max_tokens"]),
                "HTTP handoff input/params/cap differ")
        require("prompt_logprobs" not in d_request, "Do not recompute full prompt LP on decoder")
    reference_summary = None
    if start["mode"] == "smoke" and start["arm"] == "stock":
        require(set(result["stock_reference_files_sha256"]) == {"stock-monolithic-reference.json", "stock-handoff-fidelity.json"},
                "Stock handoff pilot requires native monolithic reference")
        for name, expected in result["stock_reference_files_sha256"].items():
            require(file_sha(args.result / name) == expected, "Stock reference file changed")
        references = read(args.result / "stock-monolithic-reference.json")
        require(set(references) == {r["id"] for r in records}, "Native reference item set differs")
        fidelity = []
        for i, record in enumerate(records):
            reference = references[record["id"]]
            stem = args.result / "raw" / f"reference-{i:06d}"
            envelope = read(stem.with_suffix(".request.json"))
            body = base_body(start["served_model"], jobs[i][2], start["launch_id"] + f"-reference-{i:06d}", start["max_tokens"])
            body["stop"] = ["Q:", "</s>", "<|im_end|>"]
            require(envelope["body"] == body and envelope["sha256"] == sha(encoded(body))
                    and envelope["headers"] == {"X-data-parallel-rank": str(i % 4)} and "kv_transfer_params" not in body,
                    "Native stock reference request differs or imports KV")
            raw, transport = read(stem.with_suffix(".response.json")), reference["transport"]
            choice = choices(raw, [body["prompt"]])[0]
            require(choice == reference["choice"] and transport["response_complete"] is True and transport["http_status"] == 200
                    and transport["response_sha256"] == file_sha(stem.with_suffix(".response.json"))
                    and read(stem.with_suffix(".transport.json")) == transport, "Stock reference raw output differs")
            require(any(s["tokens"] > 1 for s in steps[raw["id"] + "-0"]), "Native stock reference did not compute a real prompt")
            fidelity.append({"id": record["id"], "equal_token_ids": choice["token_ids"] == record["choice"]["token_ids"],
                             "equal_finish_reason": choice["finish_reason"] == record["choice"]["finish_reason"]})
        require(fidelity == read(args.result / "stock-handoff-fidelity.json")
                and all(r["equal_token_ids"] and r["equal_finish_reason"] for r in fidelity), "Stock handoff token fidelity failed")
        reference_summary = {"training_items": len(fidelity), "all_output_tokens_equal": True,
                             "scope": "Finite deterministic training pilot; not population equivalence"}
    require(len(raw_paths) == len(records) * 6 + (len(records) * 3 if reference_summary else 0), "Unexpected raw HTTP file count")
    write_new(args.output, {"complete": True, "requests": len(records), "prefill_arm": start["arm"], "decode_arm": "stock",
              "actual_nixl_bytes": transferred_bytes, "every_request_has_positive_remote_allocation_and_receive": True,
              "every_decode_scheduled_request_one_token": True, "no_prompt_recompute_or_transfer_errors_observed": True,
              "stock_monolithic_reference": reference_summary,
              "worker_identity": worker_proofs, "phase_file_sha256": {**p_hashes, **d_hashes},
              "campaign_result_sha256": file_sha(args.result / "result.json"),
              "audit_helper_sha256": file_sha(__file__), "common_sha256": file_sha(Path(__file__).with_name("common_v2.py")),
              "scope": "Instrumented correctness/quality; no TTFT or uninstrumented performance claim"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("result", "prepared", "prefill-probe", "decode-probe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
