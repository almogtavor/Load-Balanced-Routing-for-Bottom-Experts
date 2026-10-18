#!/usr/bin/env python3
"""Audit a CLOSED instrumented two-leg campaign, including real transfer use."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
from common_v2 import SOURCE_COMMIT, SOURCE, base_body, choices, encoded, file_sha, read, require, sha, write_new
from native_accounting import zero_cached_transfer, zero_cached_usage
from hybrid_client import paired_body, transfer_params, reference_difference
from native_runner_contract import NATIVE, RUNNER
from native_request_id import (SOURCES as REQUEST_SOURCES, assignment_index, external_item_id, producer_internal_id, resolve_assignment)


def runner_contract(worker, contract):
    proof = worker.get("native_runner")
    require(isinstance(proof, dict) and proof["class"] == RUNNER and proof["use_v2_model_runner"] is True
            and proof["native_sources"] == contract["native_runner_sources"]
            and proof["native_sources"]["sha256"] == NATIVE
            and proof["class_source"] == proof["native_sources"]["files"]["v1/worker/gpu/model_runner.py"]
            and proof["runner_selection"] == "observed native default; no V2 environment override",
            "Actual selected native V2 class/config/source differs")
    execute = proof["original_execute"]
    require(execute["module"] == "vllm.v1.worker.gpu.model_runner"
            and execute["qualname"] == "GPUModelRunner.execute_model"
            and execute["unwrapped_code_filename"] == proof["class_source"]
            and execute["unwrapped_source_sha256"] == NATIVE["v1/worker/gpu/model_runner.py"]
            and execute["parameter_names"][:3] == ["self", "scheduler_output", "intermediate_tensors"]
            and execute["dummy_run_default"] is False and execute["is_profile_default"] is False,
            "Actual original V2 execute contract differs")
    require(worker["runner_contract_sha256"] == contract["runner_contract_sha256"]
            == file_sha(Path(__file__).with_name("native_runner_contract.py")), "Runner helper binding differs")


def steps_by_request(events, workers):
    by_rank = {r["ep_rank"]: r["pid"] for r in workers}
    result = defaultdict(list)
    for row in events:
        if row["kind"] != "worker_step":
            continue
        scheduled = row["scheduled_tokens"]
        require(isinstance(scheduled, dict) and all(isinstance(k, str) and type(v) is int and v > 0 for k, v in scheduled.items())
                and row["total_scheduled_tokens"] == sum(scheduled.values())
                and row["pid"] == by_rank.get(row["ep_rank"]), "Native scheduled-token/worker accounting differs")
        require(type(row["dummy_run"]) is bool and type(row["is_profile"]) is bool
                and row["has_real_scheduled_requests"] is (bool(scheduled) and not row["dummy_run"]),
                "V2 explicit dummy/profile phase flags differ")
        if row["dummy_run"]:
            continue
        require(row["is_profile"] is False, "Real application requests cannot use profile execution")
        for request_id, n in scheduled.items():
            result[request_id].append({"tokens": n, "rank": row["ep_rank"],
                "contract": row["request_contracts"].get(request_id),
                "router_objects": row["distinct_router_objects"], "router_calls": row["router_calls"]})
    return result


def verify_handoff_steps(record, prompt, request_steps, production):
    require(request_steps and all(r["tokens"] == 1 and r["rank"] == record["dp_rank"]
                                  and r["router_objects"] == r["router_calls"] == 48 for r in request_steps),
            "Decode recomputed multi-token prompt, used wrong rank, or missed routing phase coverage")
    require(len(request_steps) >= max(1, len(record["choice"]["token_ids"])),
            "Native stock decode phase records do not cover all returned output tokens")
    initial = next((r["contract"] for r in request_steps if r["contract"] is not None), None)
    require(initial and initial["prompt_len"] == record["prompt_tokens"]
            and initial["initial_computed_tokens"] == record["prompt_tokens"] - 1
            and initial["prompt_sha256"] == record["prompt_sha256"]
            and initial["last_prompt_token_id"] == prompt[-1]
            and initial["native_all_token_ids_sha256"] == record["prompt_sha256"]
            and initial["native_all_token_ids_len"] == len(prompt) and initial["native_last_prefill_token_id"] == prompt[-1],
            "Decoder must compute precisely the final native prompt token after transferred prefix")
    require(production and sum(r["tokens"] for r in production) == len(prompt)
            and all(type(r["tokens"]) is int and r["tokens"] > 0 and r["rank"] == record["dp_rank"]
                    and r["router_objects"] == r["router_calls"] == 48 for r in production),
            "Producer did not compute exact full prompt with all48 routers")
    production_initial = next((r["contract"] for r in production if r["contract"] is not None), None)
    require(production_initial and production_initial["initial_computed_tokens"] == 0
            and production_initial["prompt_len"] == len(prompt) and production_initial["prompt_sha256"] == record["prompt_sha256"]
            and production_initial["last_prompt_token_id"] == prompt[-1]
            and production_initial["native_all_token_ids_sha256"] == record["prompt_sha256"]
            and production_initial["native_all_token_ids_len"] == len(prompt)
            and production_initial["native_last_prefill_token_id"] == prompt[-1], "Producer prompt/source native scheduler identity differs")


def load_events(folder):
    import json
    files = sorted(folder.glob("pid*.jsonl"))
    require(files, "Missing native phase evidence")
    require(not folder.is_symlink() and set(folder.iterdir()) == set(files)
            and all(p.is_file() and not p.is_symlink() for p in files), "Unknown/symlink phase evidence")
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
    require(start["stock_reference_policy"]=="V6 diagnostic only; stock-handoff is baseline; failedV5 preserved", "Wrong prospective monolithic diagnostic policy")
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
    allowed = {"installed", "worker_identity", "runner_identity", "worker_step", "route_outside_execute",
               "scheduler_matched", "scheduler_allocated", "scheduler_finished", "worker_transfer_finished",
               "worker_transfer_stats", "worker_transfer_errors", "request_id_assigned"}
    require(all(r["kind"] in allowed for r in p_events + d_events), "Unknown native phase-event kind")
    earliest_request = min(read(p)["submitted_unix_ns"] for p in raw_paths if p.name.endswith(".request.json"))
    require(all(r["unix_ns"] < earliest_request for r in p_events + d_events if r["kind"] == "route_outside_execute"),
            "Application routing occurred outside the observed native V2 execute path")
    require(all(r["probe_sha256"] == start["phase_probe_sha256"] for r in p_events + d_events), "Phase probe identity differs")
    require(all(r["runner_contract_sha256"] == file_sha(Path(__file__).with_name("native_runner_contract.py"))
                for r in p_events + d_events), "Phase runner helper identity differs")
    require(all(r["role"] == "prefill" and r["arm"] == start["arm"] for r in p_events), "Prefill role/arm differs")
    require(all(r["role"] == "decode" and r["arm"] == "stock" for r in d_events), "Decode must remain stock")
    request_helper_sha = file_sha(Path(__file__).with_name("native_request_id.py"))
    require(all(r["request_id_contract_sha256"]==request_helper_sha for r in p_events+d_events), "Native request-ID helper binding differs")
    assignments = {}
    for role, events in (("prefill",p_events),("decode",d_events)):
        contract=start[role+"_contract"]
        require(contract["request_id_contract_sha256"]==request_helper_sha
                and contract["request_id_sources"]["sha256"]==REQUEST_SOURCES, "Native request-ID source contract absent")
        require(all(r["request_id_sources"]==contract["request_id_sources"] for r in events if r["kind"]=="installed"),
                "Actual installed native API source differs")
        assignments[role]=assignment_index(events,contract["request_id_sources"])
    expected_assignment_keys={"prefill":set(),"decode":set()}
    resolved_native_ids=[]
    worker_proofs = {}
    for role, events in (("prefill", p_events), ("decode", d_events)):
        identities = [r for r in events if r["kind"] == "worker_identity"]
        runners = [r for r in events if r["kind"] == "runner_identity"]
        require(len(identities) == 4 and {r["ep_rank"] for r in identities} == set(range(4))
                and len({r["pid"] for r in identities}) == 4, "Missing/duplicate physical EP workers")
        require(len(runners) == 4 and {r["ep_rank"] for r in runners} == set(range(4))
                and {r["pid"] for r in runners} == {r["pid"] for r in identities}, "Missing actual V2 runner worker proof")
        for worker in identities:
            expected_source = dict(zip(("kernel", "plugin", "vendor"), SOURCE.values()))
            require(worker["source_commit"] == SOURCE_COMMIT and len(worker["contracts"]) == 48
                    and worker["ep_size"] == 4 and worker["kv_connector"] == "NixlConnector"
                    and worker["kv_load_failure_policy"] == "fail" and worker["prefix_caching"] is False
                    and worker["source_sha256"] == expected_source,
                    "Incomplete/wrong runtime router/transfer identity")
            contract = start[role + "_contract"]
            runner_contract(worker, contract)
            runner = next(r for r in runners if r["pid"] == worker["pid"])
            require(runner["native_runner"] == worker["native_runner"] and runner["ep_rank"] == worker["ep_rank"],
                    "First actual execute/48-router worker identity differs")
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
    for row in d_events:
        if row["kind"] == "scheduler_allocated":
            allocations[row["request_id"]].append(row)
    steps = steps_by_request(d_events, [r for r in d_events if r["kind"] == "worker_identity"])
    prefill_steps = steps_by_request(p_events, [r for r in p_events if r["kind"] == "worker_identity"])
    seen = set()
    for i, record in enumerate(records):
        row, position, prompt = jobs[i]
        require(record["id"] == row["id"] and record["position"] == position and record["dp_rank"] == i % 4
                and record["request_id"] == start["launch_id"] + f"-{i:06d}", "Request/frozen source identity differs")
        request_id = resolve_assignment(assignments["decode"],record["consumer_response_id"],prompt,record["consumer_transport"])
        producer_id = resolve_assignment(assignments["prefill"],record["producer_response_id"],prompt,record["producer_transport"])
        require(record["consumer_external_request_id"]==external_item_id(record["consumer_response_id"])
                and producer_id==record["producer_native_request_id"]==record["transfer_params"]["remote_request_id"],
                "Full producer/consumer native ID mapping differs")
        expected_assignment_keys["prefill"].add(external_item_id(record["producer_response_id"]))
        expected_assignment_keys["decode"].add(external_item_id(record["consumer_response_id"]))
        resolved_native_ids.append({"request_id":record["request_id"],"producer_internal_request_id":producer_id,"consumer_internal_request_id":request_id})
        require(request_id not in seen and request_id in received, "Missing/duplicate actual receive completion")
        seen.add(request_id)
        allocated = allocations[request_id]
        require(any(r["external_tokens"] >= record["prompt_tokens"] - 1 and r["external_tokens"] > 0
                    and r["params"].get("remote_request_id") == record["producer_native_request_id"] for r in allocated),
                "No matched external KV allocation bound to this producer request")
        request_steps = steps[request_id]
        production = prefill_steps[record["producer_native_request_id"]]
        verify_handoff_steps(record, prompt, request_steps, production)
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
                d_response = raw
                require(len(raw["choices"]) == 1 and raw["choices"][0] == record["choice"]
                        and raw["id"] == record["consumer_response_id"]
                        and external_item_id(raw["id"]) == record["consumer_external_request_id"]
                        and resolve_assignment(assignments["decode"],raw["id"],prompt,transport)==request_id, "Recorded output differs from preserved raw response")
            else:
                require(transfer_params(raw, len(prompt)) == record["transfer_params"]
                        and raw["id"] == record["producer_response_id"]
                        and producer_internal_id(raw)==producer_id==record["producer_native_request_id"], "Actual producer transfer identity differs")
        require(p_request["prompt"] == d_request["prompt"]
                and sha(encoded(d_request["prompt"])) == record["prompt_sha256"]
                and d_request["kv_transfer_params"] == record["transfer_params"]
                and d_request["max_tokens"] == (1 if start["mode"] == "prefix-score" else start["max_tokens"]),
                "HTTP handoff input/params/cap differ")
        require("prompt_logprobs" not in d_request, "Do not recompute full prompt LP on decoder")
        zero_cached_transfer(record["transfer_params"], p_request["kv_transfer_params"], d_request["kv_transfer_params"])
        zero_cached_usage(d_response["usage"], len(prompt), len(record["choice"]["token_ids"]))
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
            reference_native = resolve_assignment(assignments["decode"],raw["id"],body["prompt"],transport)
            expected_assignment_keys["decode"].add(external_item_id(raw["id"]))
            require(any(s["tokens"] > 1 for s in steps[reference_native]), "Native stock reference did not compute a real prompt")
            fidelity.append(reference_difference(record["id"], record["choice"], choice))
        require(fidelity == read(args.result / "stock-handoff-fidelity.json"), "Stock reference diagnostic changed")
        reference_summary = {"training_items": len(fidelity),
                             "all_output_tokens_equal": all(r["equal_token_ids"] for r in fidelity),
                             "all_finish_reasons_equal": all(r["equal_finish_reason"] for r in fidelity),
                             "per_item_differences": fidelity,
                             "scope": "V6 diagnostic only; no monolithic output equivalence claim; failedV5 preserved"}
    require(all(set(assignments[role])==expected_assignment_keys[role] for role in assignments), "Unknown/missing native API assignment inventory")
    mapping_pids={role:sorted({r["pid"] for r in index.values()}) for role,index in assignments.items()}
    require(len(raw_paths) == len(records) * 6 + (len(records) * 3 if reference_summary else 0), "Unexpected raw HTTP file count")
    write_new(args.output, {"complete": True, "prospective_revision": 6, "monolithic_equivalence_claim": False, "comparison_baseline": "Actual stock-prefill/stock-decode NIXL handoff", "requests": len(records), "prefill_arm": start["arm"], "decode_arm": "stock",
              "actual_nixl_bytes": transferred_bytes, "every_request_has_positive_remote_allocation_and_receive": True,
              "every_decode_scheduled_request_one_token": True, "no_prompt_recompute_or_transfer_errors_observed": True,
              "actual_v2_runner_on_all8_workers": True, "every_producer_full_prompt_all48_routers": True,
              "decoder_exact_transferred_prefix_then_native_last_prompt_token": True,
              "stock_monolithic_reference": reference_summary,
              "exact_native_external_internal_mapping": True, "resolved_native_ids": resolved_native_ids,
              "request_id_mapping_pids":mapping_pids, "request_id_sources":start["prefill_contract"]["request_id_sources"],
              "request_id_contract_sha256":request_helper_sha,
              "worker_identity": worker_proofs, "phase_file_sha256": {**p_hashes, **d_hashes},
              "campaign_result_sha256": file_sha(args.result / "result.json"),
              "audit_helper_sha256": file_sha(__file__), "common_sha256": file_sha(Path(__file__).with_name("common_v2.py")),
              "scope": "Instrumented correctness/quality; no TTFT or uninstrumented performance claim"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("result", "prepared", "prefill-probe", "decode-probe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
