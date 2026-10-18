#!/usr/bin/env python3
"""Canonical task mechanics with honest experimental activation validation.

Never invokes frozen run_stage/rank_evidence or relabels private selection as
public PAR. Frozen public-source checks here identify preparation lineage only.
No cluster, server launch or model operations are implemented.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import urllib.parse
import urllib.request
import urllib.error
import uuid

from stochastic_tail import plugin

TASK_SHA = "6e7905cff18904677c705f233ea502b21c3bf2a746e013c483086ba74fb58cec"


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_module(path):
    need(sha(path) == TASK_SHA, "Canonical task mechanics changed")
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec = importlib.util.spec_from_file_location("stochastic_canonical_task_mechanics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker_identities(directory, plan, plan_sha):
    """Read immutable identity records from otherwise still-active CPU ledgers."""
    values, paths = [], []
    expected_module = ("vllm.model_executor.layers.fused_moe.router.fused_topk_router" if plan["parameters"]["mode"] == "stock"
                       else "placement_router.vendor.fused_topk_router" if plan["parameters"]["mode"] == "par" else "stochastic_tail.plugin")
    expected_source = (plan["native_source_sha256"]["vllm/model_executor/layers/fused_moe/router/fused_topk_router.py"]
                       if plan["parameters"]["mode"] == "stock" else plugin.PUBLIC["placement_router/vendor/fused_topk_router.py"]
                       if plan["parameters"]["mode"] == "par" else plan["private_source_sha256"]["stochastic_tail/plugin.py"])
    expected_file = (Path(plan["native_source_files"]["vllm/model_executor/layers/fused_moe/router/fused_topk_router.py"])
                     if plan["parameters"]["mode"] == "stock" else Path(plan["public_site"]) / "placement_router/vendor/fused_topk_router.py"
                     if plan["parameters"]["mode"] == "par" else Path(plan["private_site"]) / "stochastic_tail/plugin.py")
    private_file = (Path(plan["private_site"]) / "stochastic_tail/plugin.py").resolve()
    gate = json.loads(Path(plan["gpu_gate"]["path"]).read_text())
    for path in sorted(Path(directory).glob("pid*.jsonl")):
        selected = None
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break  # mutable tail may be in progress; identity is earlier
                row = json.loads(line)
                if row.get("kind") == "native_runner_selected":
                    need(selected is None and row.get("complete") is True,"Duplicate/incomplete actual runner selection")
                    selected = {"row":row,"sha256":hashlib.sha256(line).hexdigest()}
                if row.get("kind") != "worker_identity":
                    continue
                need(selected is not None and row.get("observer_revision") == 4
                     and row["actual_runner"] == selected["row"]["actual_runner"], "Missing actual selected native runner callback proof")
                runner=row["actual_runner"]
                native_path=Path(plan["native_source_files"]["vllm/v1/worker/gpu/model_runner.py"]).resolve()
                hooks_path=(Path(plan["private_site"])/"stochastic_tail/native_hooks.py").resolve()
                need(runner["module"] == "vllm.v1.worker.gpu.model_runner" and runner["qualname"] == "GPUModelRunner"
                     and runner["use_v2_model_runner"] is True and Path(runner["source_path"]).resolve() == native_path
                     and runner["source_sha256"] == plan["native_source_sha256"]["vllm/v1/worker/gpu/model_runner.py"], "Actual MRV2 class/source differs")
                need(set(runner["native_methods"]) == set(runner["actual_patched_methods"]) == {"execute_model","_dummy_run","shutdown"}, "Native runner method set differs")
                for name,method in runner["native_methods"].items():
                    need(Path(method["code_filename"]).resolve() == native_path and method["source_sha256"] == runner["source_sha256"], "Unwrapped native runner method source differs")
                    effective=runner["actual_patched_methods"][name]
                    need(Path(effective["code_filename"]).resolve() == hooks_path
                         and effective["source_sha256"] == plan["private_source_sha256"]["stochastic_tail/native_hooks.py"], "Actual native observer wrapper source differs")
                proof=selected["row"]
                need(proof["actual_worker_module"] == "vllm.v1.worker.gpu_worker" and proof["actual_worker_qualname"] == "Worker"
                     and Path(proof["native_worker_load"]["code_filename"]).resolve() == Path(plan["native_source_files"]["vllm/v1/worker/gpu_worker.py"]).resolve()
                     and proof["native_worker_load"]["source_sha256"] == plan["native_source_sha256"]["vllm/v1/worker/gpu_worker.py"]
                     and Path(proof["actual_worker_load"]["code_filename"]).resolve() == hooks_path
                     and proof["actual_worker_load"]["source_sha256"] == plan["private_source_sha256"]["stochastic_tail/native_hooks.py"], "Actual worker load/runner-instance callback differs")
                need(row.get("complete") and row["plan_sha256"] == plan_sha
                     and row["parameters"] == plan["parameters"] and row["runtime"] == plan["runtime"], "Experimental worker plan/runtime differs")
                need(row["private_source_sha256"] == plan["private_source_sha256"]
                     and row["public_source_sha256"] == plugin.PUBLIC
                     and row["source_lineage_commit"] == plugin.PUBLIC_COMMIT
                     and row["native_source_sha256"] == plan["native_source_sha256"]
                     and Path(row["private_module_root"]).resolve() == private_file.parent
                     and all(Path(row["public_module_paths"][name]).resolve() == (Path(plan["public_site"]) / name).resolve()
                             for name in plugin.PUBLIC), "Experimental/private or lineage source differs")
                need(row["actual_compute_callable"]["module"] == row["expected_compute_module"] == expected_module
                     and row["actual_compute_callable"]["source_sha256"] == expected_source
                     and Path(row["actual_compute_callable"]["code_filename"]).resolve() == expected_file.resolve(),
                     "Actual compute callable is not the declared native/PAR/private implementation")
                need(row["actual_route_callable"]["source_sha256"] == row["actual_constructor_callable"]["source_sha256"]
                     == plan["private_source_sha256"]["stochastic_tail/plugin.py"]
                     and Path(row["actual_route_callable"]["code_filename"]).resolve()
                     == Path(row["actual_constructor_callable"]["code_filename"]).resolve() == private_file,
                     "Actual overlay route/constructor source differs")
                p = plan["parameters"]
                need(row["public_par_patched"] is (p["mode"] == "par") and row["experimental_overlay_active"] is True
                     and row["record_calls"] is plan["record_calls"], "Declared selector activation/instrumentation differs")
                need(row["gpu_gate"] == plan["gpu_gate"] and row["checkpoint_proof"] == plan["checkpoint_proof"], "Worker correctness/checkpoint proof differs")
                need(row["model_id"] == plan["model_id"] and row["revision"] == plan["revision"]
                     and Path(row["model_path"]).resolve() == Path(plan["model_path"]).resolve(), "Actual model identity differs")
                need(row["initial_rng_counters"] == {str(i): 0 for i in range(48)}, "Native RNG counters did not start from zero")
                layers = row["native_layers"]
                need(len(layers) == 48 and {x["layer_name"] for x in layers} == plugin.EXPECTED_LAYERS
                     and {x["layer_index"] for x in layers} == set(range(48))
                     and len({x["router_id"] for x in layers}) == 48, "Actual48-layer/router mapping differs")
                for layer in layers:
                    need(layer["top_k"] == 8 and layer["experts"] == 128 and layer["renormalize"] is True
                         and layer["scoring_func"] == "softmax" and layer["eplb"] is False
                         and layer["compute_callable"]["module"] == expected_module
                         and layer["compute_callable"]["source_sha256"] == expected_source
                         and Path(layer["compute_callable"]["code_filename"]).resolve() == expected_file.resolve(),
                         "A native layer has an unexpected actual selector")
                    need(layer["actual_public_keep"] == (p["keep"] if p["mode"] == "par" else None), "Public/native/private keep activation differs")
                need(row["public_load_penalty"] == p["load_penalty"], "Actual deterministic load penalty differs")
                need(row["effective_load_tile"] == ((16 if p["load_penalty"] else 1) if p["mode"] == "par" else None), "Effective load tile differs")
                need(row["pid"] == int(path.stem.removeprefix("pid")) and "H100" in row["gpu"]
                     and str(row["vllm_version"]).split("+")[0] == "0.31.0"
                     and row["torch_version"] == gate["torch_version"], "Actual worker PID/GPU/runtime differs")
                values.append(row)
                paths.append({"ledger_path": str(path), "identity_record_sha256": hashlib.sha256(line).hexdigest(), "native_runner_selected_record_sha256": selected["sha256"]})
                break
    need(len(values) == 4 and {x["ep_rank"] for x in values} == set(range(4)) and len({x["pid"] for x in values}) == 4,
         "Require exactly4 distinct physical experimental worker identities")
    return values, paths


def native_checkpoint(url, output, plan, plan_sha, phase, workers):
    """Native pause is the all-DP idle barrier; RPC snapshots precede teardown."""
    prefix=Path(output)/"native-control";prefix.mkdir(exist_ok=False)
    transcript=[]
    def control(name,path,body=None):
        raw_body=None if body is None else json.dumps(body,sort_keys=True,separators=(",",":")).encode()
        body_file=prefix/(name+".body.json");body_file.write_bytes(raw_body or b"")
        task_request={"method":"GET" if raw_body is None else "POST","url":url+path,"body":body,
                      "body_file":str(body_file),"body_sha256":sha(body_file)}
        req=prefix/(name+".request.json");req.write_text(json.dumps(task_request,sort_keys=True,indent=2)+"\n")
        response_path=prefix/(name+".response.json")
        submitted=time.time_ns();status=None;error=None;raw=b"";headers=[];response_complete=False
        try:
            request=urllib.request.Request(url+path,raw_body,{"Content-Type":"application/json"})
            with urllib.request.urlopen(request,timeout=120) as response:
                status=response.status;headers=list(response.headers.items())
                raw=response.read();response_complete=True
        except urllib.error.HTTPError as failure:
            status=failure.code;headers=list(failure.headers.items());error=str(failure)
            try:raw=failure.read();response_complete=True
            except http.client.IncompleteRead as partial:raw=partial.partial
            raise
        except http.client.IncompleteRead as failure:
            raw=failure.partial;error=repr(failure);raise
        except BaseException as failure:
            error=repr(failure);raise
        finally:
            returned=time.time_ns()
            response_path.write_bytes(raw)
            metadata={"name":name,"submitted_unix_ns":submitted,"returned_unix_ns":returned,"status":status,"error":error,
                      "headers":headers,"response_complete":response_complete}
            (prefix/(name+".transport.json")).write_text(json.dumps(metadata,sort_keys=True,indent=2)+"\n")
        need(status==200 and response_complete,"Native control failed: "+name)
        transcript.append({"name":name,"request_path":str(req),"request_sha256":sha(req),
                           "response_path":str(response_path),"response_sha256":sha(response_path),"body_path":str(body_file),"body_sha256":sha(body_file),"status":status,"headers":headers,"response_complete":response_complete,"submitted_unix_ns":submitted,"returned_unix_ns":returned})
        return json.loads(raw)
    need(control("pause","/pause?mode=wait&clear_cache=false",{}).get("status")=="paused","Native pause did not complete")
    need(control("is-paused","/is_paused").get("is_paused") is True,"Native engine not paused")
    nonce=uuid.uuid4().hex;sealed=phase in ("proof","score")
    payload={"plan_sha256":plan_sha,"phase":phase,"nonce":nonce,"sealed":sealed}
    result=control("checkpoint","/collective_rpc",{"method":"stochastic_tail_checkpoint",
        "args":[json.dumps(payload,sort_keys=True,separators=(",",":"))],"timeout":90})
    need(isinstance(result.get("results"),list) and result["results"],"Native collective RPC returned no first-core evidence")
    # DPLB utility broadcasts every DP engine but returns only first-core result.
    # Completeness comes from exact matching immutable records on all4workers.
    snapshots=[]
    expected={(w["pid"],w["ep_rank"]) for w in workers}
    for worker in workers:
        ledger=Path(plan["evidence_directory"])/f"pid{worker['pid']}.jsonl"
        matches=[]
        need(ledger.is_file(),"Missing physical worker checkpoint ledger")
        with ledger.open("rb") as stream:
            for line in stream:
                need(line.endswith(b"\n"),"Partial worker record at paused checkpoint")
                row=json.loads(line)
                if row.get("kind")=="native_checkpoint" and row.get("nonce")==nonce:
                    need(row.get("complete") is True and row["phase"]==phase and row["sealed"] is sealed
                         and row["plan_sha256"]==plan_sha and row["parameters"]==plan["parameters"]
                         and (row["pid"],row["ep_rank"]) == (worker["pid"],worker["ep_rank"]) and row["cuda_synchronized"] is True
                         and row["active_step"] is None and row["actual_runner"]==worker["actual_runner"],"Native paused checkpoint differs")
                    callback=row["checkpoint_callable"]
                    need(callback["module"]=="stochastic_tail.worker_extension"
                         and Path(callback["code_filename"]).resolve()==(Path(plan["private_site"])/"stochastic_tail/worker_extension.py").resolve()
                         and callback["source_sha256"]==plan["private_source_sha256"]["stochastic_tail/worker_extension.py"],"Actual native checkpoint callback source differs")
                    need(set(row["counters"])=={str(i) for i in range(48)}
                         and all(type(v) is int and v>0 for v in row["counters"].values()),"Missing/invalid native final counters")
                    matches.append({"row":row,"ledger_path":str(ledger),"record_sha256":hashlib.sha256(line).hexdigest()})
        need(len(matches)==1,"Need exactlyone same-nonce checkpoint per physical worker")
        snapshots.extend(matches)
    need(len(snapshots)==4 and {(r["row"]["pid"],r["row"]["ep_rank"]) for r in snapshots}==expected,"Incomplete all4native checkpoint")
    for response in result["results"]:
        need(response["row"]["kind"]=="native_checkpoint" and response["row"]["nonce"]==nonce
             and any(response["record_sha256"]==r["record_sha256"] and response["row"]==r["row"] for r in snapshots),"RPC response differs from actual worker record")
    if not sealed:
        need(control("resume","/resume",{}).get("status")=="resumed","Native resume failed")
        need(control("is-resumed","/is_paused").get("is_paused") is False,"Native scheduler stayed paused")
    receipt={"complete":True,"phase":phase,"nonce":nonce,"sealed":sealed,"snapshots":snapshots,"transcript":transcript,
             "scope":"All4raw native collective-RPC checkpoints after allDPpause; response may include firstcore only. Terminal seal forbids any subsequent routing."}
    path=prefix/"receipt.json";path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return {"path":str(path),"sha256":sha(path),"nonce":nonce,"phase":phase,"sealed":sealed,"complete":True}


def run(args):
    task = task_module(args.task_harness)
    plan = plugin.read_plan(args.plan, args.plan_sha256)
    need(plan["purpose"] == ("proof" if args.mode == "proof" else "quality") and plan["record_calls"], "Quality adapter requires declared call ledger")
    parsed = urllib.parse.urlsplit(args.url)
    need(parsed.scheme == "http" and parsed.hostname == "127.0.0.1" and parsed.port
         and not parsed.username and not parsed.password, "Use explicit local serving URL")
    metadata, requests, training = task.load_prepared(args.prepared)
    lineage = json.loads(Path(plan["public_source_proof"]["path"]).read_text())
    need(task.source_identity(plan["public_source_proof"]["path"], lineage["source_root"]) == metadata["source"], "Canonical prompt preparation lineage differs")
    need(task.checkpoint_identity(plan["checkpoint_proof"]["path"], plan["model_path"]) == metadata["checkpoint"], "Canonical preparation checkpoint/stat identity differs")
    need(metadata["protocol"]["gsm8k"]["max_output_tokens"] == 2048 and metadata["max_model_len"] == 8192, "Frozen prompt budget differs")
    if args.mode == "score":
        before, before_paths = worker_identities(plan["evidence_directory"], plan, args.plan_sha256)
        prior = task.read(args.preflight / "result.json")
        need(prior["complete"] and prior["preflight_pass"] and prior["experiment_plan_sha256"] == args.plan_sha256
             and prior["preparation_sha256"] == task.digest(args.prepared / "preparation.json")
             and prior["worker_pids"] == sorted(x["pid"] for x in before)
             and prior["worker_identity_records"] == before_paths, "Full scoring requires the same-server private-contract training preflight")
    task.own_directory(args.output)
    state = {"complete": False, "mode": args.mode, "arm": plan["arm"], "router_seed": plan["parameters"]["seed"],
             "ep": 4, "experiment_plan_sha256": args.plan_sha256, "parameters": plan["parameters"],
             "canonical_harness_sha256": TASK_SHA, "adapter_sha256": sha(__file__),
             "preparation_sha256": task.digest(args.prepared / "preparation.json"),
             "started_unix_ns": time.time_ns(), "cohorts_completed": 0,
             "observer_revision": 4,
             "observer_control_protocol": "Actual MRV2 hooks; native pause/checkpoint/resume instrumentation changes scheduling; do not pool failed v3 outcomes",
             "actual_selection_contract": "Private entrypoint actual callable/source+same-server48routers; public COMMIT hashes establish lineage only",
             "score_interpretation": "Teacher-forced Hella classification under the recorded routing/scheduling/RNG; not a claim of strict causal model likelihood",
             "records": []}
    task.atomic(args.output / "result.json", state)
    try:
        for number, rows in enumerate(task.cohorts(training if args.mode in ("preflight", "proof") else requests)):
            request_id = f"stochastic-{plan['arm']}-seed{plan['parameters']['seed']}-{args.mode}-{number:05d}"
            state["records"].extend(task.http_cohort(args.url, rows, args.served_model, args.output / f"cohort{number:05d}",
                    2048, args.request_timeout, request_id))
            state["cohorts_completed"] += 1
            task.atomic(args.output / "result.json", state)
        evidence, paths = worker_identities(plan["evidence_directory"], plan, args.plan_sha256)
        state.update(worker_pids=sorted(x["pid"] for x in evidence), worker_evidence=evidence,
                     worker_identity_records=paths, requests_completed=len(state["records"]))
        state["native_checkpoint"] = native_checkpoint(args.url, args.output, plan, args.plan_sha256, args.mode, evidence)
        if args.mode in ("preflight", "proof"):
            gsm = [r for r in state["records"] if r["task"] == "gsm8k"]
            need(len(gsm) == 8 and len(state["records"]) == 24, "Incomplete training-only protocol preflight")
            state.update(preflight_pass=True, complete=True,
                training_generation_diagnostics={key: sum(bool(r[key]) for r in gsm) for key in ["empty", "parse_failed", "truncated"]},
                preflight_note="Training protocol/response/source checks only; no accuracy threshold, seed/range/temperature selection or best-seed filtering. Empty/parse/truncation outcomes retained.")
        else:
            need(len(state["records"]) == len(requests), "Incomplete held-out request set")
            state.update(complete=True, task_summary=task.summarize_records(state["records"]))
    except BaseException:
        state["error"] = traceback.format_exc()
        raise
    finally:
        state["finished_unix_ns"] = time.time_ns()
        task.atomic(args.output / "result.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["proof", "preflight", "score"])
    for name in ["task-harness", "plan", "prepared", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8077")
    parser.add_argument("--served-model", default="task-model")
    parser.add_argument("--request-timeout", type=int, default=600)
    args = parser.parse_args()
    need(args.mode != "score" or args.preflight is not None, "Score requires same-server --preflight")
    run(args)
