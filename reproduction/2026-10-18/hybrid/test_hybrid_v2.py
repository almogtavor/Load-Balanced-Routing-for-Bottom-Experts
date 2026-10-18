"""CPU-only native-V2/last-token/source/lifetime regression guards."""
import ast
import copy
import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import types
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import native_runner_contract as native
import hybrid_phase_probe as probe
import audit_hybrid as audit
import hybrid_driver as driver
import hybrid_client as client
import native_request_id as request_ids
from common_v2 import encoded, sha, write_new, read, file_sha

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[3] / 'runs/paper-followup-20261018/hybrid-causality'
DIAG = BASE.parent / "audit/native-runner-diagnosis-v1"


class V2Tests(unittest.TestCase):
    def function(self):
        namespace = {}
        exec(compile("def execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False, "
                     "skip_attn_for_dummy_run=False, is_profile=False, context_len=0, "
                     "valid_dummy_state_slots=False, randomize_inputs=False): return None",
                     str(DIAG / "v1/worker/gpu/model_runner.py"), "exec"), namespace)
        fn = namespace["execute_model"]
        fn.__module__ = "vllm.v1.worker.gpu.model_runner"
        fn.__qualname__ = "GPUModelRunner.execute_model"
        return fn

    def test_actual_image_native_source_readback(self):
        source = native.installed_sources(DIAG)
        self.assertEqual(source["sha256"], native.NATIVE)

    def test_actual_v2_self_source_and_signature(self):
        cls = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
        obj = cls(); obj.vllm_config = SimpleNamespace(use_v2_model_runner=True)
        proof = native.actual_runner(obj, cls, self.function(), native.installed_sources(DIAG))
        self.assertEqual(proof["class"], native.RUNNER)
        self.assertFalse(proof["original_execute"]["dummy_run_default"])
        for changed in (False, None, 1):
            obj.vllm_config.use_v2_model_runner = changed
            with self.assertRaises(RuntimeError):
                native.actual_runner(obj, cls, self.function(), native.installed_sources(DIAG))

    def test_legacy_v1_class_and_changed_source_cannot_pass(self):
        old = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu_model_runner"})
        obj = old(); obj.vllm_config = SimpleNamespace(use_v2_model_runner=True)
        with self.assertRaises(RuntimeError):
            native.actual_runner(obj, old, self.function(), native.installed_sources(DIAG))
        source = native.installed_sources(DIAG)
        source["files"]["v1/worker/gpu/model_runner.py"] = str(HERE / "hybrid_phase_probe.py")
        cls = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
        obj = cls(); obj.vllm_config = SimpleNamespace(use_v2_model_runner=True)
        with self.assertRaises(RuntimeError):
            native.actual_runner(obj, cls, self.function(), source)

    def test_native_v2_dummy_flag_bound_positional_and_keyword(self):
        signature = inspect.signature(self.function())
        self.assertEqual(native.execute_flags(signature, object(), object(), (), {}), {"dummy_run": False, "is_profile": False})
        self.assertEqual(native.execute_flags(signature, object(), object(), (None, True), {"is_profile": True}),
                         {"dummy_run": True, "is_profile": True})
        for kwargs in ({"dummy_run": 1}, {"is_profile": None}):
            with self.assertRaises(RuntimeError): native.execute_flags(signature, object(), object(), (), kwargs)

    def test_native_request_shape_has_no_invented_prompt_len(self):
        row = SimpleNamespace(req_id="r", prompt_token_ids=[11, 22, 33], num_computed_tokens=2, prefill_token_ids=[11, 22, 33])
        self.assertFalse(hasattr(row, "prompt_len"))
        self.assertEqual(probe.native_prompt(row), {"prompt_len": 3, "initial_computed_tokens": 2,
            "prompt_sha256": sha(encoded([11, 22, 33])), "last_prompt_token_id": 33,
            "native_all_token_ids_sha256": sha(encoded([11, 22, 33])), "native_all_token_ids_len": 3, "native_last_prefill_token_id": 33})
        for ids in (None, [], [1, True], [-1]):
            row.prompt_token_ids = ids
            with self.assertRaises(RuntimeError): probe.native_prompt(row)
        row.prompt_token_ids = [11, 22, 33]; row.prefill_token_ids = [11, 22, 99]
        with self.assertRaises(RuntimeError): probe.native_prompt(row)

    def test_actual_retained_new_request_dataclass(self):
        path = DIAG / "v1/core/sched/output.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NewRequestData")
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        module = types.ModuleType("retained_native_new_request_fixture")
        module.__dict__["dataclass"] = dataclasses.dataclass
        sys.modules[module.__name__] = module
        try:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"), module.__dict__)
            row = module.NewRequestData(req_id="r", prompt_token_ids=[11, 22, 33], mm_features=[], sampling_params=None,
                pooling_params=None, block_ids=([],), num_computed_tokens=2, lora_request=None, prefill_token_ids=[11, 22, 33])
            self.assertEqual(probe.native_prompt(row)["prompt_len"], 3)
            self.assertEqual(probe.native_prompt(row)["last_prompt_token_id"], 33)
        finally:
            sys.modules.pop(module.__name__)

    def fixture(self):
        prompt = [11, 22, 33, 44]
        record = {"dp_rank": 2, "prompt_tokens": 4, "prompt_sha256": sha(encoded(prompt)), "choice": {"token_ids": [5]}}
        contract = {"prompt_len": 4, "initial_computed_tokens": 3, "prompt_sha256": record["prompt_sha256"],
                    "last_prompt_token_id": 44, "native_all_token_ids_sha256": record["prompt_sha256"],
                    "native_all_token_ids_len": 4, "native_last_prefill_token_id": 44}
        decode = [{"tokens": 1, "rank": 2, "router_objects": 48, "router_calls": 48, "contract": contract}]
        production = [{"tokens": 3, "rank": 2, "router_objects": 48, "router_calls": 48,
                       "contract": {**contract, "initial_computed_tokens": 0}},
                      {"tokens": 1, "rank": 2, "router_objects": 48, "router_calls": 48, "contract": None}]
        return record, prompt, decode, production

    def test_exact_transferred_prefix_last_token_and_chunked_producer(self):
        audit.verify_handoff_steps(*self.fixture())

    def test_recompute_missing_router_wrong_last_token_and_source_rejected(self):
        for field, value in (("initial_computed_tokens", 2), ("initial_computed_tokens", 4),
                             ("prompt_sha256", "wrong"), ("last_prompt_token_id", 99),
                             ("native_all_token_ids_sha256", "wrong"), ("native_last_prefill_token_id", 99),
                             ("native_all_token_ids_len", 3)):
            args = copy.deepcopy(self.fixture()); args[2][0]["contract"][field] = value
            with self.assertRaises(RuntimeError): audit.verify_handoff_steps(*args)
        for field, value in (("tokens", 4), ("router_calls", 47), ("rank", 1)):
            args = copy.deepcopy(self.fixture()); args[2][0][field] = value
            with self.assertRaises(RuntimeError): audit.verify_handoff_steps(*args)
        args = copy.deepcopy(self.fixture()); args[3].pop()
        with self.assertRaises(RuntimeError): audit.verify_handoff_steps(*args)
        args = copy.deepcopy(self.fixture()); args[0]["choice"]["token_ids"] = [5, 6]
        with self.assertRaises(RuntimeError): audit.verify_handoff_steps(*args)

    def test_native_dummy_requests_cannot_supply_application_coverage(self):
        event = {"kind": "worker_step", "pid": 40, "ep_rank": 0, "scheduled_tokens": {"r": 1},
                 "total_scheduled_tokens": 1, "dummy_run": True, "is_profile": False,
                 "has_real_scheduled_requests": False, "request_contracts": {"r": {}},
                 "router_calls": 48, "distinct_router_objects": 48}
        self.assertEqual(dict(audit.steps_by_request([event], [{"ep_rank": 0, "pid": 40}])), {})
        for patch_field, value in (("has_real_scheduled_requests", True), ("pid", 41), ("total_scheduled_tokens", 2)):
            with self.assertRaises(RuntimeError):
                audit.steps_by_request([{**event, patch_field: value}], [{"ep_rank": 0, "pid": 40}])

    def test_scientific_server_commands_and_clean_environment_unchanged(self):
        plan = {"vllm": "/usr/local/bin/vllm", "model": "pinned", "gpu_util": .88,
                "site_root": "/pristine", "phase_probe": "/new/probe.py"}
        for arm in driver.ARMS:
            for role in ("prefill", "decode"):
                env = driver.environment(plan, role, arm)
                self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", env)
                self.assertNotIn("PLACEMENT_PROBE", env)
                self.assertEqual("PLACEMENT_KEEP" in env, role == "prefill" and arm != "stock")
                command = driver.commands(plan, role)
                self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
                self.assertEqual(command[command.index("--max-num-batched-tokens") + 1], "16384")

    def test_frozen_wire_helpers_remain_byte_identical(self):
        for name in ("common_v2.py", "score_hybrid_gsm.py"):
            self.assertEqual((HERE / name).read_bytes(), (BASE / name).read_bytes())
        original = ast.parse((BASE / "hybrid_driver.py").read_text())
        derived = ast.parse((HERE / "hybrid_driver.py").read_text())
        for name in ("commands", "environment"):
            a = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in derived.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))

    def test_all_failed_prior_closures_match_retained_bytes(self):
        captures = BASE.parent / "audit/initial-native-observer-failures-v1"
        mapping = {remote: captures / remote.removeprefix("/work/results/") for remote in driver.FAILED_PRIORS}
        mapping["/work/results/prefill-profiles-v2/receipt.json"] = BASE.parent / "audit/profile-v2-first-attempt-failure/receipt.json"
        mapping["/work/results/prefill-profiles-v3/receipt.json"] = HERE / "fixtures/profile-v3-failed-receipt.json"
        mapping["/work/results/hybrid-v2/proof/receipt.json"] = HERE/"fixtures/hybrid-v3-failed-proof-receipt.json"
        mapping["/work/results/hybrid-outer-driver-v3.json"] = HERE/"fixtures/hybrid-v3-failed-outer-receipt.json"
        mapping["/work/results/hybrid-v2/proof/sweep00-position0-stock/launch.json"] = HERE/"fixtures/hybrid-v3-failed-stock-launch.json"
        binding={"original_pod":driver.recovery.OLD,"prior_controller_closure":driver.recovery.CLOSURE}
        paths=[driver.recovery.prior_path(r) for r in driver.FAILED_PRIORS]
        old_identity_lookup=lambda p: mapping[driver.recovery.original_path(p)]
        lib=SimpleNamespace(process_record=lambda pid: self.fail("Archived PIDs must never be inspected on replacement host"))
        with patch.object(driver, "read", side_effect=lambda p: json.loads(old_identity_lookup(p).read_bytes())), \
             patch.object(driver, "file_sha", side_effect=lambda p: native.digest(old_identity_lookup(p))):
            rows = driver.failed_prior_gate(paths, lib, binding)
            self.assertEqual(len(rows), 7)
            self.assertTrue(all(row["closure_scope"]==driver.recovery.CLOSURE for row in rows))
            with self.assertRaises(RuntimeError):driver.failed_prior_gate(paths[:-1], lib, binding)
            with self.assertRaises(RuntimeError):driver.failed_prior_gate([Path(r) for r in driver.FAILED_PRIORS], lib, binding)

    def test_four_request_two_leg_wire_and_native_phase_audit(self):
        """Real file writer + parser + V2 phase auditor, synthetic worker events.

        This exercises integrity plumbing only and is never a CUDA/KV proof.
        """
        class Response:
            status, headers = 200, {"Content-Type": "application/json"}
            def __init__(self, request): self.request = request
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                body = json.loads(self.request.data); ids = body["prompt"]; identifier = "cmpl-" + body["request_id"]
                is_prefill = "prefill" in self.request.full_url
                choice = {"index": 0, "prompt_token_ids": ids, "token_ids": [99] if is_prefill else [5, 6],
                          "text": "x" if is_prefill else "xx", "finish_reason": "length"}
                result = {"id": identifier, "choices": [choice], "usage": {"prompt_tokens":len(ids),"completion_tokens":len(choice["token_ids"]),"total_tokens":len(ids)+len(choice["token_ids"]),"prompt_tokens_details":{"cached_tokens":0}}}
                if is_prefill:
                    result["kv_transfer_params"] = {"do_remote_prefill": True, "do_remote_decode": False,
                        "remote_engine_id": "p", "remote_request_id": identifier + "-0-a8477bf8", "remote_host": "127.0.0.1",
                        "remote_port": 5600, "remote_block_ids": [[1, 2]], "remote_num_tokens": len(ids),
                        "tp_size": 1, "pp_size": 1, "transfer_mode": "pull", "remote_prefill_cached_tokens": 0}
                return encoded(result)
        sources = native.installed_sources(DIAG)
        cls = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
        obj = cls(); obj.vllm_config = SimpleNamespace(use_v2_model_runner=True)
        runner = native.actual_runner(obj, cls, self.function(), sources)
        helper_sha, probe_sha = file_sha(HERE / "native_runner_contract.py"), file_sha(probe.__file__)
        id_sha=file_sha(HERE/"native_request_id.py")
        id_sources={"files":{name:"/native/vllm/"+name for name in request_ids.SOURCES},"sha256":request_ids.SOURCES}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prepared = root / "preflight.json"
            rows = [{"id": f"train:{i}", "task": "gsm8k", "split": "train", "token_ids": [11, 22, 33, 44+i]} for i in range(4)]
            write_new(prepared, rows)
            contract = {"complete": True, "ep": 4, "source_commit": client.SOURCE_COMMIT,
                "checkpoint_revision": client.REVISION, "kv_connector": "NixlConnector", "kv_load_failure_policy": "fail",
                "prefix_caching": False, "source_proof_sha256": "source", "checkpoint_proof_sha256": "checkpoint",
                "native_runner_sources": sources, "runner_contract_sha256": helper_sha, "model_path": "/pinned/model",
                "request_id_contract_sha256":id_sha,"request_id_sources":id_sources}
            pc, dc = root / "p.json", root / "d.json"
            write_new(pc, {**contract, "role": "prefill", "arm": "keep6p8"})
            write_new(dc, {**contract, "role": "decode", "arm": "stock"})
            args = SimpleNamespace(prepared=prepared, prepared_sha256=file_sha(prepared), prefill_contract=pc,
                decode_contract=dc, mode="smoke", arm="keep6p8", expected_split="train", concurrency=4,
                max_requests=4, stop_unix=time.time()+60, output=root / "client", launch_id="fixture", served_model="task-model",
                max_tokens=2, timeout=5, prefill_url="http://prefill", decode_url="http://decode")
            with patch("urllib.request.urlopen", side_effect=lambda request, **kw: Response(request)):
                client.run(args)
            records = [read(p) for p in sorted((args.output / "records").glob("*.json"))]
            for role, arm, pids in (("prefill", "keep6p8", range(100,104)), ("decode", "stock", range(200,204))):
                folder = root / (role + "-probe"); folder.mkdir()
                for rank, pid in enumerate(pids):
                    base = {"role": role, "arm": arm, "pid": pid, "ep_rank": rank, "unix_ns": time.time_ns(),
                            "probe_sha256": probe_sha, "runner_contract_sha256": helper_sha,"request_id_contract_sha256":id_sha}
                    keep = 6 if role == "prefill" else None
                    contracts = [{"router_id": j, "keep": keep, "routing_module": "placement_router.vendor.fused_topk_router" if keep else
                        "vllm.model_executor.layers.fused_moe.router.fused_topk_router", "top_k": 8, "experts": 128,
                        "renormalize": True, "scoring_func": "softmax", "eplb": False} for j in range(48)]
                    events = [{**base, "kind": "runner_identity", "native_runner": runner},
                        {**base, "kind": "worker_identity", "native_runner": runner, "source_commit": client.SOURCE_COMMIT,
                         "ep_size": 4, "contracts": contracts, "kv_connector": "NixlConnector", "kv_load_failure_policy": "fail",
                         "prefix_caching": False, "source_sha256": probe.SOURCE, "graphs": "NONE", "seed": 0, "dtype": "torch.bfloat16",
                         "vllm_version": "0.31.0", "model_path": "/pinned/model", "kv_role": "kv_producer" if role=="prefill" else "kv_consumer",
                         "max_model_len": 8192, "max_num_seqs": 256, "max_num_batched_tokens": 16384, "logprobs_mode": "raw_logprobs"}]
                    record = records[rank]; prompt = rows[rank]["token_ids"]
                    request_id = record["producer_native_request_id"] if role=="prefill" else record["consumer_external_request_id"]+"-b8477bf8"
                    transport=record["producer_transport"] if role=="prefill" else record["consumer_transport"]
                    instant=(transport["submitted_unix_ns"]+transport["returned_unix_ns"])//2
                    external=request_ids.external_item_id(record["producer_response_id"] if role=="prefill" else record["consumer_response_id"])
                    proof={"module":"vllm.v1.engine.input_processor","qualname":"InputProcessor.assign_request_id","parameter_names":["request"],
                           "code_filename":id_sources["files"]["v1/engine/input_processor.py"],"source_sha256":request_ids.SOURCES["v1/engine/input_processor.py"],
                           "random_uuid_callable":{"module":"vllm.utils","qualname":"random_uuid","code_filename":id_sources["files"]["utils/__init__.py"],"source_sha256":request_ids.SOURCES["utils/__init__.py"]}}
                    events.append({**base,"kind":"request_id_assigned","external_request_id":external,"internal_request_id":request_id,
                        "unix_ns":instant,"assignment_started_unix_ns":instant,"assignment_finished_unix_ns":instant,
                        "native_assign_callable":proof,"request_id_sources":id_sources,**request_ids.prompt_identity(prompt)})
                    request_contract = probe.native_prompt(SimpleNamespace(prompt_token_ids=prompt, prefill_token_ids=prompt, num_computed_tokens=0 if role=="prefill" else 3))
                    events.append({**base, "kind": "worker_step", "scheduled_tokens": {request_id: 4 if role=="prefill" else 1},
                        "total_scheduled_tokens": 4 if role=="prefill" else 1, "request_contracts": {request_id: request_contract},
                        "router_calls": 48, "distinct_router_objects": 48, "dummy_run": False, "is_profile": False,
                        "has_real_scheduled_requests": True})
                    if role=="decode":
                        events.append({**events[-1], "request_contracts": {request_id: request_contract}})
                        events.extend([{**base, "kind": "scheduler_allocated", "request_id": request_id,
                            "external_tokens": 3, "params": record["transfer_params"]},
                            {**base, "kind": "worker_transfer_finished", "done_receiving": [request_id], "done_sending": [], "failed_receiving": []},
                            {**base, "kind": "worker_transfer_stats", "data": {"bytes_transferred": [4096]}}])
                    (folder / f"pid{pid}.jsonl").write_bytes(b"".join(encoded(event)+b"\n" for event in events))
            audit_args = SimpleNamespace(result=args.output, prepared=prepared, prefill_probe=root/"prefill-probe",
                decode_probe=root/"decode-probe", output=root/"phase-audit.json")
            audit.audit(audit_args)
            self.assertTrue(read(audit_args.output)["decoder_exact_transferred_prefix_then_native_last_prompt_token"])
            bad = root/"decode-probe/pid200.jsonl"
            events = [json.loads(line) for line in bad.read_bytes().splitlines()]
            next(e for e in events if e["kind"]=="worker_step")["request_contracts"][records[0]["consumer_external_request_id"]+"-b8477bf8"]["initial_computed_tokens"] = 0
            bad.write_bytes(b"".join(encoded(event)+b"\n" for event in events))
            audit_args.output = root/"bad.json"
            with self.assertRaises(RuntimeError): audit.audit(audit_args)


if __name__ == "__main__": unittest.main()
