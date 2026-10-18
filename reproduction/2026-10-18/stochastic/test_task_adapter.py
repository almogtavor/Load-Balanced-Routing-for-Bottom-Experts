#!/usr/bin/env python3
"""Adversarial source-contract fixtures; never a substitute for actual workers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import task_adapter as adapter
import make_launch_plan as builder
from stochastic_tail import plugin


class WorkerContractTests(unittest.TestCase):
    def fixture(self, root, mode="weighted_hot", keep=6, price=None):
        native = {name: str(root / "native" / name) for name in plugin.NATIVE_FILES}
        plan = {"parameters": {"mode": mode, "keep": keep, "load_penalty": price, "seed": 0},
                "runtime": {"ep": 4}, "record_calls": True,
                "private_site": str(root / "private"), "public_site": str(root / "public"),
                "private_source_sha256": {"stochastic_tail/plugin.py": "private", "stochastic_tail/native_hooks.py":"hooks", "stochastic_tail/worker_extension.py":"extension"},
                "native_source_files": native, "native_source_sha256": {n: "native" for n in native},
                "gpu_gate": {"path": str(root / "gate.json"), "sha256": "gate"},
                "checkpoint_proof": {"path": "/checkpoint", "sha256": "checkpoint"},
                "model_id": plugin.MODEL_ID, "revision": plugin.REVISION, "model_path": str(root / "model")}
        (root / "gate.json").write_text(json.dumps({"torch_version": "2.13.0+cu130"}))
        module = ("vllm.model_executor.layers.fused_moe.router.fused_topk_router" if mode == "stock"
                  else "placement_router.vendor.fused_topk_router" if mode == "par" else "stochastic_tail.plugin")
        source = ("native" if mode == "stock" else plugin.PUBLIC["placement_router/vendor/fused_topk_router.py"]
                  if mode == "par" else "private")
        private_path = root / "private/stochastic_tail/plugin.py"
        compute_path = (Path(native["vllm/model_executor/layers/fused_moe/router/fused_topk_router.py"])
                        if mode == "stock" else root / "public/placement_router/vendor/fused_topk_router.py"
                        if mode == "par" else private_path)
        compute = {"module": module, "source_sha256": source, "code_filename": str(compute_path)}
        wrapper = {"source_sha256": "private", "code_filename": str(private_path)}
        native_v2=native["vllm/v1/worker/gpu/model_runner.py"]
        hook_path=str(root/"private/stochastic_tail/native_hooks.py")
        runner={"module":"vllm.v1.worker.gpu.model_runner","qualname":"GPUModelRunner","source_path":native_v2,
                "source_sha256":"native","use_v2_model_runner":True,
                "native_methods":{n:{"code_filename":native_v2,"source_sha256":"native"} for n in ("execute_model","_dummy_run","shutdown")},
                "actual_patched_methods":{n:{"code_filename":hook_path,"source_sha256":"hooks"} for n in ("execute_model","_dummy_run","shutdown")}}
        values = []
        for rank in range(4):
            values.append({"kind": "worker_identity", "complete": True, "plan_sha256": "plan", "pid": 100 + rank,
                "ep_rank": rank, "observer_revision":4,"actual_runner":runner,"parameters": plan["parameters"], "runtime": plan["runtime"],
                "private_source_sha256": plan["private_source_sha256"], "public_source_sha256": plugin.PUBLIC,
                "native_source_sha256": plan["native_source_sha256"], "source_lineage_commit": plugin.PUBLIC_COMMIT,
                "private_module_root": str(private_path.parent),
                "public_module_paths": {name: str(root / "public" / name) for name in plugin.PUBLIC},
                "actual_compute_callable": compute, "expected_compute_module": module,
                "actual_route_callable": wrapper, "actual_constructor_callable": wrapper,
                "public_par_patched": mode == "par", "experimental_overlay_active": True, "record_calls": True,
                "gpu_gate": plan["gpu_gate"], "checkpoint_proof": plan["checkpoint_proof"],
                "model_id": plugin.MODEL_ID, "revision": plugin.REVISION, "model_path": plan["model_path"],
                "initial_rng_counters": {str(i): 0 for i in range(48)},
                "native_layers": [{"layer_name": f"model.layers.{i}.mlp.experts", "layer_index": i,
                    "router_id": rank * 1000 + i, "top_k": 8, "experts": 128, "renormalize": True,
                    "scoring_func": "softmax", "eplb": False, "compute_callable": compute,
                    "actual_public_keep": keep if mode == "par" else None} for i in range(48)],
                "public_load_penalty": price,
                "effective_load_tile": (16 if price else 1) if mode == "par" else None,
                "gpu": "NVIDIA H100 80GB HBM3", "vllm_version": "0.31.0+local", "torch_version": "2.13.0+cu130"})
        return plan, values

    def write(self, root, rows, mutable_tail=False, omit_selected=False):
        ledger = root / "ledger"
        ledger.mkdir(exist_ok=True)
        for row in rows:
            selected={"kind":"native_runner_selected","complete":True,"pid":row["pid"],"ep_rank":row["ep_rank"],
                "plan_sha256":row["plan_sha256"],"parameters":row["parameters"],"actual_runner":row["actual_runner"],
                "actual_worker_module":"vllm.v1.worker.gpu_worker","actual_worker_qualname":"Worker",
                "native_worker_load":{"code_filename":str(root/"native/vllm/v1/worker/gpu_worker.py"),"source_sha256":"native"},
                "actual_worker_load":{"code_filename":str(root/"private/stochastic_tail/native_hooks.py"),"source_sha256":"hooks"}}
            prefix=b"" if omit_selected else json.dumps(selected,sort_keys=True).encode()+b"\n"
            raw = json.dumps(row, sort_keys=True).encode() + b"\n"
            (ledger / f"pid{row['pid']}.jsonl").write_bytes(prefix+raw + (b'{"unfinished":' if mutable_tail else b""))
        return ledger

    def test_all_nine_semantics_accept_real_callable_contract(self):
        for mode, keep, price in [("stock", None, None), ("par", 6, 8.), ("par", 4, 8.),
                                  ("par", 6, 0.), ("par", 4, 0.), ("uniform_hot", 6, None),
                                  ("uniform_hot", 4, None), ("weighted_hot", 6, None), ("weighted_hot", 4, None)]:
            with self.subTest(mode=mode, keep=keep, price=price), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, rows = self.fixture(root, mode, keep, price)
                ledger = self.write(root, rows, mutable_tail=True)
                values, records = adapter.worker_identities(ledger, plan, "plan")
                self.assertEqual(len(values), 4)
                raw = (ledger / "pid100.jsonl").read_bytes().splitlines(keepends=True)[1]
                self.assertEqual(records[0]["identity_record_sha256"], hashlib.sha256(raw).hexdigest())

    def test_public_lineage_cannot_bless_private_or_fake_native_callable(self):
        changes = [lambda r: r["actual_compute_callable"].update(module="placement_router.vendor.fused_topk_router"),
                   lambda r: r["actual_compute_callable"].update(code_filename="/other-copy/stochastic_tail/plugin.py"),
                   lambda r: r.update(private_module_root="/other-copy/stochastic_tail"),
                   lambda r: r["native_source_sha256"].update({"vllm/v1/worker/gpu_model_runner.py": "changed"}),
                   lambda r: r["actual_route_callable"].update(code_filename="/fake-wrapper.py"),
                   lambda r: r.update(torch_version="2.12.0"),
                   lambda r: r.update(vllm_version="0.31.1"),
                   lambda r: r["parameters"].update(seed=1),
                   lambda r: r["initial_rng_counters"].update({"0": 7}),
                   lambda r: r["native_layers"][0].update(layer_name="model.layers.99.mlp.experts")]
        for number, change in enumerate(changes):
            with self.subTest(change=number), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, rows = self.fixture(root)
                rows = copy.deepcopy(rows)  # adversary cannot mutate the expected plan
                change(rows[0])
                with self.assertRaises(RuntimeError):
                    adapter.worker_identities(self.write(root, rows), plan, "plan")

    def test_duplicate_rank_missing_rank_and_wrong_price_rejected(self):
        for kind in ("duplicate_rank", "missing", "wrong_price", "wrong_tile", "private_keep"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plan, rows = self.fixture(root, "par", 6, 0.)
                rows = copy.deepcopy(rows)
                if kind == "duplicate_rank": rows[3]["ep_rank"] = 2
                if kind == "missing": rows = rows[:3]
                if kind == "wrong_price": rows[0]["public_load_penalty"] = 8.
                if kind == "wrong_tile": rows[0]["effective_load_tile"] = 16
                if kind == "private_keep": rows[0]["native_layers"][0]["actual_public_keep"] = 4
                with self.assertRaises(RuntimeError):
                    adapter.worker_identities(self.write(root, rows), plan, "plan")

    def test_incomplete_identity_and_no_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, rows = self.fixture(root)
            rows[0]["complete"] = False
            with self.assertRaises(RuntimeError):
                adapter.worker_identities(self.write(root, rows), plan, "plan")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _ = self.fixture(root)
            self.assertRaises(RuntimeError, adapter.worker_identities, self.write(root, []), plan, "plan")

    def test_actual_runner_callback_required_and_wrong_nativeclass_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,rows=self.fixture(root)
            with self.assertRaises(RuntimeError):adapter.worker_identities(self.write(root,rows,omit_selected=True),plan,"plan")
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);plan,rows=self.fixture(root);rows=copy.deepcopy(rows)
            rows[0]["actual_runner"]["module"]="vllm.v1.worker.gpu_model_runner"
            with self.assertRaises(RuntimeError):adapter.worker_identities(self.write(root,rows),plan,"plan")

    def test_adapter_has_no_old_public_activation_or_performance_claim(self):
        # Old full-stage entrypoint would silently reapply public-only validation.
        source = Path(adapter.__file__).read_text()
        self.assertNotIn("task.run_stage(", source)
        self.assertNotIn("task.rank_evidence(", source)
        self.assertIn("not a claim of strict causal model likelihood", source)


class GateProofTests(unittest.TestCase):
    def fixture(self):
        def case(t, k, g, mode, seed, zdtype, idtype, rank, call, layer):
            return dict(T=t, keep=k, ranks=g, mode=mode, seed=seed, logits_dtype=zdtype, indices_dtype=idtype,
                        physical_rng_rank=rank, call=call, layer=layer, weighted_near_tie_tail_rows=0, max_weighted_key_error=0.)
        modes = ("uniform_hot", "weighted_hot", "uniform_range")
        cases = [case(17, k, g, m, seed, z, "torch.int64", g-1, 13, 47)
                 for z in ("torch.float32", "torch.bfloat16", "torch.float16") for g in (4,8)
                 for k in (4,6) for seed in (0,1,2) for m in modes]
        cases += [case(t,4,4,m,0,"torch.float32",i,0,0,0) for t in (1,2,15,16,33,257,2051)
                  for m in modes for i in ("torch.int32", "torch.int64", "torch.uint32")]
        cases += [case(33,k,4,m,0,"torch.float32","torch.int64",0,13,47) for k in (4,6,8) for m in modes]
        return {"schema": "stochastic-tail-kernel-gate-v1", "complete": True, "cuda_required": True,
                "gpu": "NVIDIA H100 80GB HBM3", "private_source_sha256": plugin.private_hashes(),
                "gate_helper_sha256": builder.sha(Path(builder.__file__).with_name("gpu_gate.py")),
                "weighted_key_tolerance": 2e-5, "weighted_tail_reference_regret_tolerance": 4e-5,
                "checks": {n:True for n in ["native_head", "distinct_experts", "dtype_and_sentinel", "original_weights",
                    "ragged_and_padding", "rng_words_and_repeat", "fanout", "ties_and_range"]}, "cases": cases}

    def test_complete_frozen_inventory_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "gate.json"
            value = self.fixture()
            p.write_text(json.dumps(value))
            self.assertEqual(builder.validate_gate(p), value)

    def test_duplicate_case_cannot_replace_missing_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "gate.json"
            value = self.fixture()
            value["cases"][-1] = value["cases"][0]
            p.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError): builder.validate_gate(p)

    def test_old_helper_or_relaxed_tolerance_cannot_bless_gate(self):
        for field, bad in [("gate_helper_sha256", "old"), ("weighted_key_tolerance", 1.), ("cuda_required", False)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                p = Path(folder) / "gate.json"
                value = self.fixture()
                value[field] = bad
                p.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError): builder.validate_gate(p)


if __name__ == "__main__":
    unittest.main(verbosity=2)
