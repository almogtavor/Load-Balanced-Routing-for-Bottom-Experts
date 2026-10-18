"""Post-training held-out subset, serial-prefix and raw-rescoring CPU guards."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import common_v3 as common
import heldout_causal_v2 as heldout


class HeldoutTests(unittest.TestCase):
    def make_inputs(self, root, patch_digest=True):
        inputs = root / "all-validation.json"
        rows = [{"task": "hellaswag", "split": "validation", "id": f"hellaswag:validation:{i}:{c}",
                 "item_index": i, "choice": c, "gold": i % 4, "token_ids": [33, 44 + c],
                 "continuation_start": 1, "continuation_tokens": 1, "normalization_chars": 1}
                for i in range(10042) for c in range(4)]
        common.write_new(inputs, rows)
        if patch_digest:
            # Synthetic fixtures cannot have the canonical dataset's SHA. Only
            # this test process substitutes its expected digest; production
            # source and helper file hashes remain untouched.
            patcher = patch.object(heldout, "CANONICAL_REQUESTS_SHA", common.file_sha(inputs))
            patcher.start()
            self.addCleanup(patcher.stop)
        analyses = []
        for arm in ("stock", "keep6p8", "keep4p8"):
            for budget in (1024, 16384):
                path = root / f"training-{arm}-{budget}.json"
                common.write_new(path, {"complete": True, "training_only": True, "arm": arm,
                    "chunking_label": f"budget{budget}", "source_commit": common.SOURCE_COMMIT,
                    "checkpoint_revision": common.REVISION, "source_proof_sha256": "source", "checkpoint_proof_sha256": "checkpoint"})
                analyses.append(path)
        return SimpleNamespace(full_requests=inputs, full_requests_sha256=common.file_sha(inputs),
            training_analyses=analyses, source_proof_sha256="source", checkpoint_proof_sha256="checkpoint", output=root / "prepared")

    def test_preparation_rejects_missing_training_cell_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_inputs(Path(directory))
            args.training_analyses = args.training_analyses[:-1]
            with self.assertRaises(RuntimeError):
                heldout.prepare(args)
            self.assertFalse(args.output.exists())

    def test_actual_canonical_digest_is_required_without_fixture_override(self):
        self.assertEqual(heldout.CANONICAL_REQUESTS_SHA, "6c3cdfb53702f5b1e07f8d38a9770b5da45b5d419b5ad81d5b9e81908c0cf63b")
        with tempfile.TemporaryDirectory() as directory:
            args = self.make_inputs(Path(directory), patch_digest=False)
            with self.assertRaisesRegex(RuntimeError, "Canonical full input"):
                heldout.prepare(args)
            self.assertFalse(args.output.exists())

    def native_contract(self, root):
        source = Path(heldout.__file__).resolve().parent
        probe_sha = common.file_sha(source / "causal_worker_probe.py")
        paths = {}
        for rank in range(4):
            path = root / f"native-rank{rank}.json"
            common.write_new(path, {"complete": True, "arm": "stock", "all_routers_expected": True,
                "source_commit": common.SOURCE_COMMIT, "deployment_proof_sha256": "source", "probe_sha256": probe_sha,
                "observed_router_objects": 48, "registered_layers": 48, "ep_rank": rank, "pid": 100 + rank})
            paths[str(path)] = common.file_sha(path)
        return {"complete": True, "arm": "stock", "chunking_label": "budget16384", "prefix_caching": False,
                "source_commit": common.SOURCE_COMMIT, "checkpoint_revision": common.REVISION,
                "checkpoint_proof_sha256": "checkpoint", "source_proof_sha256": "source",
                "builder_sha256": common.file_sha(source / "causal_contract.py"), "probe_sha256": probe_sha,
                "probe_files_sha256": paths}

    def test_native_worker_proof_bytes_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = self.native_contract(root)
            heldout.native_contract_gate(contract)
            path = Path(next(iter(contract["probe_files_sha256"])))
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(RuntimeError, "proof bytes changed"):
                heldout.native_contract_gate(contract)

    def test_full_serial_run_and_independent_raw_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prep = self.make_inputs(root)
            heldout.prepare(prep)
            protocol = common.read(prep.output / "protocol.json")
            self.assertEqual(protocol["selected_item_indices"], heldout.selected_indices())
            self.assertEqual(protocol["prefix_requests"], 1024)
            contract = root / "contract.json"
            common.write_new(contract, self.native_contract(root))
            state = {"active": 0, "max_active": 0, "prefix_bodies": []}
            class Response:
                status, headers = 200, {}
                def __init__(self, request): self.body = json.loads(request.data)
                def __enter__(self):
                    state["active"] += 1
                    state["max_active"] = max(state["max_active"], state["active"])
                    return self
                def __exit__(self, *args):
                    state["active"] -= 1
                    return False
                def read(self):
                    body = self.body
                    prompts = body["prompt"] if isinstance(body["prompt"][0], list) else [body["prompt"]]
                    values = []
                    for i, ids in enumerate(prompts):
                        choice = {"index": i, "prompt_token_ids": ids, "token_ids": [55], "text": "unit", "finish_reason": "length"}
                        if "prompt_logprobs" in body:
                            choice["prompt_logprobs"] = [None] + [{str(t): {"logprob": -t / 10}} for t in ids[1:]]
                        else:
                            state["prefix_bodies"].append(body)
                            choice["logprobs"] = {"top_logprobs": [{f"token_id:{t}": -t / 10 for t in body["logprob_token_ids"]}]}
                        values.append(choice)
                    return common.encoded({"choices": values})
            run = SimpleNamespace(prepared=prep.output, server_contract=contract, arm="stock", url="http://unused",
                output=root / "run", exclusive_server_ack="no-other-live-evaluation-requests", max_requests=1028,
                stop_unix=time.time() + 300, launch_id="heldout-cpu", served_model="unit", timeout=5)
            with patch("urllib.request.urlopen", side_effect=lambda request, **kwargs: Response(request)):
                heldout.run(run)
            self.assertEqual(state["max_active"], 1)
            self.assertEqual(len(state["prefix_bodies"]), 1024)
            self.assertTrue(all(b["prompt"] == [33] and b["ignore_eos"] is True
                and "allowed_token_ids" not in b and "logit_bias" not in b for b in state["prefix_bodies"]))
            analysis = root / "analysis.json"
            heldout.analyze(SimpleNamespace(prepared=prep.output, result=run.output, output=analysis))
            value = common.read(analysis)
            self.assertEqual(value["n"], 256)
            expected = sum(i % 4 == 0 for i in heldout.selected_indices())
            self.assertEqual(value["correct"], dict.fromkeys(("teacher_raw", "teacher_norm", "prefix_raw", "prefix_norm"), expected))
            # Even if a corrupted raw file and its inventory hash are both
            # changed, exact registered wire reconstruction rejects the target
            # injected into the model input.
            request = run.output / "raw" / "000005-prefix.request.json"
            original_request = request.read_bytes()
            original_result = (run.output / "result.json").read_bytes()
            envelope = common.read(request)
            envelope["body"]["prompt"].append(44)
            envelope["sha256"] = common.sha(common.encoded(envelope["body"]))
            request.write_bytes(common.encoded(envelope) + b"\n")
            result_path = run.output / "result.json"
            result = common.read(result_path)
            result["raw_files_sha256"][str(request.relative_to(run.output))] = common.file_sha(request)
            result_path.write_bytes(common.encoded(result) + b"\n")
            with self.assertRaises(RuntimeError):
                heldout.analyze(SimpleNamespace(prepared=prep.output, result=run.output, output=root / "bad-analysis.json"))
            request.write_bytes(original_request)
            result_path.write_bytes(original_result)
            transport_path = request.with_name("000005-prefix.transport.json")
            transport = common.read(transport_path)
            previous = common.read(request.with_name("000004-teacher.transport.json"))
            transport["submitted_monotonic_ns"] = previous["returned_monotonic_ns"] - 1
            envelope = common.read(request)
            envelope["submitted_monotonic_ns"] = transport["submitted_monotonic_ns"]
            request.write_bytes(common.encoded(envelope) + b"\n")
            transport_path.write_bytes(common.encoded(transport) + b"\n")
            result = common.read(result_path)
            for p in (request, transport_path):
                result["raw_files_sha256"][str(p.relative_to(run.output))] = common.file_sha(p)
            result_path.write_bytes(common.encoded(result) + b"\n")
            with self.assertRaisesRegex(RuntimeError, "intervals overlap"):
                heldout.analyze(SimpleNamespace(prepared=prep.output, result=run.output, output=root / "overlap-analysis.json"))


if __name__ == "__main__":
    unittest.main()
