"""CPU-only scoring/protocol tests; no model, network, GPU or dataset dependency."""
import copy
from collections import UserDict
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import task_bench as b
import task_bench_probe as probe


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(x) for x in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and return_dict is False
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")


def hs_row(choice=0, gold=0):
    return {"id": f"hellaswag:validation:0:{choice}", "task": "hellaswag", "item_index": 0,
            "choice": choice, "gold": gold, "token_ids": [10, 20, 30, 40],
            "continuation_start": 2, "continuation_tokens": 2, "normalization_chars": 10}


def hs_choice(index=0):
    return {"index": index, "prompt_token_ids": [10, 20, 30, 40], "token_ids": [50],
            "finish_reason": "length", "text": "unused",
            "prompt_logprobs": [None, {"20": {"logprob": -100}},
                                {"999": {"logprob": -.01}, "30": {"logprob": -2}},
                                {"40": {"logprob": -3}}]}


class TaskBenchTests(unittest.TestCase):
    def test_gsm_gold_and_filters(self):
        self.assertEqual(b.gsm_gold("work\n#### 1,200"), "1200")
        self.assertEqual(b.gsm_answers("Some work 12. The answer is -1,234.50.")["strict"], "-1234.50")
        self.assertEqual(b.gsm_answers("Compute 4 then 7; final $1,200.")["flexible"], "1200")
        self.assertIsNone(b.gsm_answers("I do not know")["flexible"])
        self.assertEqual(b.gsm_answers("The answer is 4. The answer is 5.")["strict"], "4")

    def test_prompt_eight_ordered_examples_single_chat_turn(self):
        samples = [{"question": f"ex{i}", "target": f"ans{i}"} for i in range(8)]
        ids = b.gsm_prompt("held-out", samples, CharacterTokenizer())
        text = "".join(chr(x) for x in ids)
        self.assertTrue(text.startswith("<user>Q: ex0\nA: ans0\n\nQ: ex1"))
        self.assertTrue(text.endswith("Q: held-out\nA:<assistant>"))
        with self.assertRaises(RuntimeError):
            b.gsm_prompt("held-out", samples[:7], CharacterTokenizer())

    def test_transformers5_mapping_default_explicit_list_preserves_ids(self):
        samples = [{"question": f"ex{i}", "target": f"ans{i}"} for i in range(8)]
        expected = b.gsm_prompt("held-out", samples, CharacterTokenizer())
        for mapping in (dict, UserDict):
            class MappingTokenizer(CharacterTokenizer):
                def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_dict=True):
                    ids = super().apply_chat_template(messages, tokenize, add_generation_prompt, return_dict=False)
                    return mapping(input_ids=ids, attention_mask=[1] * len(ids)) if return_dict else ids
            tokenizer = MappingTokenizer()
            ids = b.gsm_prompt("held-out", samples, tokenizer)
            text = "\n\n".join("Q: " + s["question"] + "\nA: " + s["target"] for s in samples)
            default = tokenizer.apply_chat_template([{"role": "user", "content": text + "\n\nQ: held-out\nA:"}],
                                                    tokenize=True, add_generation_prompt=True)
            self.assertEqual(ids, default["input_ids"])
            self.assertEqual(ids, expected)
        class NestedTokenizer(CharacterTokenizer):
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [expected]}
        with self.assertRaises(RuntimeError):
            b.gsm_prompt("held-out", samples, NestedTokenizer())

    def test_hellaswag_canonical_preprocessing(self):
        doc = {"activity_label": "Act", "ctx_a": "A [title] thing", "ctx_b": "mIXed CASE.",
               "endings": [" [annotation] He  goes. "] * 4, "label": "1"}
        context, endings = b.hs_document(doc)
        self.assertEqual(context, "Act: A. thing Mixed case.")
        self.assertEqual(endings[0], " He goes.")
        rows = b.make_hs([doc], CharacterTokenizer())
        self.assertEqual([x["choice"] for x in rows], [0, 1, 2, 3])
        self.assertEqual(rows[0]["normalization_chars"], len(" He goes."))
        self.assertEqual(rows[0]["continuation_tokens"], len("  He goes."))

    def test_hellaswag_ind_can_repeat_across_sources(self):
        rows = [{"ind": 0, "source_id": "source-a"}, {"ind": 0, "source_id": "source-b"}]
        b.validate_hs_identity(rows)
        with self.assertRaises(RuntimeError):
            b.validate_hs_identity(rows + [rows[0]])

    def test_joint_tokenization_moves_trailing_whitespace(self):
        ids, start = b.encode_pair(CharacterTokenizer(), "abc ", "def")
        self.assertEqual(ids, [ord(x) for x in "abc def"])
        self.assertEqual(start, 3)
        class BoundaryMerge(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [8, 9] if text == "a" else [88, 99]
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            b.encode_pair(BoundaryMerge(), "a", "b")
        with self.assertRaises(RuntimeError):
            b.encode_pair(CharacterTokenizer(), "", "a")

    def test_actual_token_likelihood_only_continuation(self):
        scored = b.score_choice(hs_row(), hs_choice())
        self.assertEqual(scored["token_logprobs"], [-2., -3.])
        self.assertEqual(scored["loglikelihood"], -5.)
        self.assertEqual(scored["normalized_loglikelihood"], -.5)
        # The highest-probability alternative is deliberately first in the map.
        self.assertNotEqual(scored["loglikelihood"], -.01 - 3)

    def test_likelihood_missing_nonfinite_and_wrong_id_fail(self):
        for mutate in (lambda c: c["prompt_logprobs"].__setitem__(2, None),
                       lambda c: c["prompt_logprobs"][2].pop("30"),
                       lambda c: c["prompt_logprobs"][2]["30"].update(logprob=float("nan")),
                       lambda c: c["prompt_token_ids"].__setitem__(0, 42)):
            choice = hs_choice()
            mutate(choice)
            with self.assertRaises(RuntimeError):
                b.score_choice(hs_row(), choice)

    def test_unparsed_and_truncated_gsm_are_retained(self):
        row = {"id": "gsm8k:test:0", "task": "gsm8k", "item_index": 0, "gold": "3", "token_ids": [1, 2]}
        result = b.score_choice(row, {"prompt_token_ids": [1, 2], "token_ids": [3], "text": "uncertain",
                                      "finish_reason": "length"})
        self.assertFalse(result["correct"])
        self.assertTrue(result["truncated"])
        self.assertTrue(result["parse_failed"])

    def test_response_choice_reordering_and_usage(self):
        rows = [hs_row(0), hs_row(1)]
        response = {"choices": [hs_choice(1), hs_choice(0)], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}
        self.assertEqual([r["choice"] for r in b.validate_response(rows, response)], [0, 1])
        response["choices"][1]["index"] = 1
        with self.assertRaises(RuntimeError):
            b.validate_response(rows, response)

    def test_request_natural_eos_fixed_generation_and_logprobs(self):
        hs = b.request_body([hs_row()], "m", 2048, "id")
        self.assertEqual(hs["prompt"], [[10, 20, 30, 40]])
        self.assertEqual(hs["prompt_logprobs"], 0)
        self.assertFalse(hs["ignore_eos"])
        self.assertFalse(hs["add_special_tokens"])
        self.assertEqual(hs["max_tokens"], 1)
        gs = b.request_body([{"task": "gsm8k", "token_ids": [1]}], "m", 2048, "id")
        self.assertEqual(gs["stop"], b.STOPS)
        self.assertEqual(gs["max_tokens"], 2048)
        self.assertNotIn("prompt_logprobs", gs)

    def test_cohort_no_task_crossing_and_all_hs_choices_together(self):
        rows = [{"task": "gsm8k", "id": i} for i in range(259)]
        rows += [{"task": "hellaswag", "item_index": i, "choice": j} for i in range(65) for j in range(4)]
        groups = list(b.cohorts(rows))
        self.assertEqual([len(x) for x in groups], [256, 3, 256, 4])
        self.assertEqual([r["choice"] for r in groups[2]], [0, 1, 2, 3] * 64)

    def test_raw_and_character_normalized_choice_can_differ(self):
        rows = []
        for c, ll, length in ((0, -2, 1), (1, -3, 10), (2, -9, 1), (3, -10, 1)):
            rows.append({"task": "hellaswag", "item_index": 0, "choice": c, "gold": 1,
                         "loglikelihood": ll, "normalized_loglikelihood": ll / length})
        item = b.item_scores(rows)["hellaswag"][0]
        self.assertEqual(item["prediction"], 0)
        self.assertEqual(item["normalized_prediction"], 1)
        self.assertTrue(item["normalized_correct"])
        with self.assertRaises(RuntimeError):
            b.item_scores(rows[:3])

    def test_paired_exact_and_standard_error(self):
        self.assertEqual(b.paired([False] * 4, [True] * 4)["mcnemar_exact_two_sided_p"], .125)
        result = b.paired([0, 1, 0, 1], [1, 1, 0, 0])
        self.assertEqual(result["gains"], 1)
        self.assertEqual(result["losses"], 1)
        self.assertGreater(result["paired_se"], 0)
        exact = b.paired([0, 1], [0, 1])
        self.assertEqual(exact["paired_se"], 0)
        self.assertEqual(exact["mcnemar_exact_two_sided_p"], 1)
        # Extreme differences should not overflow comb(10000, 5000).
        self.assertLessEqual(b.paired([0] * 10000, [1] * 10000)["mcnemar_exact_two_sided_p"], 1)

    def test_wilson_boundaries_and_known_value(self):
        value = b.wilson([True, False])
        self.assertEqual(value["accuracy"], .5)
        self.assertLess(value["ci95_wilson"][0], .1)
        self.assertGreater(value["ci95_wilson"][1], .9)
        self.assertGreaterEqual(b.wilson([False] * 10)["ci95_wilson"][0], -1e-16)

    def test_exclusive_output_creation(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "new"
            b.own_directory(target)
            with self.assertRaises(FileExistsError):
                b.own_directory(target)

    def test_response_saved_before_invalid_likelihood_rejected(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"choices":[]}'
        with tempfile.TemporaryDirectory() as d, patch.object(b.urllib.request, "urlopen", return_value=Response()):
            target = Path(d) / "cohort00000"
            with self.assertRaises(RuntimeError):
                b.http_cohort("http://localhost", [hs_row()], "m", target, 2048, 10, "fixed")
            self.assertEqual(target.with_suffix(".response.json").read_bytes(), b'{"choices":[]}')
            self.assertFalse(b.read(target.with_suffix(".error.json"))["complete"])

    def test_public_source_fingerprint_and_deployment_schema(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proof = {"source_commit": b.COMMIT, "source_root": str(root),
                     "installed_package": {"installed_source_sha256": probe.SOURCE}}
            path = root / "proof.json"
            b.atomic(path, proof)
            def fake_digest(p):
                name = str(Path(p).relative_to(root))
                return probe.SOURCE.get(name, "proof-sha")
            with patch.object(b, "digest", side_effect=fake_digest):
                identity = b.source_identity(path, root)
            self.assertEqual(identity["files"], probe.SOURCE)
            proof["installed_package"]["installed_source_sha256"] = {**probe.SOURCE, "src/placement_router/kernel.py": "bad"}
            b.atomic(path, proof)
            with self.assertRaises(RuntimeError):
                b.source_identity(path, root)

    def test_checkpoint_requires_exact_verified_shards_and_pinned_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            files = []
            for i in range(16):
                name = f"model-{i:05d}.safetensors"
                (root / name).write_bytes(b"test")
                files.append({"filename": name, "bytes": 4, "sha256": "root-verified"})
            proof = {"canonical_identity_verified": True, "model_id": b.MODEL_ID, "revision": b.REVISION,
                     "local_model_path": str(root), "canonical_config_sha256": b.CONFIG_SHA256,
                     "safetensors": files, "weights_verified": files}
            path = root / "proof.json"
            b.atomic(path, proof)
            with patch.object(b, "digest", return_value=b.CONFIG_SHA256):
                self.assertEqual(len(b.checkpoint_identity(path, root)["file_stat_signatures"]), 16)
                proof["weights_verified"] = [{**x, "sha256": "different"} for x in files]
                b.atomic(path, proof)
                with self.assertRaises(RuntimeError):
                    b.checkpoint_identity(path, root)
                proof["weights_verified"] = files
                proof["canonical_config_sha256"] = "unverified-config"
                b.atomic(path, proof)
                with self.assertRaises(RuntimeError):
                    b.checkpoint_identity(path, root)

    def test_probe_contract_proves_native_vs_production(self):
        class Router:
            top_k, global_num_experts, renormalize, scoring_func, eplb_state = 8, 128, True, "softmax", None
            def _compute_routing(self): pass
        r = Router()
        Router._compute_routing.__module__ = "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
        self.assertTrue(probe.router_contract(r, None)["expected"])
        self.assertFalse(probe.router_contract(r, 6)["expected"])
        r._keep_for = 6
        Router._compute_routing.__module__ = "placement_router.vendor.fused_topk_router"
        self.assertTrue(probe.router_contract(r, 6)["expected"])
        r.renormalize = False
        self.assertFalse(probe.router_contract(r, 6)["expected"])

    def test_all_rank_runtime_evidence_and_checkpoint_binding(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            source = {"files": probe.SOURCE, "proof_sha256": "deployed"}
            model = {"model_path": str(folder / "model")}
            template = {"complete": True, "arm": "keep6p8", "ep_world_size": 4, "placement_keep": 6,
                        "load_penalty": 8., "load_tile": 16, "source_sha256": probe.SOURCE,
                        "probe_sha256": b.digest(Path(b.__file__).with_name("task_bench_probe.py")),
                        "deployment_proof_sha256": "deployed", "model_path": model["model_path"],
                        "registered_layers": 48, "observed_router_objects": 48, "physical_dp": 4,
                        "tp": 1, "pp": 1, "backend": "deepep_high_throughput", "graphs": "NONE",
                        "dtype": "torch.bfloat16", "hidden_size": 2048, "prefix_caching": False,
                        "ubatching": False, "enable_expert_parallel": True, "expert_placement_strategy": "linear",
                        "max_model_len": 8192, "max_num_seqs": 256, "max_num_batched_tokens": 16384,
                        "seed": 0, "all_routers_expected": True}
            for rank in range(4):
                b.atomic(folder / f"rank{rank}_pid{rank+100}.json", {**template, "ep_rank": rank, "pid": rank + 100})
            self.assertEqual(len(b.rank_evidence(folder, "keep6p8", 4, source, model)), 4)
            path = folder / "rank0_pid100.json"
            for key, wrong in (("load_penalty", 1.), ("model_path", str(folder / "wrong-model")),
                               ("deployment_proof_sha256", "wrong-proof"), ("expert_placement_strategy", "round_robin")):
                b.atomic(path, {**template, "ep_rank": 0, "pid": 100, key: wrong})
                with self.assertRaises(RuntimeError):
                    b.rank_evidence(folder, "keep6p8", 4, source, model)
            path.unlink()
            with self.assertRaises(RuntimeError):
                b.rank_evidence(folder, "keep6p8", 4, source, model)

    def test_analysis_rejects_partial_counts(self):
        with self.assertRaises(RuntimeError):
            b.summarize_records([])
        self.assertEqual(b.protocol()["gsm8k"]["max_output_tokens"], 2048)
        self.assertEqual(set(b.protocol()["arms"]), set(b.ARMS))

    def test_analysis_six_files_with_same_parent_basename(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = []
            for a in b.ARMS:
                path = root / a / "score" / "result.json"
                b.atomic(path, {"complete": True, "mode": "full", "arm": a, "ep": 4,
                                "preparation_sha256": "same", "records": []})
                paths.append(path)
            gsm = [{"correct": True, "answers": {"flexible": "1"}}] * 2
            hs = [{"correct": True, "normalized_correct": True, "prediction": 0, "normalized_prediction": 0}] * 2
            with patch.object(b, "item_scores", return_value={"gsm8k": gsm, "hellaswag": hs}), \
                 patch.object(b, "summarize_records", return_value={"test_summary": True}):
                b.analyze(SimpleNamespace(results=paths, output=root / "analysis.json"))
            self.assertEqual(set(b.read(root / "analysis.json")["summaries"]), set(b.ARMS))


if __name__ == "__main__":
    unittest.main()
