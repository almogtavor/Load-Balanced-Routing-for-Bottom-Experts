"""CPU-only checks for corpus identity, historical targets, and fail-closed run gates."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import paper_quality as quality


def args(root):
    return SimpleNamespace(output=root, model=root / "model", reference_tokenizer=root / "model",
        checkpoint_proof=root / "checkpoint.json", original_provenance=root / "original.json",
        stream_file=root / "wikitext.npy", corpus_mode="wikitext-substitution",
        historical_stream_sha256=None, logical_ep=4, cuda_visible_devices="0,1,2,3",
        tag="test", python="python3", mode="run")


def valid_result(prepared):
    return {"complete": True, "validation": {"passed": True},
        "arms": {a: {"per_window": [1.] * 32, "changed_frac": 0., **dict(zip(("arm", "keep", "load_penalty"), spec))}
                 for a, spec in quality.ARM_SPEC.items()},
        "n_windows": 32, "seq_len": 2048, "scored_tokens_per_arm": 65504,
        "logical_ep_size": 4, "visible_cuda_devices": 4, "n_moe_layers": 48, "n_routers_hooked": 48,
        "model": prepared["checkpoint"]["model_path"],
        "data": {**{k: prepared["tokens"][k] for k in ("input_token_sha256", "scored_target_token_sha256")},
                 "bos_token_id": None, "model_input_protocol": "raw_windows"}}


class ProtocolTests(unittest.TestCase):
    def test_scope_target_budget_and_serialization(self):
        protocol = quality.protocol(args(Path("/tmp/test")))
        self.assertEqual(protocol["scored_targets_per_arm"], 65504)
        self.assertEqual(protocol["physical_gpu_count_expected"], 4)
        self.assertFalse(protocol["distributed_expert_parallel_execution"])
        self.assertFalse(protocol["exact_table2_reproduction_claim"])
        self.assertEqual(json.loads(json.dumps(protocol)), protocol)

    def test_historical_missing_sha_and_wikitext_relabel_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); a.stream_file.write_bytes(b"known-Wikitext")
            original = {"stream": {"source_sha256": quality.digest(a.stream_file)}}
            self.assertEqual(quality.corpus_proof(a, original)["mode"], "wikitext-substitution")
            a.corpus_mode = "historical"
            with self.assertRaisesRegex(RuntimeError, "independently recovered"):
                quality.corpus_proof(a, original)
            a.historical_stream_sha256 = quality.digest(a.stream_file)
            with self.assertRaisesRegex(RuntimeError, "cannot be relabeled"):
                quality.corpus_proof(a, original)

    def test_unknown_wikitext_bytes_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); a.stream_file.write_bytes(b"new-corpus")
            with self.assertRaisesRegex(RuntimeError, "captured original"):
                quality.corpus_proof(a, {"stream": {"source_sha256": "0" * 64}})

    def test_clean_environment_and_complete_arm_grid(self):
        a = args(Path("/tmp/test"))
        prepared = {"tokens": {"path": "/tmp/prepared.npy"}}
        with mock.patch.dict(os.environ, {"PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4_direct",
                "PLACEMENT_PROBE": "old_probe", "PAPER_PROBE_DIR": "/old", "BOS_TOKEN_ID": "1",
                "TEXT_FILE": "/wrong", "PENS": "3", "KERNEL_ARMS": "1m0.5p1"}):
            env = quality.environment(a, prepared)
        for name in ("PLACEMENT_EXPERIMENTAL_SELECTOR", "PLACEMENT_PROBE", "PAPER_PROBE_DIR", "BOS_TOKEN_ID", "TEXT_FILE", "KERNEL_ARMS"):
            self.assertNotIn(name, env)
        self.assertEqual((env["KEEPS"], env["PENS"], env["PRODUCTION_NOOP"]), ("6,4", "0,1", "1"))
        self.assertEqual((env["NWIN"], env["SEQ_LEN"], env["BATCH"]), ("32", "2048", "4"))

    def test_result_target_hash_and_context_guards(self):
        prepared = {"protocol": quality.protocol(args(Path("/tmp/test"))),
            "checkpoint": {"model_path": "/tmp/test/model"},
            "tokens": {"input_token_sha256": "inputs", "scored_target_token_sha256": "targets"}}
        quality.validate_result(valid_result(prepared), prepared)
        for field, value in (("scored_target_token_sha256", "shifted"), ("bos_token_id", 1)):
            result = valid_result(prepared); result["data"][field] = value
            with self.assertRaisesRegex(RuntimeError, "Scored tokens|BOS/context"):
                quality.validate_result(result, prepared)
        result = valid_result(prepared); result["arms"].pop("production_noop")
        with self.assertRaisesRegex(RuntimeError, "arm set"):
            quality.validate_result(result, prepared)
        result = valid_result(prepared); result["arms"]["placement_6p1"]["load_penalty"] = 0.
        with self.assertRaisesRegex(RuntimeError, "routing parameters"):
            quality.validate_result(result, prepared)

    def test_source_guard_refuses_modified_kernel(self):
        with mock.patch.object(quality, "digest", return_value="modified"):
            with self.assertRaisesRegex(RuntimeError, "pristine"):
                quality.source_identity()

    def test_constant_production_noop_bias_is_flagged(self):
        prepared = {"protocol": quality.protocol(args(Path("/tmp/test"))),
            "checkpoint": {"model_path": "/tmp/test/model"}, "corpus": {}, "source_sha256": {},
            "tokens": {"input_token_sha256": "inputs", "scored_target_token_sha256": "targets"}}
        result = valid_result(prepared)
        self.assertFalse(quality.summarize(result, prepared)["quality_interpretation_provisional"])
        result["arms"]["production_noop"]["per_window"] = [1.01] * 32
        summary = quality.summarize(result, prepared)
        self.assertTrue(summary["quality_interpretation_provisional"])
        self.assertTrue(summary["production_noop_vs_repeat"]["zero_se_nonzero_delta"])

    def test_run_rejects_changed_preparation_before_subprocess(self):
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); a.original_provenance.write_text("{}")
            quality.save(a.output / "quality-preparation.json", {"protocol": {"changed": True}})
            with mock.patch.object(quality, "arguments", return_value=a), \
                    mock.patch.object(quality, "corpus_proof", return_value={}), \
                    mock.patch.object(quality, "checkpoint_proof", return_value={}), \
                    mock.patch.object(quality, "source_identity", return_value={}), \
                    mock.patch.object(quality.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "Frozen quality preparation"):
                    quality.main([])
                run.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is needed only for token-preparation CPU test")
    def test_historical_decode_prefix_33window_requirement_and_targets(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            a = args(Path(folder)); raw = np.arange(3 * quality.NEED + 100, dtype=np.int32)
            np.save(a.stream_file, raw)
            reference, tokenizer = mock.Mock(), mock.Mock()
            reference.decode.return_value = "decoded"
            tokenizer.return_value = SimpleNamespace(input_ids=list(range(quality.NEED)))
            fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(side_effect=[reference, tokenizer])))
            with mock.patch.dict("sys.modules", {"transformers": fake}):
                result = quality.prepare_tokens(a)
            decoded = reference.decode.call_args
            self.assertEqual(len(decoded.args[0]), 202752)
            self.assertEqual(decoded.kwargs, {"skip_special_tokens": True})
            tokenizer.assert_called_once_with("decoded", add_special_tokens=False)
            tokens = np.load(result["path"])
            self.assertEqual(len(tokens), 65536)
            self.assertEqual(int(tokens[-1]), 65535)
            self.assertEqual(result["historical_required_tokens"], 67584)
            self.assertEqual(result["scored_target_token_sha256"], quality.hashlib.sha256(tokens.reshape(32, 2048)[:, 1:].astype("<i8").tobytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
