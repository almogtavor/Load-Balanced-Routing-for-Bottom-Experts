#!/usr/bin/env python3
"""CPU invariants and fail-closed provenance; never a substitute for CUDA gate."""
import ctypes
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from stochastic_tail import reference as ref
from stochastic_tail import plugin


def uint32_independent(value):
    value = ctypes.c_uint32(value).value
    value = ctypes.c_uint32((value ^ (value >> 16)) * 0x7FEB352D).value
    value = ctypes.c_uint32((value ^ (value >> 15)) * 0x846CA68B).value
    return ctypes.c_uint32(value ^ (value >> 16)).value


def logits(rows=17):
    rng = random.Random(311)
    return [[rng.gauss(0, 1) for _ in range(128)] for _ in range(rows)]


class SelectorTests(unittest.TestCase):
    def test_unsigned32_overflow_matches_independent_ctypes(self):
        for value in [0, 1, 17, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0x100000001]:
            self.assertEqual(ref.mix32(value), uint32_independent(value))

    def test_uniform_open_endpoints_are_finite_exact_float32(self):
        for word in [0, 1, 511, 512, ref.MASK - 511, ref.MASK]:
            u = ref.uniform_open(word)
            self.assertTrue(0 < u < 1)
            self.assertEqual(u, ref.f32(u))
            self.assertTrue(math.isfinite(ref.gumbel(word)))
        self.assertEqual(ref.uniform_open(0), 2 ** -24)
        self.assertEqual(ref.uniform_open(ref.MASK), 1 - 2 ** -24)

    def test_rng_repeats_and_uses_every_namespace_dimension(self):
        base = [ref.random_word(0, 1, 2, row, expert, physical_rank=0) for row in range(3) for expert in range(128)]
        self.assertEqual(base, [ref.random_word(0, 1, 2, row, expert, physical_rank=0) for row in range(3) for expert in range(128)])
        for seed, layer, call, rank in [(1, 1, 2, 0), (0, 2, 2, 0), (0, 1, 3, 0), (0, 1, 2, 1)]:
            changed = [ref.random_word(seed, layer, call, row, expert, physical_rank=rank) for row in range(3) for expert in range(128)]
            self.assertNotEqual(base, changed)
        self.assertEqual(len(set(base)), len(base))

    def test_rng_overflow_or_invalid_native_identity_rejected(self):
        for values in [(-1, 0, 0), (0, 48, 0), (0, 0, 2 ** 32)]:
            with self.assertRaises(ValueError):
                ref.rng_key(*values)
        with self.assertRaises(ValueError):
            ref.random_word(0, 0, 0, 2 ** 32, 0)

    def test_all_methods_exact_head_distinct_fanout_and_original_weights(self):
        z = logits()
        for ranks in (4, 8):
            for keep in (4, 6, 8):
                for mode in ref.MODES:
                    for seed in (0, 1, 2):
                        w, ids = ref.select_python(z, keep, ranks, mode, seed=seed, layer=47, call=103)
                        for row, selected, weights in zip(z, ids, w):
                            native = sorted(range(128), key=lambda e: (-ref.f32(row[e]), e))[:keep]
                            self.assertEqual(selected[:keep], native)
                            self.assertEqual(len(set(selected)), 8)
                            self.assertEqual({e * ranks // 128 for e in selected}, {e * ranks // 128 for e in native})
                            self.assertAlmostEqual(sum(weights), 1, places=6)
                            for i in range(1, 8):
                                expected_ratio = math.exp(ref.f32(row[selected[i]]) - ref.f32(row[selected[0]]))
                                self.assertAlmostEqual(weights[i] / weights[0], expected_ratio, places=5)

    def test_allowed_range_has_no_global_rank_shortage(self):
        # Head all onrank0; every other rank is globally stronger than most
        # remaining rank0 experts. Global-top32 intersection could be empty.
        z = [[100.0 if i < 4 else 90.0 - i / 1000 if i >= 32 else -100.0 - i for i in range(128)]]
        for mode in ("weighted_hot", "uniform_range"):
            _, ids = ref.select_python(z, 4, 4, mode, seed=0, layer=0, call=0)
            self.assertTrue(all(i < 32 for i in ids[0]))
            self.assertEqual(len(set(ids[0])), 8)

    def test_ties_have_explicit_lower_expert_id_head_and_range(self):
        z = [[0.0] * 128]
        for mode in ref.MODES:
            _, ids = ref.select_python(z, 4, 4, mode, seed=0, layer=0, call=0)
            self.assertEqual(ids[0][:4], [0, 1, 2, 3])
            self.assertTrue(all(i < 32 for i in ids[0]))

    def test_weighted_range_is_top32_allowed_including_head(self):
        z = logits(1)
        head = sorted(range(128), key=lambda e: (-ref.f32(z[0][e]), e))[:6]
        hot = {e // 32 for e in head}
        allowed = sorted([e for e in range(128) if e // 32 in hot], key=lambda e: (-ref.f32(z[0][e]), e))
        for mode in ("weighted_hot", "uniform_range"):
            for seed in range(20):
                _, ids = ref.select_python(z, 6, 4, mode, seed=seed, layer=1, call=2)
                self.assertTrue(set(ids[0][6:]) <= set(allowed[6:32]))

    def test_uniform_all_hot_is_not_silently_range_limited(self):
        z = logits(1)
        head = sorted(range(128), key=lambda e: (-ref.f32(z[0][e]), e))[:6]
        hot = {e // 32 for e in head}
        allowed = sorted([e for e in range(128) if e // 32 in hot], key=lambda e: (-ref.f32(z[0][e]), e))
        seen = set()
        for call in range(64):
            _, ids = ref.select_python(z, 6, 4, "uniform_hot", seed=0, layer=1, call=call)
            seen.update(ids[0][6:])
        self.assertTrue(seen - set(allowed[:32]))

    def test_uniform_candidate_pool_coverage_across_frozen_counter_keys(self):
        z = [[10.0 if e < 4 else 0.0 for e in range(128)]]
        counts = Counter()
        for call in range(256):
            _, ids = ref.select_python(z, 4, 4, "uniform_hot", seed=0, layer=0, call=call)
            counts.update(ids[0][4:])
        self.assertEqual(set(counts), set(range(4, 32)))
        self.assertLess(max(counts.values()) / min(counts.values()), 4)

    def test_padding_sentinels_and_real_rows_independent_of_padding_logits(self):
        z = logits(17)
        pad = [i % 3 == 0 for i in range(17)]
        changed = copy.deepcopy(z)
        for i, masked in enumerate(pad):
            if masked:
                changed[i] = [-value * 100 for value in changed[i]]
        for mode in ref.MODES:
            w, ids = ref.select_python(z, 6, 4, mode, seed=2, layer=5, call=13, padding=pad)
            cw, ci = ref.select_python(changed, 6, 4, mode, seed=2, layer=5, call=13, padding=pad)
            self.assertEqual((w, ids), (cw, ci))
            self.assertTrue(all(ids[i] == [-1] * 8 and w[i] == [0.0] * 8 for i in range(17) if pad[i]))

    def test_same_logits_different_local_rows_are_not_request_invariant(self):
        z = [logits(1)[0]] * 8
        _, ids = ref.select_python(z, 6, 4, "uniform_hot", seed=0, layer=0, call=0)
        self.assertGreater(len({tuple(row) for row in ids}), 1)

    def test_nonfinite_and_malformed_logits_rejected(self):
        for rows in [[[float("nan")] * 128], [[float("inf")] * 128], [[0.] * 128, [0.] * 127]]:
            with self.assertRaises(ValueError):
                ref.select_python(rows, 4, 4, "weighted_hot", seed=0, layer=0, call=0)

    def test_invalid_range_rank_and_temperature_rejected(self):
        for args in [(128, 8, 4, 1, "uniform_hot", 32, 1.), (128, 8, 4, 4, "weighted_hot", 7, 1.),
                     (128, 8, 4, 4, "weighted_hot", 32, 0.), (128, 8, 4, 4, "weighted_hot", 32, float("nan"))]:
            with self.assertRaises(ValueError):
                ref.parameters(*args)


class ProvenanceTests(unittest.TestCase):
    def fixture(self, folder):
        root = Path(folder)
        model = root / "model"
        model.mkdir()
        config = model / "config.json"
        config.write_text("{}")
        shards = []
        for i in range(16):
            path = model / f"model-{i:05d}.safetensors"
            path.write_bytes(b"a")
            shards.append({"filename": path.name, "bytes": 1, "sha256": plugin.sha(path)})
        checkpoint = root / "checkpoint.json"
        checkpoint.write_text(json.dumps({"canonical_identity_verified": True, "model_id": plugin.MODEL_ID,
            "revision": plugin.REVISION, "canonical_config_sha256": plugin.sha(config),
            "local_model_path": str(model), "safetensors": shards, "weights_verified": shards}))
        public = root / "public"
        lineage = root / "lineage.json"
        lineage.write_text(json.dumps({"source_commit": plugin.PUBLIC_COMMIT, "site_root": str(public)}))
        gate = root / "gate.json"
        gate.write_text(json.dumps({"complete": True, "cuda_required": True, "private_source_sha256": plugin.private_hashes(),
            "checks": {name: True for name in ["native_head", "distinct_experts", "dtype_and_sentinel", "original_weights",
                     "ragged_and_padding", "rng_words_and_repeat", "fanout", "ties_and_range"]}}))
        native_files = {}
        for name in plugin.NATIVE_FILES:
            native_path = root / "native" / name
            native_path.parent.mkdir(parents=True, exist_ok=True)
            native_path.write_text("# frozen fixture native source\n")
            native_files[name] = str(native_path)
        value = {"parameters": {"mode": "weighted_hot", "keep": 6, "seed": 0, "range_limit": 32, "temperature": 1., "ranks": 4, "load_penalty": None},
            "native_source_files": native_files, "native_source_sha256": {name: plugin.sha(path) for name, path in native_files.items()},
            "private_source_sha256": plugin.private_hashes(), "private_site": str(Path(plugin.__file__).resolve().parent.parent),
            "model_id": plugin.MODEL_ID, "revision": plugin.REVISION, "model_path": str(model),
            "public_site": str(public), "public_source_proof": {"path": str(lineage), "sha256": plugin.sha(lineage)},
            "checkpoint_proof": {"path": str(checkpoint), "sha256": plugin.sha(checkpoint)},
            "checkpoint_stat_signatures": {s["filename"]: [(model / s["filename"]).stat().st_size,
                (model / s["filename"]).stat().st_mtime_ns] for s in shards},
            "gpu_gate": {"path": str(gate), "sha256": plugin.sha(gate)}, "record_calls": True, "purpose": "quality",
            "runtime": {"ep": 4, "dp": 4, "tp": 1, "pp": 1, "max_model_len": 8192, "max_num_seqs": 256,
                 "max_num_batched_tokens": 16384, "graphs": "NONE", "prefix_caching": False,
                 "all2all_backend": "deepep_high_throughput", "dtype": "torch.bfloat16",
                 "expert_placement": "linear", "seed": 0, "logprobs_mode": "raw_logprobs", "use_v2_model_runner": True,
               "worker_extension_cls": "stochastic_tail.worker_extension.CheckpointExtension", "server_dev_mode": True}}
        plan = root / "plan.json"
        return value, plan, plugin.sha(config)

    def read_fixture(self, value, path, config_sha):
        path.write_text(json.dumps(value))
        with patch.object(plugin, "CONFIG_SHA", config_sha):
            return plugin.read_plan(path, plugin.sha(path))

    def test_complete_same_source_gpu_gate_and_model_identity_accept(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            self.assertEqual(self.read_fixture(value, path, config_sha)["parameters"]["seed"], 0)

    def test_changed_private_overlay_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            value["private_source_sha256"]["stochastic_tail/kernel.py"] = "stale"
            with self.assertRaisesRegex(RuntimeError, "overlay source drift"):
                self.read_fixture(value, path, config_sha)

    def test_failed_cuda_gate_cannot_authorize_any_quality_server(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            gate_path = Path(value["gpu_gate"]["path"])
            gate = json.loads(gate_path.read_text())
            gate["complete"] = False
            gate_path.write_text(json.dumps(gate))
            value["gpu_gate"]["sha256"] = plugin.sha(gate_path)
            with self.assertRaisesRegex(RuntimeError, "GPU correctness gate"):
                self.read_fixture(value, path, config_sha)

    def test_model_shard_changed_after_root_verification_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            next(Path(value["model_path"]).glob("*.safetensors")).write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "shard stat identity"):
                self.read_fixture(value, path, config_sha)

    def test_wrong_score_processing_mode_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            value["runtime"]["logprobs_mode"] = "processed_logprobs"
            with self.assertRaisesRegex(RuntimeError, "Runtime differs"):
                self.read_fixture(value, path, config_sha)

    def test_quality_cannot_disable_call_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            value["record_calls"] = False
            with self.assertRaisesRegex(RuntimeError, "disable the CPU call ledger"):
                self.read_fixture(value, path, config_sha)

    def test_private_entrypoint_outside_installed_site_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            value, path, config_sha = self.fixture(folder)
            value["private_site"] = "/different/site"
            with self.assertRaisesRegex(RuntimeError, "outside the frozen overlay site"):
                self.read_fixture(value, path, config_sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
