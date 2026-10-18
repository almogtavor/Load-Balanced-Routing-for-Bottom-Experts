"""CPU checks for paper protocol invariants and independent-unit statistics."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import paper_reproduce as paper


def config():
    # Small synthetic dimensions exercise the same all-rank/all-layer grouping.
    # Real CLI configuration is independently fixed to DP4/8 and 48 layers.
    return {"dp": 2, "model": "/models/verified", "expected_layers": 2, "target_tokens": 16384,
            "minimum_complete_steps_per_rank_per_measured_round": 1,
            "measured_rounds": 3, "expected_gpu_name": "H100",
            "shape": {"name": "paper_prefill", "prompt_tokens": 4,
                      "output_tokens": 1, "concurrency": 1, "requests": 1}}


def snapshot(rank=0, arm="base", central=1.0):
    rows, forward = [], 0
    factor = (0.8, 1.2)[rank]
    for phase in ("warmup", "measure"):
        for round_number in range(1, 4):
            epoch = round_number + (3 if phase == "measure" else 0)
            for layer in range(2):
                value = central * factor * (2 if phase == "warmup" else 1)
                rows.append({"forward_id": forward, "seen_forward": forward + 97,
                    "T": 16384, "bucket": 16384, "layer_name": f"model.layers.{layer}.mlp",
                    "layer_index": layer, "step_id": epoch, "epoch": epoch, "phase": phase,
                    "round": round_number, "moe_ms": value * 7, "dispatch_ms": value * 2,
                    "combine_ms": value * 3, "flush_origin": "chunk64", "valid": True, "errors": []})
                rows[-1].update(dispatch_input_T=16384,
                    span_calls={p: 1 for p in ("moe", "dispatch", "combine")},
                    span_ends={p: 1 for p in ("moe", "dispatch", "combine")})
                forward += 1
    keep, penalty = paper.ARMS[arm]
    return {"schema_version": 1, "ready": True, "drained": True, "errors": [], "records_included": True,
        "pending_spans": 0, "open_forwards": 0, "flush_in_progress": False,
        "control_ack": {"phase": "drain", "epoch": 7, "round": 3},
        "metadata": {"ep_rank": rank, "dp_rank": rank, "ep_world_size": 2,
            "pid": 500 + rank, "gpu": "NVIDIA H100 80GB HBM3", "device_index": rank,
            "arm": arm, "repeat": "0", "placement_keep": str(keep) if keep is not None else None,
            "model": "/models/verified",
            "load_penalty": str(penalty) if penalty is not None else None,
            "backend": "deepep_high_throughput", "cudagraph_mode": "NONE",
            "activation_dtype": "torch.bfloat16", "hidden_size": 2048,
            "selector": "baseline", "experimental_impl_is_none": True,
            "kernel_source_sha256": paper.ORIGINAL_SOURCE["src/placement_router/kernel.py"],
            "pristine_source_commit": paper.BASE_COMMIT, "probe_sha256": "probe-sha",
            "registered_MoE_layer_names": ["model.layers.0.mlp", "model.layers.1.mlp"],
            "timed_methods": {"moe": {"source_sha256": "moe-source"}},
            "timing_semantics": {"warmup_global_forwards": 96, "flush_whole_forwards": 64}},
        "forward_records": rows}


def http_round(epoch, phase, index):
    record = {"request_id": 0, "prompt_id": 0, "ttft_ms": 1.0, "latency_ms": 1.1,
        "last_token_latency_ms": 1.0, "tpot_ms": None, "itl_ms": [], "inter_chunk_ms": [],
        "all_chunks_single_token": True, "chunks": [{"tokens": 1}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        "finish_reason": "length"}
    return {"probe_control": {"epoch": epoch, "phase": phase, "round": index},
        "started_unix_ns": 1000000000, "finished_unix_ns": 2000000000,
        "elapsed_seconds": 1.,
        "requests": 1, "concurrency": 1, "prompt_tokens": 4, "completion_tokens": 1,
        "output_tokens_per_second": 1., "requests_per_second": 1., "total_tokens_per_second": 5.,
        "ttft_ms": {"count": 1, "p50": 1., "p95": 1., "p99": 1.},
        "request_latency_ms": {"count": 1, "p50": 1.1, "p95": 1.1, "p99": 1.1},
        "tpot_ms": None, "itl_ms": None, "requests_with_single_token_chunks": 1,
        "records": [record], "jit_events_observed_in_log_slice": []}


def write_launch(root, identity, repeat, arm, central):
    directory = root / f"r{repeat:02d}_{arm}"
    rows = [snapshot(rank, arm, central) for rank in range(2)]
    for row in rows:
        row["metadata"]["repeat"] = str(repeat)
    files = []
    for rank, row in enumerate(rows):
        file = directory / "probe" / f"rank{rank}_pid{500 + rank}.json"
        paper.common.atomic_json(file, row)
        files.append({"path": str(file.relative_to(directory)), "sha256": paper.digest(file),
                      "bytes": file.stat().st_size})
    data = {"complete": True, "arm": arm, "repeat": repeat,
        "source_sha256": identity["source_sha256"], "probe_files": files,
        "timing_summary": paper.validate_snapshot(rows, identity["config"], expected_arm=arm,
                                                   expected_sources=identity["source_sha256"]),
        "warmups": [http_round(i, "warmup", i) for i in range(1, 4)],
        "rounds": [http_round(i + 3, "measure", i) for i in range(1, 4)]}
    paper.common.atomic_json(directory / "result.json", data)


class ProtocolTests(unittest.TestCase):
    def test_fixed_full_schedule(self):
        schedule = paper.schedule()
        self.assertEqual(len(schedule), 6)
        for row in schedule:
            self.assertEqual(len(row["arms"]), 8)
            self.assertEqual(set(row["arms"]), set(paper.ARMS))
            self.assertEqual([row["arms"][i] for i in (0, 3, 7)], list(paper.BASES))
        slots = (1, 2, 4, 5, 6)
        for arm in paper.TREATMENTS:
            self.assertEqual({row["arms"].index(arm) for row in schedule[:5]}, set(slots))
        first = [a for a in schedule[0]["arms"] if a not in paper.BASES]
        last = [a for a in schedule[5]["arms"] if a not in paper.BASES]
        self.assertEqual(last, list(reversed(first)))
        self.assertEqual(paper.schedule(), schedule)

    def test_pristine_source_guard(self):
        identity = {"source_sha256": dict(paper.ORIGINAL_SOURCE)}
        paper.require_original_source(identity)
        identity["source_sha256"]["src/placement_router/kernel.py"] = "modified"
        with self.assertRaisesRegex(RuntimeError, "pristine"):
            paper.require_original_source(identity)

    def test_server_and_environment_scope(self):
        args = SimpleNamespace(vllm="vllm", model="/models/verified", port=8077,
                               dp=8, gpu_util=.88, tag="unit")
        command = paper.server_command(args)
        self.assertEqual(command[command.index("--all2all-backend") + 1], "deepep_high_throughput")
        self.assertIn('--compilation-config={"cudagraph_mode":"NONE"}', command)
        self.assertNotIn("--worker-extension-cls", command)
        with mock.patch.dict(paper.os.environ, {"PLACEMENT_EXPERIMENTAL_SELECTOR": "rows4_direct",
                    "PLACEMENT_DEEPEP_NUM_SMS": "64", "PLACEMENT_KEEP": "6"}):
            native = paper.arm_environment(args, "base", 0, Path("/results/test"))
            tuned = paper.arm_environment(args, "balanced6p1", 0, Path("/results/test"))
        self.assertNotIn("PLACEMENT_KEEP", native)
        self.assertNotIn("PLACEMENT_EXPERIMENTAL_SELECTOR", tuned)
        self.assertNotIn("PLACEMENT_DEEPEP_NUM_SMS", tuned)
        self.assertEqual(tuned["PLACEMENT_KEEP"], "6")
        self.assertEqual(tuned["PLACEMENT_LOAD_PENALTY"], "1")

    def test_missing_preflight_starts_no_full_server(self):
        with tempfile.TemporaryDirectory() as temp:
            identity = {"source_sha256": dict(paper.ORIGINAL_SOURCE)}
            with mock.patch.object(paper, "ensure_manifest", return_value=identity), \
                    mock.patch.object(paper, "make_prompts", return_value=[]), \
                    mock.patch.object(paper.signal, "signal"), \
                    mock.patch.object(paper, "measure_launch") as launch:
                with self.assertRaises(FileNotFoundError):
                    paper.main(["--mode", "run", "--output", temp, "--model", "/model",
                                "--checkpoint-proof", "/proof"])
                launch.assert_not_called()


class CoverageTests(unittest.TestCase):
    def test_exact_T_not_power_bucket_and_separate_estimators(self):
        rows = [snapshot(r) for r in range(2)]
        extra = copy.deepcopy(rows[0]["forward_records"][0])
        extra.update(T=32767, dispatch_input_T=32767, forward_id=9999, dispatch_ms=999., combine_ms=999., moe_ms=999.)
        rows[0]["forward_records"].append(extra)
        result = paper.validate_snapshot(rows, config(), expected_arm="base")
        self.assertAlmostEqual(result["mean_rank_medians"]["fixed_warmup_measured_only"]["dispatch_ms"], 2.)
        self.assertAlmostEqual(result["mean_rank_medians"]["paper_all_post96"]["dispatch_ms"], 3.)
        self.assertEqual(len(result["coverage"]), 6)

    def test_partial_step_and_missing_rank_fail(self):
        rows = [snapshot(r) for r in range(2)]
        rows[0]["forward_records"].pop()
        with self.assertRaisesRegex(RuntimeError, "Partial"):
            paper.validate_snapshot(rows, config())
        with self.assertRaisesRegex(RuntimeError, "rank"):
            paper.validate_snapshot([snapshot()], config())

    def test_wrong_source_and_pending_tail_fail(self):
        rows = [snapshot(r) for r in range(2)]
        rows[1]["metadata"]["kernel_source_sha256"] = "other"
        with self.assertRaisesRegex(RuntimeError, "source"):
            paper.validate_snapshot(rows, config())
        rows = [snapshot(r) for r in range(2)]
        rows[1]["pending_spans"] = 1
        with self.assertRaisesRegex(RuntimeError, "drained"):
            paper.validate_snapshot(rows, config())

    def test_actual_penalty_and_checkpoint_identity_fail(self):
        rows = [snapshot(r, "balanced6p1") for r in range(2)]
        rows[0]["metadata"]["load_penalty"] = "0"
        with self.assertRaisesRegex(RuntimeError, "penalty"):
            paper.validate_snapshot(rows, config(), expected_arm="balanced6p1")
        rows = [snapshot(r) for r in range(2)]
        rows[0]["metadata"]["model"] = "/models/different"
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            paper.validate_snapshot(rows, config(), expected_arm="base")

    def test_compact_or_undrained_ack_is_not_final_evidence(self):
        rows = [snapshot(r) for r in range(2)]
        rows[0]["records_included"] = False
        with self.assertRaisesRegex(RuntimeError, "raw forward"):
            paper.validate_snapshot(rows, config())
        rows = [snapshot(r) for r in range(2)]
        rows[0]["control_ack"]["epoch"] = 6
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            paper.validate_snapshot(rows, config())


class StatisticsTests(unittest.TestCase):
    def test_student_t_and_zero_SE(self):
        self.assertAlmostEqual(paper.two_sided_t_pvalue(2.57058183564, 5), .05, places=10)
        self.assertEqual(paper.two_sided_t_pvalue(0, 5), 1)
        self.assertEqual(paper.sample_summary([])["n"], 0)
        self.assertIsNone(paper.sample_summary([1])["two_sided_t_pvalue"])
        self.assertEqual(paper.sample_summary([2] * 6)["two_sided_t_pvalue"], 0)

    def test_complete_paper_floor_ratios_and_missing_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = {"config": config(), "schedule": paper.schedule(),
                        "source_sha256": {**paper.ORIGINAL_SOURCE, "experiments/paper_probe.py": "probe-sha"}}
            paper.common.atomic_json(root / "manifest.json", {"identity": identity})
            times = {"base": 1., "base2": 1.02, "base3": 1.04, "keep8": 1.02,
                     "plain6": 1., "balanced6p1": .98, "plain4": .99, "balanced4p1": .97}
            for repeat in range(6):
                for arm, central in times.items():
                    write_launch(root, identity, repeat, arm, central)
            report = paper.write_analysis(root)
            self.assertTrue(report["complete"], report["issues"])
            self.assertEqual(report["counts"]["valid_fresh_servers"], 48)
            cell = next(c for c in report["comparisons"] if c["estimator"] == "fixed_warmup_measured_only"
                        and c["phase"] == "dispatch_ms" and c["arm"] == "plain6")
            self.assertEqual(cell["paper_arithmetic_speedup_pct"]["n"], 6)
            self.assertAlmostEqual(cell["paper_arithmetic_speedup_pct"]["mean"], 2.)
            self.assertAlmostEqual(cell["floor_pct"], 2.)
            load = next(c for c in report["load_term_comparisons"] if c["estimator"] == "fixed_warmup_measured_only"
                        and c["phase"] == "dispatch_ms" and c["plain"] == "plain6")
            self.assertAlmostEqual(load["paper_arithmetic_speedup_pct"]["mean"], (1 / .98 - 1) * 100)
            (root / "r05_base3" / "result.json").unlink()
            incomplete = paper.analyze(root)
            self.assertFalse(incomplete["complete"])
            self.assertTrue(any("r05_base3" in issue for issue in incomplete["issues"]))
            reduced = next(c for c in incomplete["comparisons"] if c["estimator"] == "fixed_warmup_measured_only"
                           and c["phase"] == "dispatch_ms" and c["arm"] == "plain6")
            self.assertEqual(reduced["paper_arithmetic_speedup_pct"]["n"], 5)
            self.assertFalse(reduced["complete_pairs"])


if __name__ == "__main__":
    unittest.main()
