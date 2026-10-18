"""Meaningful CPU fixtures for typed-cast and exact-shape profile attribution."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import analyze_paper_profiles as audit


def cpu(name, start, duration, ext, dims=None, types=None, tid=1):
    return {"ph": "X", "cat": "cpu_op", "pid": 10, "tid": tid, "name": name,
        "ts": start, "dur": duration, "args": {"External id": ext,
            "Input Dims": dims or [], "Input type": types or []}}


def kernel(name, start, duration, ext=None, corr=None):
    return {"ph": "X", "cat": "kernel", "pid": 0, "tid": 9, "name": name,
        "ts": start, "dur": duration, "args": {"External id": ext, "correlation": corr}}


def one_forward(arm="balanced6p1", T=16384, start=0, ext=100, correlation_only=False):
    events = [cpu("vllm::moe_forward", start, 100, ext, [[T, 2048]])]
    router = "_par_select" if arm == "balanced6p1" else "void vllm::moe::topkGating<8,128>()"
    events += [cpu("router_cpu", start + 1, 5, ext + 1),
        kernel(router, start + 1000, 4, ext + 1),
        cpu("expert_cpu", start + 30, 5, ext + 4),
        kernel("fused_moe_kernel", start + 1020, 20, ext + 4),
        cpu("dispatch_cpu", start + 40, 5, ext + 5),
        kernel("void deep_ep::legacy::intranode::notify_dispatch<4>()", start + 1040, 8, ext + 5),
        kernel("void deep_ep::legacy::intranode::dispatch<4>()", start + 1048, 7, ext + 5),
        cpu("combine_cpu", start + 50, 5, ext + 6),
        kernel("void deep_ep::legacy::intranode::cached_notify_combine<4>()", start + 1060, 6, ext + 6),
        kernel("void deep_ep::legacy::intranode::combine<4>()", start + 1066, 9, ext + 6)]
    if arm == "balanced6p1":
        events += [cpu("aten::_to_copy", start + 10, 15, ext + 2, [[T, 8]], ["int"]),
            cpu("aten::copy_", start + 15, 5, ext + 3, [[T, 8], [T, 8]], ["long int", "int"]),
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": 10, "tid": 1,
             "ts": start + 16, "dur": 1, "args": {"correlation": ext + 900}},
            kernel("some_vectorized_elementwise_kernel", start + 1010, 2,
                   None if correlation_only else ext + 3, ext + 900)]
    return events


class PaperProfileAnalysisTests(unittest.TestCase):
    def write_trace(self, path, events):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as stream:
            json.dump({"traceEvents": events, "baseTimeNanoseconds": 1000000000}, stream)
        return path

    def test_typed_cast_external_and_launch_correlations_survive_gpu_async_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.gz"
            for correlation_only in (False, True):
                events = one_forward(correlation_only=correlation_only)
                row = audit.summarize_trace(self.write_trace(path, events))
                cast = row["routing_id_cast"]
                self.assertEqual(cast["typed_int32_to_int64_T8_cpu_copy"]["calls"], 1)
                self.assertEqual(cast["correlated_gpu_cast"]["calls"], 1)
                method = "launch_inside_typed_copy" if correlation_only else "typed_copy_external_id"
                self.assertEqual(cast["gpu_correlation_methods"], {method: 1})
                self.assertEqual(row["by_exact_local_hidden_T"]["16384"]["routing_id_cast"]["calls"], 1)
                self.assertEqual(cast["cpu_copy_calls_per_moe_forward"], 1)
                self.assertEqual(row["categories"]["routing_id_cast"]["summed_gpu_duration_us"], 2)
                self.assertEqual(row["categories"]["dispatch_notify_wait"]["calls"], 1)
                self.assertEqual(row["categories"]["dispatch_payload"]["calls"], 1)
                self.assertEqual(row["categories"]["combine_notify_wait"]["calls"], 1)
                self.assertEqual(row["categories"]["combine_payload"]["calls"], 1)
                # GPU starts >900us after CPU scope ends; CPU linkage, not proximity, proves T.
                self.assertNotIn("unattributed", row["gpu_to_moe_attribution_counts"])

    def test_wrong_type_outside_moe_and_kernel_name_alone_cannot_prove_cast(self):
        with tempfile.TemporaryDirectory() as temp:
            events = one_forward("stock")
            events += [cpu("aten::copy_", 12, 4, 201, [[16384, 8], [16384, 8]], ["long int", "float"]),
                kernel("int32_to_int64_cast_kernel", 1100, 3, 201),
                cpu("aten::copy_", 200, 4, 202, [[16384, 8], [16384, 8]], ["long int", "int"]),
                kernel("typed_cast_outside_moe", 1200, 3, 202),
                cpu("aten::copy_", 16, 4, 203, [[16384, 8], [16384, 8]], ["long int", "int"], tid=2),
                kernel("typed_cast_wrong_thread", 1300, 3, 203)]
            row = audit.summarize_trace(self.write_trace(Path(temp) / "trace.gz", events))
            self.assertEqual(row["routing_id_cast"]["typed_int32_to_int64_T8_cpu_copy"]["calls"], 0)
            self.assertEqual(row["routing_id_cast"]["correlated_gpu_cast"]["calls"], 0)
            self.assertEqual(row["gpu_to_moe_attribution_counts"]["unattributed"], 2)

    def test_exact_T_is_distinct_even_with_same_legacy_bucket(self):
        with tempfile.TemporaryDirectory() as temp:
            events = one_forward(T=16384) + one_forward(T=32767, start=2000, ext=200)
            row = audit.summarize_trace(self.write_trace(Path(temp) / "trace.gz", events))
            self.assertEqual(row["moe_actual_hidden_T_counts"], {"16384": 1, "32767": 1})
            self.assertEqual(row["moe_power_of_two_bucket_counts"], {"16384": 2})
            self.assertEqual(set(row["by_exact_local_hidden_T"]), {"16384", "32767"})
            self.assertEqual(row["routing_id_cast"]["by_actual_source_T"]["32767"]["gpu_cast"]["calls"], 1)

    def test_missing_gpu_correlation_is_reported_without_inventing_gpu_cast(self):
        with tempfile.TemporaryDirectory() as temp:
            events = one_forward()
            events = [e for e in events if e["name"] != "cudaLaunchKernel"]
            for event in events:
                if event["name"] == "some_vectorized_elementwise_kernel":
                    event["args"] = {}
            row = audit.summarize_trace(self.write_trace(Path(temp) / "trace.gz", events))
            cast = row["routing_id_cast"]
            self.assertEqual(cast["typed_int32_to_int64_T8_cpu_copy"]["calls"], 1)
            self.assertEqual(cast["correlated_gpu_cast"]["calls"], 0)
            self.assertEqual(cast["typed_copy_scopes_without_gpu_kernel"], 1)

    def test_all_rank_evidence_is_positive_and_rejects_tuned_selector_or_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            events = []
            for i in range(48):
                events += one_forward(start=i * 2000, ext=i * 100)
            row = audit.summarize_trace(self.write_trace(Path(temp) / "trace.gz", events))
            self.assertEqual(audit.validate_rank(row, "balanced6p1"), [])
            row["unexpected_placement_kernel_calls"] = 1
            self.assertIn("Experimental placement kernel present", audit.validate_rank(row, "balanced6p1"))
            row["unexpected_placement_kernel_calls"] = 0
            row["cuda_graph_launch_api_counts"] = {"cudaGraphLaunch": 1}
            self.assertIn("Unexpected graph launch in HT/NONE profile", audit.validate_rank(row, "balanced6p1"))
            row["moe_actual_hidden_T_counts"] = {"32767": 48}
            self.assertTrue(any("actual-T16384" in value for value in audit.validate_rank(row, "balanced6p1")))

    def test_trace_copy_resolution_uses_hash_not_first_matching_filename(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = self.write_trace(root / "z_good/trace.json.gz", one_forward())
            self.write_trace(root / "a_bad/trace.json.gz", one_forward("stock"))
            record = {"path": "/missing/trace.json.gz", "bytes": good.stat().st_size, "sha256": audit.sha256(good)}
            self.assertEqual(audit.locate_trace(record, [root]), good.resolve())
            record["sha256"] = "0" * 64
            with self.assertRaises(FileNotFoundError):
                audit.locate_trace(record, [root])

    def make_result(self, root, arm, dp=4):
        profile_root = root / f"ep{dp}"
        directory = profile_root / f"profile_{arm}"
        directory.mkdir(parents=True, exist_ok=True)
        config = {"model": "/model", "vllm": "vllm", "port": 8077, "dp": dp, "ep": dp, "tp": 1,
            "graphs": "NONE", "backend": "deepep_high_throughput", "dtype": "bfloat16",
            "num_sms_source_default": 20, "gpu_util": .88}
        sources = dict(audit.paper.ORIGINAL_SOURCE)
        sources["experiments/paper_profile.py"] = "profile-source"
        identity = {"source_sha256": sources, "configuration": config,
            "checkpoint": {"proof": {"model_id": audit.paper.MODEL_ID, "canonical_identity_verified": True,
                "revision": "1" * 40}, "proof_sha256": "checkpoint-proof"}}
        audit.paper.common.atomic_json(profile_root / "manifest.json", {"identity": identity})
        env = {"PLACEMENT_KEEP": "6" if arm == "balanced6p1" else None,
            "PLACEMENT_LOAD_PENALTY": "1" if arm == "balanced6p1" else None,
            "PLACEMENT_PROBE": None, "PLACEMENT_EXPERIMENTAL_SELECTOR": None,
            "OMP_NUM_THREADS": "1", "NVSHMEM_QP_DEPTH": "4096", "VLLM_PLUGINS": "placement_aware_router"}
        command = audit.paper.server_command(SimpleNamespace(**config))
        command += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_record_shapes": True})]
        audit.paper.common.atomic_json(directory / "server.json", {"controlled_environment": env,
            "source_sha256": sources, "cuda_event_probe_active": False, "paper_probe_environment": {}, "command": command})
        log = "'cudagraph_mode': <CUDAGraphMode.NONE: 0>\n"
        if arm == "balanced6p1":
            log += f"placement-aware routing active, keep=6 of top_k=8, ep_size={dp}\n" * dp
        (directory / "server.log").write_text(log)
        traces = []
        for rank in range(dp):
            events = []
            for i in range(48):
                events += one_forward(arm, start=i * 2000, ext=i * 100)
            path = self.write_trace(directory / f"dp{rank}.pt.trace.json.gz", events)
            traces.append({"path": str(path), "dp_rank": rank, "bytes": path.stat().st_size, "sha256": audit.sha256(path)})
        auxiliary = self.write_trace(directory / "engine.async_llm.pt.trace.json.gz", [cpu("async_request", 0, 10, 7)])
        traces.append({"path": str(auxiliary), "dp_rank": None, "bytes": auxiliary.stat().st_size,
                       "sha256": audit.sha256(auxiliary)})
        def workload(count):
            return {"requests": count, "concurrency": 128, "prompt_tokens": 4096 * count, "completion_tokens": count}
        audit.paper.common.atomic_json(directory / "result.json", {"complete": True, "measurement_type": "profile_only",
            "arm": arm, "identity_sha256": audit.sha256(profile_root / "manifest.json"),
            "warmups": [workload(256) for _ in range(3)], "profile_workload": workload(128),
            "trace_summary": {"traces": traces}})
        return directory / "result.json"

    def test_end_to_end_offline_pair_has_cast_evidence_and_safe_analysis_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stock, balanced = (self.make_result(root, arm) for arm in ("stock", "balanced6p1"))
            output = root / "local-only/audit.json"
            with patch("builtins.print"):
                self.assertEqual(audit.main([str(stock), str(balanced), "--output", str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertTrue(result["valid"])
            for profile in result["profiles"]:
                self.assertEqual(len(profile["auxiliary_non_DP_traces"]), 1)
                self.assertEqual(len(profile["ranks"]), 4)
                for row in profile["ranks"]:
                    expected = 48 if profile["arm"] == "balanced6p1" else 0
                    self.assertEqual(row["routing_id_cast"]["correlated_gpu_cast"]["calls"], expected)
            self.assertTrue(any(row["category"] == "router" for row in result["matched_local_T_comparisons"]))
            self.assertIn("not clean-serving", output.with_suffix(".md").read_text())
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                audit.main([str(stock), "--output", str(stock)])

    def test_probe_contamination_and_incomplete_warmups_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.make_result(Path(temp), "stock")
            server_path = result.parent / "server.json"
            server = json.loads(server_path.read_text())
            original = copy.deepcopy(server)
            server["controlled_environment"]["PLACEMENT_PROBE"] = "/paper_probe.py"
            audit.paper.common.atomic_json(server_path, server)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                audit.inspect_result(result, [])
            audit.paper.common.atomic_json(server_path, original)
            data = json.loads(result.read_text())
            data["warmups"].pop()
            audit.paper.common.atomic_json(result, data)
            with self.assertRaisesRegex(ValueError, "three full warmup"):
                audit.inspect_result(result, [])

    def test_event_findings_remain_separate_and_checkpoint_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "paper_analysis.json"
            source = dict(audit.paper.ORIGINAL_SOURCE)
            values = {"mean": 1.0, "sd": .1, "n": 6, "ci95": [.8, 1.2]}
            finding = {"complete": True, "scope": {"config": {"dp": 4}, "source_sha256": source,
                "checkpoint": {"proof_sha256": "events-checkpoint"}}, "counts": {"valid_fresh_servers": 48},
                "comparisons": [{"arm": "balanced6p1", "estimator": "paper_all_post96", "phase": "moe_ms",
                    "paper_arithmetic_speedup_pct": values, "floor_pct": .5}], "noise_floors": {}}
            audit.paper.common.atomic_json(path, finding)
            profile = {"ep": 4, "canonical_checkpoint": {"proof_sha256": "other-checkpoint"},
                       "production_source_sha256": source}
            rows = audit.phase_findings([path], [profile])
            self.assertFalse(rows[0]["same_checkpoint_proof_and_production_source"])
            self.assertEqual(rows[0]["balanced6p1_phase_comparisons"][0]["paper_arithmetic_speedup_pct"], values)
            self.assertEqual(json.loads(path.read_text()), finding)
            profile["canonical_checkpoint"]["proof_sha256"] = "events-checkpoint"
            self.assertTrue(audit.phase_findings([path], [profile])[0]["same_checkpoint_proof_and_production_source"])


if __name__ == "__main__":
    unittest.main()
