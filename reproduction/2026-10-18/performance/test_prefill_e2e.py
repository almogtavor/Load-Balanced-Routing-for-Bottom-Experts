#!/usr/bin/env python3
"""CPU protocol failures and real local HTTP concurrency; no GPU/model imports."""
import concurrent.futures as cf
from collections import Counter
import http.server
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("prefill_e2e", Path(__file__).with_name("prefill_e2e.py"))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 128
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        index = body["prompt"][0] - 1
        self.server.received.append(index)
        fail = self.server.fail_index == index
        if fail and self.server.failure == "http":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"fixture500"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        first = {"choices": [{"index": 0, "token_ids": [100 + index], "text": "X", "finish_reason": None,
                               "prompt_token_ids": body["prompt"]}]}
        finish = {"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"prompt_tokens": 4096, "completion_tokens": 1}}
        if fail and self.server.failure == "usage":
            usage["usage"]["completion_tokens"] = 0
        payloads = [json.dumps(first), json.dumps(finish), json.dumps(usage)]
        if not (fail and self.server.failure == "incomplete"):
            payloads.append("[DONE]")
        self.wfile.write((": comment\n\n" + "".join("data: " + p + "\n\n" for p in payloads)).encode())


class ProtocolTests(unittest.TestCase):
    def test_williams_position_and_predecessor_balance(self):
        rows = m.schedule()
        self.assertEqual(len(rows), 6)
        for position in range(6):
            self.assertEqual({r["arms"][position] for r in rows}, set(m.ARMS))
        pairs = Counter((a, b) for r in rows for a, b in zip(r["arms"], r["arms"][1:]))
        self.assertEqual(len(pairs), 30)
        self.assertEqual(set(pairs.values()), {1})

    def test_frozen_actual_prompts(self):
        rows = m.verify_prompts(Path(__file__).with_name("prompts64.json"))
        self.assertEqual(len(rows), 64)
        self.assertEqual(len(rows[-1]["token_ids"]), 4096)

    def test_changed_prompt_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prompts.json"
            path.write_text("[]")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                m.verify_prompts(path)

    def test_timed_env_sanitizes_all_instrumentation_and_tuning(self):
        original = {"PLACEMENT_KEEP": "4", "PLACEMENT_PROBE": "/bad.py", "TASK_BENCH_ARM": "bad",
                    "VLLM_TORCH_PROFILER_DIR": "bad", "VLLM_DEEPEP_NUM_SMS": "99", "PAPER_PROBE_DIR": "bad",
                    "HF_TOKEN": "private", "PATH": "/bin"}
        with patch.dict(os.environ, original, clear=True):
            env = m.sanitized_env({"environment": {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}}, "stock")
        self.assertEqual(env, {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1,2,3"})

    def test_arm_environment_exact(self):
        with patch.dict(os.environ, {}, clear=True):
            env = m.sanitized_env({"environment": {}}, "keep6p8")
        self.assertEqual(env, {"PLACEMENT_KEEP": "6", "PLACEMENT_LOAD_PENALTY": "8.0"})

    def test_command_has_no_profiler_and_exact_geometry(self):
        cmd = m.server_command({"vllm": "/bin/vllm", "model": "/model", "port": 8077,
                                "max_num_seqs": 256, "gpu_util": .9})
        self.assertEqual(cmd[cmd.index("--max-model-len") + 1], "4105")
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "256")
        self.assertEqual(cmd[cmd.index("--max-num-batched-tokens") + 1], "16384")
        self.assertIn('--compilation-config={"cudagraph_mode":"NONE"}', cmd)
        self.assertFalse(any("profiler" in c for c in cmd))

    def test_finish_only_chunk_without_optional_ids_is_valid(self):
        value = {"chunks": []}
        m.parse_event(json.dumps({"choices": [{"index": 0, "finish_reason": "length", "text": ""}]}), value, 1)
        self.assertEqual(value["finish_reason"], "length")

    def test_text_without_ids_is_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "return_token_ids"):
            m.parse_event(json.dumps({"choices": [{"index": 0, "text": "X"}]}), {"chunks": []}, 1)

    def test_duplicate_finish_is_invalid(self):
        value = {"chunks": [], "finish_reason": "length"}
        with self.assertRaisesRegex(RuntimeError, "Duplicate finish"):
            m.parse_event(json.dumps({"choices": [{"index": 0, "text": "", "finish_reason": "length"}]}), value, 1)

    def test_duplicate_usage_is_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate final usage"):
            m.parse_event(json.dumps({"choices": [], "usage": {"completion_tokens": 1}}),
                          {"chunks": [], "usage": {"completion_tokens": 1}}, 1)

    def test_wrong_choice_index_is_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "choice index"):
            m.parse_event(json.dumps({"choices": [{"index": 1, "token_ids": [1]}]}), {"chunks": []}, 1)

    def test_stream_error_is_invalid(self):
        with self.assertRaisesRegex(RuntimeError, "Streaming server error"):
            m.parse_event('{"error":"bad"}', {"chunks": []}, 1)

    def test_exact_prompt_echo_required_when_present(self):
        with self.assertRaisesRegex(RuntimeError, "prompt IDs differ"):
            m.parse_event(json.dumps({"choices": [{"index": 0, "token_ids": [], "prompt_token_ids": [2]}]}),
                          {"chunks": []}, 1, [1])

    def test_exclusive_write_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.json"
            m.save(path, {"old": 1}, exclusive=True)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                m.save(path, {"new": 2}, exclusive=True)
            self.assertEqual(m.read(path), {"old": 1})

    def test_pid_reuse_guard_never_signals_new_process(self):
        child = type("Child", (), {"pid": 123})()
        identity = {"pid": 123, "pgrp": 123, "start_ticks": "10"}
        with patch.object(m, "process_record", return_value={"start_ticks": "11"}), patch.object(m.os, "killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "PID reused"):
                m.stop_owned(child, identity)
            kill.assert_not_called()

    def test_source_drift_rejected_in_installed_site(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, site = root / "source", root / "site"
            (source / "src").mkdir(parents=True)
            site.mkdir()
            (source / "src/a.py").write_bytes(b"good")
            (site / "a.py").write_bytes(b"bad")
            proof = root / "proof.json"
            m.save(proof, {"source_commit": m.COMMIT, "source_root": str(source), "site_root": str(site)})
            with patch.object(m, "SOURCE", {"src/a.py": m.bytes_sha(b"good")}):
                with self.assertRaisesRegex(RuntimeError, "Production source drift"):
                    m.source_identity(proof, source, site)

    def test_failed_cleanup_and_source_drift_still_write_failure_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan_path = root / "plan.json"
            m.save(plan_path, {})
            campaign = object.__new__(m.Campaign)
            campaign.plan_path, campaign.root, campaign.mode = plan_path, root, "timed"
            campaign.state, campaign.child, campaign.child_identity = {"launches": []}, None, None
            campaign.plan = {"port": 8077, "environment": {}, "server_command": ["fake-vllm"],
                             "source_proof": "fake", "source_root": "fake", "site_root": "fake",
                             "prompt_file": "fake", "warmup_rounds": 3, "measured_rounds": 6,
                             "request_timeout": 1, "stop_unix": time.time() + 1000}
            campaign.persist = lambda: None
            campaign.validate = lambda: None
            campaign.wait_health = lambda: None
            child = type("Child", (), {"pid": 123, "poll": lambda self: None})()
            def popen(*args, **kwargs):
                kwargs["stdout"].write(b"placement-router: placement-aware routing active\n")
                return child
            def round_result(plan, prompts, executor, directory, phase, number):
                return {"phase": phase, "round": number, "elapsed_seconds": 1.0}
            with patch.object(m, "gpu_idle", side_effect=[{}, RuntimeError("gpu remained")]), \
                    patch.object(m, "cpu_import_identity", return_value={}), \
                    patch.object(m, "source_identity", side_effect=[{}, RuntimeError("source drift")]), \
                    patch.object(m.socket.socket, "connect_ex", return_value=1), \
                    patch.object(m.subprocess, "Popen", side_effect=popen), \
                    patch.object(m, "process_record", return_value={"pid": 123, "pgrp": 123, "start_ticks": "1"}), \
                    patch.object(m, "verify_prompts", return_value=[]), \
                    patch.object(m, "run_round", side_effect=round_result), \
                    patch.object(m, "stop_owned", side_effect=RuntimeError("cleanup failed")):
                with self.assertRaisesRegex(RuntimeError, "Launch failed"):
                    campaign.launch(0, "keep6p8", 0)
            value = m.read(root / "sweep00/position0-keep6p8/launch.json")
            self.assertFalse(value["complete"])
            self.assertEqual({v["step"] for v in value["finalization_errors"]},
                             {"owned_server_cleanup", "source_after", "gpu_after"})
            self.assertEqual(len(value["rounds"]), 9)

    def test_preflight_gate_rejects_duplicate_or_missing_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.json"
            m.save(plan, {})
            m.save(root / "preflight/receipt.json", {"complete": True, "cleanup_complete": True, "plan_sha256": m.sha(plan),
                "launches": [{"arm": "stock"}] * 5})
            campaign = object.__new__(m.Campaign)
            campaign.output, campaign.plan_path, campaign.state = root, plan, {}
            with self.assertRaisesRegex(RuntimeError, "5 unique"):
                campaign.preflight_gate()

    def run_local_round(self, failure=None):
        server = Server(("127.0.0.1", 0), Handler)
        server.received, server.fail_index, server.failure = [], 7 if failure else None, failure
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder, cf.ThreadPoolExecutor(max_workers=64) as executor:
                directory = Path(folder) / "measured00"
                plan = {"url": f"http://127.0.0.1:{server.server_port}", "request_timeout": 10}
                rows = [{"request_id": i, "token_ids": [i + 1] * 4096} for i in range(64)]
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "Round incomplete"):
                        m.run_round(plan, rows, executor, directory, "measured", 0)
                else:
                    m.run_round(plan, rows, executor, directory, "measured", 0)
                value = m.read(directory / "round.json")
                self.assertEqual(len(value["records"]), 64)
                self.assertEqual(sorted(server.received), list(range(64)))
                self.assertEqual(len(list(directory.glob("*.sse"))), 64)
                for record in value["records"]:
                    self.assertEqual(m.sha(directory / record["raw_file"]), record["raw_sha256"])
                    self.assertEqual(m.sha(directory / record["request_file"]), record["request_sha256"])
                    self.assertTrue((directory / record["raw_file"]).stat().st_size > 0)
                if failure:
                    self.assertFalse(value["complete"])
                    failed = [r for r in value["records"] if not r["complete"]]
                    self.assertEqual([r["request_id"] for r in failed], [7])
                else:
                    self.assertTrue(value["complete"])
                    self.assertTrue(all(r["cohort_ttft_ms"] >= r["ttft_ms"] > 0 for r in value["records"]))
                    self.assertTrue(all(r["request_start_ns"] >= value["cohort"]["start_ns"] for r in value["records"]))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_real_64_concurrent_http_streams(self):
        self.run_local_round()

    def test_bad_usage_retains_all64_raw_responses(self):
        self.run_local_round("usage")

    def test_http500_retains_error_body_and_all64_responses(self):
        self.run_local_round("http")

    def test_incomplete_sse_retains_all64_raw_responses(self):
        self.run_local_round("incomplete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
