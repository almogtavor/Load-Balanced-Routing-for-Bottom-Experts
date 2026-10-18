"""CPU guards for future-token exclusion, arbitrary LPs, handoff and retention."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import time
import io
import threading
import hybrid_driver
import hybrid_timing
import heldout_causal
from http.server import BaseHTTPRequestHandler, HTTPServer

import causal_probe as causal
import common_v2 as common
import hybrid_client as hybrid


class FollowupTests(unittest.TestCase):
    def test_exact_arbitrary_target_lp_not_sampled_token(self):
        choice = {"logprobs": {"top_logprobs": [{"token_id:12": -.2, "token_id:97": -8.5}],
                               "token_logprobs": [-.2]}}
        self.assertEqual(common.next_lp(choice, 97), -8.5)
        with self.assertRaises(RuntimeError):
            common.next_lp(choice, 98)

    def test_suffix_mutation_preserves_target_and_prefix(self):
        ids = [20, 30, 40, 50, 60]
        changed = causal.mutate_suffix(ids, 2)
        self.assertEqual(changed[:3], ids[:3])
        self.assertEqual(len(changed), len(ids))
        self.assertNotEqual(changed, ids)

    def test_repeated_suffix_still_changes_without_touching_target(self):
        ids = [20, 30, 13, 13, 13]
        changed = causal.mutate_suffix(ids, 2)
        self.assertEqual(changed[:3], ids[:3])
        self.assertNotEqual(changed, ids)

    def test_prefix_request_is_unconstrained_and_target_absent_by_position(self):
        ids, position = [17, 29, 91, 78, 63], 2
        body = common.base_body("pinned", ids[:position], "test")
        body.update(logprobs=0, logprob_token_ids=[ids[position]], return_tokens_as_token_ids=True)
        self.assertEqual(body["prompt"], [17, 29])
        self.assertNotIn("allowed_token_ids", body)
        self.assertNotIn("logit_bias", body)
        self.assertNotIn("prompt_logprobs", body)

    def test_transfer_rejects_empty_blocks_and_wrong_mode(self):
        params = {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": "p",
                  "remote_request_id": "req", "remote_host": "127.0.0.1", "remote_port": 5600,
                  "remote_block_ids": [[1, 2]], "remote_num_tokens": 17, "tp_size": 1, "pp_size": 1,
                  "transfer_mode": "pull"}
        self.assertEqual(hybrid.transfer_params({"kv_transfer_params": params}, 17), params)
        for key, value in (("remote_block_ids", [[]]), ("transfer_mode", "push"), ("remote_num_tokens", 0)):
            changed = {**params, key: value}
            with self.assertRaises(RuntimeError):
                hybrid.transfer_params({"kv_transfer_params": changed}, 17)

    def test_producer_does_not_force_target_or_stop_early(self):
        body = common.base_body("pinned", [17, 29], "test", 2048)
        body.update(logprobs=0, logprob_token_ids=[91], stop=["Q:"])
        params = {"remote_engine_id": "native"}
        p, d = hybrid.paired_body(body, params)
        self.assertEqual(p["max_tokens"], 1)
        self.assertNotIn("logprob_token_ids", p)
        self.assertNotIn("stop", p)
        self.assertEqual(d["max_tokens"], 2048)
        self.assertEqual(d["kv_transfer_params"], params)
        self.assertNotIn("kv_transfer_params", body)

    def test_missing_arbitrary_target_fails_no_greedy_substitution(self):
        with self.assertRaises(RuntimeError):
            common.next_lp({"logprobs": {"top_logprobs": [{"token_id:1": -.2}]}}, 2)

    def test_incomplete_http_preserves_partial_raw_body(self):
        class Response:
            status = 200
            headers = {"Content-Length": "200"}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): raise http.client.IncompleteRead(b'{"partial":', 188)
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "r"
            with patch("urllib.request.urlopen", return_value=Response()):
                with self.assertRaises(http.client.IncompleteRead):
                    common.post("http://unused", common.base_body("m", [1], "r"), stem)
            self.assertEqual(stem.with_suffix(".response.json").read_bytes(), b'{"partial":')
            receipt = json.loads(stem.with_suffix(".error.json").read_bytes())
            self.assertFalse(receipt["response_complete"])
            self.assertEqual(receipt["response_bytes"], 11)

    def record_fixture(self):
        row = {"id": "train:0", "continuation_start": 2, "continuation_tokens": 2, "normalization_chars": 5,
               "token_ids": [11, 22, 33, 44]}
        spec = {"rows": [row], "teacher_rows": [{**row, "probe_id": "train:0:replica0"}],
                "orders": ["original"], "cohorts": [1]}
        records = [{"kind": "singleton", "id": row["id"], "repeat": i, "token_logprobs": [-1, -2]} for i in range(2)]
        records += [{"kind": "prefix", "id": row["id"], "position": j, "logprob": -1} for j in (2, 3)]
        records += [{"kind": "suffix", "id": row["id"], "position": 2, "logprob": -1}]
        records += [{"kind": "schedule", "order": "original", "cohort": 1,
                     "id": "train:0:replica0", "source_id": row["id"], "token_logprobs": [-1, -2]}]
        return spec, records

    def test_exact_record_completeness_accepts_valid_fixture(self):
        spec, records = self.record_fixture()
        self.assertTrue(causal.validate_records(spec, records))

    def test_record_missing_or_duplicate_rejected(self):
        spec, records = self.record_fixture()
        for changed in (records[:-1], records + [records[-1]]):
            with self.assertRaises(RuntimeError):
                causal.validate_records(spec, changed)

    def test_wrong_vector_length_cannot_zip_truncate(self):
        spec, records = self.record_fixture()
        records[0]["token_logprobs"] = [-1]
        with self.assertRaises(RuntimeError):
            causal.validate_records(spec, records)

    def test_unknown_or_nonfinite_record_rejected(self):
        spec, records = self.record_fixture()
        records[0]["token_logprobs"][0] = float("nan")
        with self.assertRaises(RuntimeError):
            causal.validate_records(spec, records)
        _, records = self.record_fixture()
        records.append({"kind": "invented"})
        with self.assertRaises(RuntimeError):
            causal.validate_records(spec, records)

    def test_actual_request_file_roundtrip_and_independent_rescore(self):
        """Exercise the entire request writer/parser/closure path without GPUs."""
        spec, _ = self.record_fixture()
        spec.update(harness_sha256=common.file_sha(causal.__file__),
                    common_sha256=common.file_sha(Path(common.__file__)),
                    checkpoint_proof_sha256="checkpoint", source_proof_sha256="source",
                    prefix_score_scope="unit fixture")
        class Response:
            status, headers = 200, {}
            def __init__(self, request): self.request = request
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                body = json.loads(self.request.data)
                prompts = body["prompt"] if isinstance(body["prompt"][0], list) else [body["prompt"]]
                choices = []
                for i, ids in enumerate(prompts):
                    choice = {"index": i, "prompt_token_ids": ids, "token_ids": [55], "text": "unit", "finish_reason": "length"}
                    if "prompt_logprobs" in body:
                        choice["prompt_logprobs"] = [None] + [{str(t): {"logprob": -1.0}} for t in ids[1:]]
                    if "logprob_token_ids" in body:
                        choice["logprobs"] = {"top_logprobs": [{f"token_id:{t}": -1.0 for t in body["logprob_token_ids"]}]}
                    choices.append(choice)
                return json.dumps({"choices": choices}).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared.mkdir()
            common.write_new(prepared / "protocol.json", spec)
            contract = root / "contract.json"
            common.write_new(contract, {"complete": True, "source_commit": common.SOURCE_COMMIT,
                "checkpoint_revision": common.REVISION, "arm": "stock", "chunking_label": "budget16384",
                "checkpoint_proof_sha256": "checkpoint", "source_proof_sha256": "source", "prefix_caching": False})
            args = SimpleNamespace(prepared=prepared, server_contract=contract, output=root / "run", arm="stock",
                chunking_label="budget16384", stop_unix=time.time() + 60, max_requests=100, url="http://unused",
                served_model="unit", launch_id="cpu", timeout=5)
            with patch("urllib.request.urlopen", side_effect=lambda request, **kw: Response(request)):
                causal.run(args)
            causal.analyze(SimpleNamespace(prepared=prepared, result=args.output, output=root / "analysis.json"))
            self.assertTrue(common.read(root / "analysis.json")["complete"])
            # Mutation cannot be hidden by editing only the aggregate score.
            records = common.read(args.output / "records.json")
            records[0]["token_logprobs"][0] = -2
            (args.output / "records.json").write_bytes(common.encoded(records))
            with self.assertRaises(RuntimeError):
                causal.analyze(SimpleNamespace(prepared=prepared, result=args.output, output=root / "bad.json"))

    def test_clean_hybrid_environments_preserve_stock_decode(self):
        plan = {"site_root": "/frozen/site", "phase_probe": "/frozen/probe.py"}
        with patch.dict("os.environ", {"PLACEMENT_KEEP": "4", "PLACEMENT_PROBE": "bad", "VLLM_TORCH_PROFILER_DIR": "bad"}):
            d = hybrid_driver.environment(plan, "decode", "keep6p8")
            p = hybrid_driver.environment(plan, "prefill", "keep6p8")
        self.assertNotIn("PLACEMENT_KEEP", d)
        self.assertNotIn("PLACEMENT_PROBE", d)
        self.assertNotIn("VLLM_TORCH_PROFILER_DIR", d)
        self.assertEqual(p["PLACEMENT_KEEP"], "6")
        self.assertEqual(d["CUDA_VISIBLE_DEVICES"], "4,5,6,7")
        self.assertNotEqual(p["VLLM_NIXL_SIDE_CHANNEL_PORT"], d["VLLM_NIXL_SIDE_CHANNEL_PORT"])

    def test_hybrid_commands_fail_on_load_and_report_native_cached_tokens(self):
        plan = {"vllm": "/frozen/vllm", "model": "/pinned/model", "gpu_util": .88}
        p, d = hybrid_driver.commands(plan, "prefill"), hybrid_driver.commands(plan, "decode")
        self.assertIn("--enable-prompt-tokens-details", d)
        self.assertIn("--no-enable-prefix-caching", d)
        self.assertEqual(json.loads(d[-1])["kv_role"], "kv_consumer")
        self.assertEqual(json.loads(p[-1])["kv_load_failure_policy"], "fail")
        self.assertNotIn("--disable-log-stats", p)

    def timed_fixture(self, cached_tokens=1024):
        module = hybrid_driver.lifecycle(Path(__file__).parents[1] / "performance" / "prefill_e2e.py")
        row = {"request_id": 0, "token_ids": [33] * 1024}
        body = common.base_body("task-model", row["token_ids"], "cpu-stream", 128)
        body["ignore_eos"] = True
        producer_body, _ = hybrid.paired_body(body, None)
        params = {"do_remote_prefill": True, "do_remote_decode": False, "remote_engine_id": "p",
                  "remote_request_id": "cmpl-cpu-stream-0", "remote_host": "127.0.0.1", "remote_port": 5600,
                  "remote_block_ids": [[1, 2]], "remote_num_tokens": 1024, "tp_size": 1, "pp_size": 1,
                  "transfer_mode": "pull"}
        p_response = {"id": "cmpl-cpu-stream", "kv_transfer_params": params,
                      "usage": {"prompt_tokens": 1024, "completion_tokens": 1}, "choices": [
                          {"index": 0, "prompt_token_ids": row["token_ids"], "token_ids": [55], "text": "x", "finish_reason": "length"}]}
        chunks = [{"choices": [{"index": 0, "prompt_token_ids": row["token_ids"], "token_ids": [55] * 128,
                                "text": "x" * 128, "finish_reason": "length"}]},
                  {"choices": [], "usage": {"prompt_tokens": 1024, "completion_tokens": 128,
                                             "prompt_tokens_details": {"cached_tokens": cached_tokens}}}]
        wire = b"".join(b"data: " + common.encoded(chunk) + b"\n\n" for chunk in chunks) + b"data: [DONE]\n\n"
        class Response:
            status = 200
            headers = {"Content-Type": "text/event-stream"}
            def __init__(self, data):
                self.data = io.BytesIO(data)
                self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda x: None)))
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read1(self, n): return self.data.read(n)
            def readline(self): return self.data.readline()
        plan = {"request_timeout": 5, "stop_unix": time.time() + 60, "prefill_url": "http://prefill",
                "decode_url": "http://decode", "output_tokens": 128}
        calls = []
        def fake(request, **kwargs):
            calls.append(request)
            return Response(common.encoded(p_response) if len(calls) == 1 else wire)
        with patch("urllib.request.urlopen", side_effect=fake):
            value = hybrid_timing.stream_handoff(plan, row, body, common.encoded(producer_body), module.parse_event,
                threading.Barrier(1), {"start_ns": time.perf_counter_ns()})
        return value, calls, params

    def test_real_two_leg_streaming_roundtrip_requires_native_cached_coverage(self):
        (record, raw_p, raw_d, wire), calls, params = self.timed_fixture()
        self.assertTrue(record["complete"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(calls[1].data)["kv_transfer_params"], params)
        self.assertTrue(raw_p and raw_d and wire)
        self.assertGreaterEqual(record["ttft_ms"], 0)

    def test_missing_native_cached_coverage_rejects_speed_claim_but_retains_raw(self):
        (record, raw_p, raw_d, wire), calls, params = self.timed_fixture(cached_tokens=0)
        self.assertFalse(record["complete"])
        self.assertIn("cached prompt coverage", record["error"])
        self.assertTrue(raw_p and raw_d and wire)

    def test_real_http_content_length_and_sse_both_legs(self):
        # Real CPython HTTPResponse exercises fp/Content-Length closure semantics
        # that a fake response cannot establish.
        (_, raw_p, raw_d, _), _, _ = self.timed_fixture()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                is_prefill = self.path.startswith("/p/")
                body = raw_p if is_prefill else raw_d
                self.send_response(200)
                self.send_header("Content-Type", "application/json" if is_prefill else "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            module = hybrid_driver.lifecycle(Path(__file__).parents[1] / "performance" / "prefill_e2e.py")
            row = {"request_id": 0, "token_ids": [33] * 1024}
            body = common.base_body("task-model", row["token_ids"], "cpu-stream", 128)
            body["ignore_eos"] = True
            p_body, _ = hybrid.paired_body(body, None)
            url = f"http://127.0.0.1:{server.server_address[1]}"
            plan = {"prefill_url": url + "/p", "decode_url": url + "/d", "request_timeout": 5,
                    "stop_unix": time.time() + 60, "output_tokens": 128}
            value, p_raw, d_raw, _ = hybrid_timing.stream_handoff(plan, row, body, common.encoded(p_body), module.parse_event,
                threading.Barrier(1), {"start_ns": time.perf_counter_ns()})
            self.assertTrue(value["complete"], value.get("error"))
            self.assertEqual(p_raw, raw_p)
            self.assertEqual(d_raw, raw_d)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_failed_native_pair_always_writes_immutable_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            driver = object.__new__(hybrid_driver.Driver)
            driver.root = Path(directory)
            driver.path = Path(directory) / "plan.json"
            common.write_new(driver.path, {})
            driver.mode, driver.plan = "timed", {}
            driver.state = {"launches": []}
            driver.validate = lambda: None
            driver.save = lambda: None
            driver.close_pair = lambda: {"all_owned_closed": True}
            driver.start_pair = lambda *args: (_ for _ in ()).throw(RuntimeError("original startup failure"))
            with self.assertRaises(RuntimeError):
                driver.launch(0, 0, "stock")
            receipt = common.read(driver.root / "sweep00-position0-stock" / "launch.json")
            self.assertFalse(receipt["complete"])
            self.assertIn("original startup failure", receipt["workload_error"])
            self.assertEqual(len(driver.state["launches"]), 1)
            self.assertFalse(driver.state["launches"][0]["complete"])

    def test_heldout_selection_is_fixed_unique_and_label_independent(self):
        indices = heldout_causal.selected_indices()
        self.assertEqual(len(indices), 256)
        self.assertEqual(len(set(indices)), 256)
        self.assertEqual(indices, sorted(indices))
        self.assertTrue(all(0 <= i < 10042 for i in indices))


if __name__ == "__main__":
    unittest.main()
