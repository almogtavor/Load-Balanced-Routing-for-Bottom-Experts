"""CPU-only protocol checks: python -m unittest discover -s experiments -p test_serve_profile.py."""
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import serve_profile as runner


def stream(chunks, usage, done=True):
    events = [{"choices": [{"text": "x" * n,
                            "token_ids": [42] * n, "finish_reason": None}]}
              for n in chunks]
    if usage is not None:
        events.append({"choices": [], "usage": usage})
    body = "".join("data: " + json.dumps(e) + "\r\n\r\n" for e in events)
    if done:
        body += "data: [DONE]\r\n\r\n"
    return io.BytesIO(body.encode())


class StreamingTests(unittest.TestCase):
    def request(self, chunks, *, done=True, usage=None):
        usage = usage or {"prompt_tokens": 2, "completion_tokens": 3}
        with patch("urllib.request.urlopen", return_value=stream(chunks, usage, done)):
            return runner.stream_request("http://localhost", "model", {"token_ids": [1, 2],
                                           "request_id": 7}, 3, 1, 0)

    def test_sse_comments_crlf_and_multiline(self):
        body = io.BytesIO(b": heartbeat\r\n\r\ndata: {\r\ndata: \"x\": 1}\r\n\r\ndata: [DONE]\r\n\r\n")
        self.assertEqual(list(runner.sse_events(body)), ['{\n"x": 1}', "[DONE]"])

    def test_single_token_chunks_have_itl(self):
        result = self.request([1, 1, 1])
        self.assertTrue(result["all_chunks_single_token"])
        self.assertEqual(len(result["itl_ms"]), 2)
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertGreaterEqual(result["latency_ms"], result["ttft_ms"])

    def test_multi_token_chunks_do_not_invent_itl(self):
        result = self.request([2, 1])
        self.assertIsNone(result["itl_ms"])
        self.assertEqual(len(result["inter_chunk_ms"]), 1)
        self.assertIsNotNone(result["tpot_ms"])

    def test_incomplete_stream_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Incomplete stream"):
            self.request([1, 1, 1], done=False)

    def test_token_budget_or_stream_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "counts disagree"):
            self.request([1, 1])
        with self.assertRaisesRegex(RuntimeError, "Token budget mismatch"):
            self.request([1, 1, 1], usage={"prompt_tokens": 1, "completion_tokens": 3})

    def test_corpus_windows_are_reproducible_and_exact_length(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return list(text.encode())
            def decode(self, ids, **kwargs):
                return bytes(ids).decode()
        shape = runner.Shape("tiny", 150, 3, 2, 4)
        source = "The library opens early and closes before sunset. " * 100
        first = runner.make_prompts(Tokenizer(), [shape], 33, source)
        second = runner.make_prompts(Tokenizer(), [shape], 33, source)
        self.assertEqual(first, second)
        self.assertEqual({len(p["token_ids"]) for p in first["tiny"]}, {150})
        self.assertGreater(len({p["corpus_token_offset"] for p in first["tiny"]}), 1)

    def test_probe_and_inherited_router_are_removed(self):
        with patch.dict(os.environ, {"PLACEMENT_PROBE": "probe.py", "PLACEMENT_KEEP": "4",
                                     "PLACEMENT_LOAD_PENALTY": "0.1", "VLLM_TORCH_PROFILER_DIR": "/old"}):
            stock = runner.arm_environment("stock")
            self.assertNotIn("PLACEMENT_PROBE", stock)
            self.assertNotIn("PLACEMENT_KEEP", stock)
            self.assertNotIn("VLLM_TORCH_PROFILER_DIR", stock)
            balanced = runner.arm_environment("balanced6p1")
            self.assertEqual(balanced["PLACEMENT_KEEP"], "6")
            self.assertEqual(balanced["PLACEMENT_LOAD_PENALTY"], "1")


class TraceTests(unittest.TestCase):
    def write_trace(self, directory, rank, include_cuda=True):
        data = {"traceEvents": [{"cat": "cpu_op", "ph": "X", "name": "aten::linear", "dur": 3}]}
        if include_cuda:
            data["traceEvents"] += [{"cat": "kernel", "ph": "X", "name": name, "dur": 5}
                                    for name in ("fused_topk", "deepep::dispatch", "cutlass_gemm", "deepep::combine")]
        path = directory / f"dp{rank}_pp0_tp0_ep{rank}_rank0.123.pt.trace.json.gz"
        with gzip.open(path, "wt") as f:
            json.dump(data, f)

    def test_cpu_trace_cannot_satisfy_gpu_rank_validation(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            self.write_trace(directory, 0)
            self.write_trace(directory, 1, include_cuda=False)
            summary = runner.summarize_traces(directory, 2)
            self.assertFalse(summary["valid_all_dp_ranks_cpu_and_cuda"])
            self.assertEqual(summary["observed_cuda_dp_ranks"], [0])
            self.write_trace(directory, 1)
            summary = runner.summarize_traces(directory, 2)
            self.assertTrue(summary["valid_all_dp_ranks_cpu_and_cuda"])
            self.assertEqual(summary["traces"][0]["categories"]["dispatch"]["calls"], 1)


if __name__ == "__main__":
    unittest.main()
