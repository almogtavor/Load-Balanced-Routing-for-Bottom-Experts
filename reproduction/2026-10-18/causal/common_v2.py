"""Dependency-free, fail-closed request preservation shared by follow-up probes."""
from __future__ import annotations

import hashlib
import http.client
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
SOURCE_COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
SOURCE = {
    "src/placement_router/kernel.py": "06bc0d28295d23d4f0ae5715a4ffa555bb8d986118af3f52fc5fdc4cf0432854",
    "src/placement_router/plugin.py": "c432fc5a35793afb8b721806656de651bc755192000bf8ad9c9eead5084cffd8",
    "src/placement_router/vendor/fused_topk_router.py": "eecf00dffc63e72289032ece5a4634e45d97a1a1ff860901ba54b6b826e5e5fc",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value) + b"\n")


def base_body(model, prompt, request_id, max_tokens=1):
    require(prompt and all(type(x) is int and x >= 0 for x in prompt), "Invalid token prefix")
    return {"model": model, "prompt": prompt, "request_id": request_id,
            "temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0,
            "repetition_penalty": 1, "frequency_penalty": 0, "presence_penalty": 0,
            "seed": 0, "max_tokens": max_tokens, "ignore_eos": False,
            "stream": False, "n": 1, "return_token_ids": True,
            "add_special_tokens": False}


def post(url, body, stem, *, timeout=900, headers=None):
    """Never retries. Preserve raw bytes before any JSON/schema validation."""
    stem = Path(stem)
    wire = encoded(body)
    started = time.time_ns()
    monotonic = time.perf_counter_ns()
    write_new(stem.with_suffix(".request.json"), {"body": body, "sha256": sha(wire),
              "headers": headers or {}, "submitted_unix_ns": started})
    request = urllib.request.Request(url.rstrip("/") + "/v1/completions", wire,
                                    {"Content-Type": "application/json", **(headers or {})})
    status = None
    raw = b""
    response_headers = {}
    response_complete = False
    try:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status, response_headers = response.status, dict(response.headers)
                try:
                    raw = response.read()
                except http.client.IncompleteRead as error:
                    raw = error.partial
                    raise
                response_complete = True
        except urllib.error.HTTPError as error:
            status, response_headers = error.code, dict(error.headers)
            try:
                raw = error.read()
            except http.client.IncompleteRead as incomplete:
                raw = incomplete.partial
                raise
            response_complete = True
        with stem.with_suffix(".response.json").open("xb") as stream:
            stream.write(raw)
        evidence = {"http_status": status, "response_sha256": sha(raw), "response_bytes": len(raw),
                    "response_headers": response_headers, "response_complete": response_complete,
                    "submitted_unix_ns": started, "returned_unix_ns": time.time_ns(),
                    "elapsed_ns": time.perf_counter_ns() - monotonic}
        write_new(stem.with_suffix(".transport.json"), evidence)
        require(status == 200, f"HTTP {status}: inspect preserved response")
        value = json.loads(raw)
        require("error" not in value, "Server error in HTTP200 response")
        return value, evidence
    except BaseException as error:
        if not stem.with_suffix(".response.json").exists():
            with stem.with_suffix(".response.json").open("xb") as stream:
                stream.write(raw)
        write_new(stem.with_suffix(".error.json"), {"complete": False,
                  "error_type": type(error).__name__, "error": str(error), "http_status": status,
                  "response_complete": response_complete, "response_headers": response_headers,
                  "response_bytes": len(raw), "response_sha256": sha(raw)})
        raise


def choices(response, prompts):
    values = response.get("choices", [])
    require(len(values) == len(prompts) and {x.get("index") for x in values} == set(range(len(prompts))),
            "Missing/duplicate response choices")
    result = sorted(values, key=lambda x: x["index"])
    for choice, prompt in zip(result, prompts):
        require(choice.get("prompt_token_ids") == prompt, "Server consumed different prompt token IDs")
        require(choice.get("finish_reason") in ("stop", "length"), "Abnormal finish reason")
        require(isinstance(choice.get("token_ids"), list), "Missing generated token IDs")
    return result


def finite_lp(value):
    require(type(value) in (int, float) and math.isfinite(value) and -9998 < value <= 1e-6,
            "Missing/nonfinite/clamped logprob")
    return float(value)


def prompt_lp(choice, token_ids, position):
    require(0 < position < len(token_ids), "Scored token needs preceding context")
    values = choice.get("prompt_logprobs")
    require(isinstance(values, list) and len(values) == len(token_ids), "Prompt LP alignment differs")
    item = values[position]
    require(isinstance(item, dict) and str(token_ids[position]) in item, "Actual token LP absent")
    return finite_lp(item[str(token_ids[position])]["logprob"])


def next_lp(choice, target):
    values = (choice.get("logprobs") or {}).get("top_logprobs")
    require(isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict),
            "Expected exactly one generated-position logprob map")
    key = "token_id:" + str(target)
    require(key in values[0], "Requested target token LP absent; do not substitute sampled-token LP")
    return finite_lp(values[0][key])
