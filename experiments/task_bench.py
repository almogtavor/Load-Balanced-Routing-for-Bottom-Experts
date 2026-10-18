#!/usr/bin/env python3
"""Frozen, item-paired GSM8K/HellaSwag accuracy through a running vLLM server.

This module never starts a model or changes a cluster. ``prepare`` uses only the
CPU tokenizer; ``preflight`` uses training inputs; ``score`` evaluates
one fresh arm. The caller owns server lifetime, source deployment and profiling.
All HTTP responses are saved before validation. A failed cohort is never retried
or silently omitted. One HTTP batch contains a fixed cohort of 256 ordered prompts;
cohorts do not overlap. This fixes client inputs, not internal DP/tile membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import time
import urllib.error
import urllib.request
import uuid

MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
CONFIG_SHA256 = "a1ee086a68d0cbfc87316da00ba4b8507bd1292978108e2496201a30a450f438"
COMMIT = "57c67bbe4228313a684d15815a684209f56e4de0"
ARMS = {"stock": (None, None), "keep8": (8, 0.), "plain6": (6, 0.),
        "keep6p8": (6, 8.), "keep4p8": (4, 8.), "stock_repeat": (None, None)}
COUNTS = {"gsm8k": 1319, "hellaswag": 10042}
COHORT = 256
SOURCE_PATHS = ("src/placement_router/kernel.py", "src/placement_router/plugin.py",
                "src/placement_router/vendor/fused_topk_router.py")
STOPS = ["Q:", "</s>", "<|im_end|>"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while part := f.read(8 << 20):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def own_directory(path):
    """Exclusive result creation also prevents resume/duplicate item execution."""
    Path(path).mkdir(parents=True, exist_ok=False)


def normalize_number(text):
    # lm_eval GSM8K CoT exact-match ignore_regexes: commas, dollars, final dot.
    return re.sub(r"\.$", "", str(text).strip().replace(",", "").replace("$", "")).lower()


def gsm_gold(answer):
    match = re.search(r"####\s*(.*?)\s*$", answer, re.S)
    require(match is not None, "GSM8K gold has no #### final answer")
    return normalize_number(match.group(1))


def gsm_answers(text):
    """Pinned gsm8k-cot v3 strict and flexible filters; absence counts wrong."""
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+).", text)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    # RegexFilter takes the first strict match; flexible configuration selects last.
    return {"strict": normalize_number(strict[0]) if strict else None,
            "flexible": normalize_number(next(v for v in flexible[-1] if v)) if flexible else None}


def gsm_prompt(question, examples, tokenizer):
    require(len(examples) == 8, "Exactly eight fixed exemplars are required")
    text = "\n\n".join("Q: " + s["question"] + "\nA: " + s["target"] for s in examples)
    text += "\n\nQ: " + question + "\nA:"
    # A single user message reproduces apply_chat_template with fewshot_as_multiturn=False.
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True, return_dict=False)
    require(isinstance(ids, list) and ids and all(type(x) is int for x in ids), "Invalid chat token IDs")
    return ids


def hs_preprocess(text):
    """EleutherAI HellaSwag utils.py v1.0, kept byte-for-byte in input provenance."""
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def hs_document(doc):
    context = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return hs_preprocess(doc["activity_label"] + ": " + context), [hs_preprocess(x) for x in doc["endings"]]


def encode_pair(tokenizer, context, continuation):
    # lm_eval TemplateLM._encode_pair moves trailing context whitespace before joint encoding.
    n_space = len(context) - len(context.rstrip())
    if n_space:
        continuation = context[-n_space:] + continuation
        context = context[:-n_space]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    require(context_ids and ids[:len(context_ids)] == context_ids,
            "Tokenizer boundary changed context prefix; refusing an ambiguous likelihood span")
    require(len(ids) > len(context_ids), "Empty HellaSwag ending token span")
    return ids, len(context_ids)


def make_gsm(rows, examples, tokenizer, split):
    return [{"id": f"gsm8k:{split}:{i}", "item_index": i, "task": "gsm8k", "split": split,
             "token_ids": gsm_prompt(row["question"], examples, tokenizer), "gold": gsm_gold(row["answer"])}
            for i, row in enumerate(rows)]


def make_hs(rows, tokenizer, split="validation"):
    result = []
    for i, row in enumerate(rows):
        context, endings = hs_document(row)
        require(len(endings) == 4 and int(row["label"]) in range(4), "Invalid HellaSwag choices/label")
        for choice, ending in enumerate(endings):
            require(len(ending) > 0, "Empty preprocessed HellaSwag ending")
            ids, start = encode_pair(tokenizer, context, " " + ending)
            result.append({"id": f"hellaswag:{split}:{i}:{choice}", "item_index": i,
                           "source_ind": str(row.get("ind", i)), "source_id": str(row.get("source_id", "")),
                           "source_row_sha256": sha(canonical(row)), "task": "hellaswag", "split": split,
                           "choice": choice, "gold": int(row["label"]), "token_ids": ids,
                           "continuation_start": start, "continuation_tokens": len(ids) - start,
                           # Standard lm_eval acc_norm denominator is ending string length, not tokens.
                           "normalization_chars": len(ending)})
    return result


def validate_hs_identity(rows):
    # HellaSwag `ind` is NOT globally unique across source datasets. Preserve all
    # source rows; compound metadata and the frozen source row index identify them.
    identities = [(str(row["ind"]), str(row["source_id"])) for row in rows]
    require(len(set(identities)) == len(rows), "Duplicate compound HellaSwag source identity")


def check_datasets(folder):
    folder = Path(folder)
    manifest = read(folder / "manifest.json")
    entries = {f["path"]: f for f in manifest["files"]}
    needed = ("gsm8k_test.jsonl", "gsm8k_train.jsonl", "gsm8k_cot_8shot.json", "hellaswag_val.jsonl",
              "hellaswag_train_preflight4.jsonl")
    for name in needed:
        require(name in entries and digest(folder / name) == entries[name]["sha256"], "Dataset hash mismatch: " + name)
    require(entries["gsm8k_test.jsonl"]["records"] == 1319 and entries["hellaswag_val.jsonl"]["records"] == 10042,
            "Dataset manifest is not the full evaluation splits")
    return manifest


def source_identity(proof_path, production_root):
    proof = read(proof_path)
    require(proof.get("source_commit", proof.get("base_commit")) == COMMIT, "Expected latest public tile16 source commit")
    from task_bench_probe import SOURCE
    installed = proof.get("installed_package", {}).get("installed_source_sha256")
    if installed is not None:
        require(all(installed.get(name) == SOURCE[name] for name in SOURCE), "Deployment installed-source identity differs")
        require(Path(proof["source_root"]).resolve() == Path(production_root).resolve(), "Deployment source root differs")
    actual = {}
    for name in SOURCE_PATHS:
        expected = SOURCE[name]
        if "files" in proof:
            require(proof["files"][name]["sha256"] == expected, "Public-source proof hash differs")
        actual[name] = digest(Path(production_root) / name)
        require(actual[name] == expected, "Production source differs from frozen public commit: " + name)
    return {"commit": COMMIT, "proof_sha256": digest(proof_path), "files": actual}


def checkpoint_identity(proof_path, model):
    proof = read(proof_path)
    require(proof.get("canonical_identity_verified") is True and proof.get("model_id") == MODEL_ID
            and proof.get("revision") == REVISION, "Canonical Instruct checkpoint proof is missing/incorrect")
    require(Path(proof["local_model_path"]).resolve() == Path(model).resolve(), "Proof and model paths differ")
    require(proof.get("canonical_config_sha256") == CONFIG_SHA256
            and digest(Path(model) / "config.json") == CONFIG_SHA256, "Pinned canonical model config hash mismatch")
    files = proof.get("safetensors", [])
    require(len(files) == 16 and len({x["filename"] for x in files}) == 16
            and proof.get("weights_verified") == files, "Root must verify exactly all16 distinct canonical model shards")
    signatures = {}
    for item in files:
        path = Path(model) / item["filename"]
        require(path.stat().st_size == item["bytes"], "Missing/truncated weight shard")
        signatures[item["filename"]] = [path.stat().st_size, path.stat().st_mtime_ns]
    return {"model_id": MODEL_ID, "revision": REVISION, "proof_sha256": digest(proof_path),
            "model_path": str(Path(model).resolve()), "file_stat_signatures": signatures,
            "verification": "Root full-shard SHA proof; wrapper checks sizes and freezes mtimes, does not reread61GB"}


def protocol(max_output_tokens=2048):
    return {"schema_version": 1, "tasks": ["gsm8k", "hellaswag"], "counts": COUNTS,
            "primary_ep": 4, "optional_separate_ep": 8, "cohort_size": COHORT,
            "request_schedule": "GSM8K source order then HellaSwag source item order/options0..3; one batchedHTTPrequest per256 prompts; no overlap/retries",
            "internal_batching": "Not controlled/equated: tile16 neighbors may differ with internalDP/scheduler, response length and run",
            "gsm8k": {"fewshot": 8, "prompt": "canonical Q/A eight-shot CoT in one user turn, no system message, chat template generation prefix",
                      "primary_metric": "flexible-extract exact match", "secondary_metric": "strict-match exact match",
                      "max_output_tokens": max_output_tokens, "temperature": 0, "seed": 0, "ignore_eos": False,
                      "stop": STOPS, "truncation": "All length-finished responses retained/scored; report counts; no data-driven max-token tuning"},
            "hellaswag": {"fewshot": 0, "prompt": "lm_eval v1.0 bare preprocessed context, no chat, noBOS, leading-space ending",
                          "scoring": "Sum actual-token prompt logprobs over jointly encoded continuation only",
                          "causality_caveat": "With nonzero tile16 load price, routing may depend on later teacher-forced tokens in the same tile. This is a batch-conditioned teacher-forced classification score, not strictly causal generative conditional loglikelihood. Stock/keep8/plain6 remain token-local. GSM8K uses actual autoregressive generation.",
                          "normalization": "Python character length of preprocessed ending excluding inserted delimiter",
                          "metric_comparability": "lm_eval preprocessing and raw/character-normalized argmax mechanics; this mechanical agreement does not make tile-priced scores causal model likelihoods. Held-out classification uses candidate text, never gold labels as model input.",
                          "max_output_tokens": 1, "generated_token": "Unscored; API generation requirement"},
            "arms": {a: {"keep": k, "load_penalty": p, "load_tile": 16 if k is not None else None} for a, (k, p) in ARMS.items()},
            "statistical_unit": "paired item conditional on one launch/arm; stock_repeat quantifies observed repeat variability; no launch-generalization/equivalence claim",
            "profile": "Optional separate stage only after every arm scored; train prompts; never pooled withaccuracy"}


def prepare(args):
    manifest = check_datasets(args.datasets)
    source = source_identity(args.source_proof, args.production_root)
    checkpoint = checkpoint_identity(args.checkpoint_proof, args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    datasets = Path(args.datasets)
    examples = read(datasets / "gsm8k_cot_8shot.json")
    gsm_rows, hs_rows = jsonl(datasets / "gsm8k_test.jsonl"), jsonl(datasets / "hellaswag_val.jsonl")
    require(len(gsm_rows) == COUNTS["gsm8k"] and len(hs_rows) == COUNTS["hellaswag"], "Wrong full split counts")
    validate_hs_identity(hs_rows)
    requests = make_gsm(gsm_rows, examples, tokenizer, "test") + make_hs(hs_rows, tokenizer)
    train = jsonl(datasets / "gsm8k_train.jsonl")[:8]
    train_questions = {s["question"] for s in examples}
    require(not any(x["question"] in train_questions for x in train), "Preflight overlaps fixed exemplars")
    preflight = make_gsm(train, examples, tokenizer, "train")
    hs_train = jsonl(datasets / "hellaswag_train_preflight4.jsonl")
    require(len(hs_train) == 4, "Require frozen four HellaSwag training preflight items")
    preflight += make_hs(hs_train, tokenizer, "train-preflight")
    spec = protocol(args.max_output_tokens)
    maximum = max(len(r["token_ids"]) + (args.max_output_tokens if r["task"] == "gsm8k" else 1) for r in requests + preflight)
    require(maximum <= args.max_model_len, f"Prompt+output budget {maximum} exceeds max_model_len {args.max_model_len}")
    token_files = {p.name: digest(p) for p in Path(args.model).iterdir() if p.is_file()
                   and (p.name.startswith("tokenizer") or p.name in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"))}
    metadata = {"complete": True, "created_unix_ns": time.time_ns(), "protocol": spec,
                "protocol_sha256": sha(canonical(spec)), "dataset_manifest_sha256": digest(datasets / "manifest.json"),
                "dataset_manifest": manifest, "source": source, "checkpoint": checkpoint,
                "tokenizer_files_sha256": token_files, "chat_template_sha256": sha(str(tokenizer.chat_template).encode()),
                "prepared_by_sha256": digest(__file__), "max_model_len": args.max_model_len,
                "probe_sha256": digest(Path(__file__).with_name("task_bench_probe.py")),
                "max_prompt_plus_output_tokens": maximum, "requests_sha256": sha(canonical(requests)),
                "preflight_sha256": sha(canonical(preflight)), "requests": len(requests),
                "prompt_tokens": sum(len(r["token_ids"]) for r in requests),
                "hellaswag_scored_tokens": sum(r.get("continuation_tokens", 0) for r in requests)}
    own_directory(args.output)
    atomic(args.output / "requests.json", requests)
    atomic(args.output / "preflight.json", preflight)
    atomic(args.output / "preparation.json", metadata)
    print(json.dumps({"complete": True, "requests": len(requests), "max_prompt_plus_output_tokens": maximum,
                      "preparation_sha256": digest(args.output / "preparation.json")}))


def load_prepared(folder):
    folder = Path(folder)
    metadata = read(folder / "preparation.json")
    require(metadata.get("complete") is True and metadata["prepared_by_sha256"] == digest(__file__), "Preparation/harness changed")
    require(metadata.get("probe_sha256") == digest(Path(__file__).with_name("task_bench_probe.py")), "Prepared identity probe changed")
    requests, preflight = read(folder / "requests.json"), read(folder / "preflight.json")
    require(sha(canonical(requests)) == metadata["requests_sha256"] and sha(canonical(preflight)) == metadata["preflight_sha256"], "Prepared prompt hash mismatch")
    require(len(requests) == 1319 + 40168 and len({r["id"] for r in requests}) == len(requests), "Full request IDs/counts differ")
    return metadata, requests, preflight


def request_body(rows, model, max_output_tokens, request_id):
    task = rows[0]["task"]
    require(all(r["task"] == task for r in rows) and 0 < len(rows) <= COHORT, "Invalid cohort")
    body = {"model": model, "prompt": [r["token_ids"] for r in rows], "temperature": 0,
            "top_p": 1, "top_k": -1, "min_p": 0, "repetition_penalty": 1,
            "frequency_penalty": 0, "presence_penalty": 0, "seed": 0,
            "max_tokens": max_output_tokens if task == "gsm8k" else 1, "ignore_eos": False,
            "stream": False, "n": 1, "return_token_ids": True, "add_special_tokens": False,
            "request_id": request_id}
    if task == "gsm8k":
        body["stop"] = STOPS
    else:
        body["prompt_logprobs"] = 0
    return body


def score_choice(row, choice):
    require(choice.get("prompt_token_ids") == row["token_ids"], "Server prompt IDs differ from frozen tokens")
    require(choice.get("finish_reason") in ("stop", "length"), "Abnormal finish reason")
    ids = choice.get("token_ids")
    require(isinstance(ids, list) and all(type(x) is int for x in ids), "Generated token IDs absent")
    if row["task"] == "gsm8k":
        text = choice.get("text")
        require(isinstance(text, str), "No generated text")
        extracted = gsm_answers(text)
        return {"id": row["id"], "task": row["task"], "item_index": row["item_index"], "gold": row["gold"],
                "answers": extracted, "correct": extracted["flexible"] == row["gold"],
                "strict_correct": extracted["strict"] == row["gold"], "empty": not text.strip(),
                "parse_failed": extracted["flexible"] is None, "truncated": choice["finish_reason"] == "length",
                "finish_reason": choice["finish_reason"], "generated_tokens": len(ids)}
    values = choice.get("prompt_logprobs")
    require(isinstance(values, list) and len(values) == len(row["token_ids"]), "Missing/misaligned prompt logprobs")
    start = row["continuation_start"]
    require(0 < start < len(values), "Every target needs preceding context")
    lps = []
    for index in range(start, len(values)):
        entry = values[index]
        actual = str(row["token_ids"][index])
        require(isinstance(entry, dict), "Scored token has no prompt likelihood")
        item = entry.get(actual, entry.get(int(actual)))
        require(isinstance(item, dict) and "logprob" in item, "Actual token absent in logprob map")
        value = item["logprob"]
        require(type(value) in (int, float) and math.isfinite(value) and value <= 1e-6, "Nonfinite/positive logprob")
        lps.append(float(value))
    require(len(lps) == row["continuation_tokens"], "Wrong scored continuation length")
    total = math.fsum(lps)
    return {"id": row["id"], "task": row["task"], "item_index": row["item_index"], "choice": row["choice"],
            "gold": row["gold"], "loglikelihood": total, "normalized_loglikelihood": total / row["normalization_chars"],
            "normalization_chars": row["normalization_chars"], "scored_tokens": len(lps), "token_logprobs": lps}


def validate_response(rows, response):
    require("error" not in response, "vLLM returned an error")
    choices = response.get("choices", [])
    require(len(choices) == len(rows) and {c.get("index") for c in choices} == set(range(len(rows))), "Missing/duplicate choices")
    usage = response.get("usage", {})
    require(usage.get("prompt_tokens") == sum(len(r["token_ids"]) for r in rows), "Usage prompt token count mismatch")
    ordered = sorted(choices, key=lambda c: c["index"])
    require(usage.get("completion_tokens") == sum(len(c.get("token_ids", [])) for c in ordered), "Usage generated token count mismatch")
    return [score_choice(r, c) for r, c in zip(rows, ordered)]


def cohorts(rows):
    # A cohort never crosses tasks; HS4 options stay together since256 divisible4.
    for task in ("gsm8k", "hellaswag"):
        selected = [r for r in rows if r["task"] == task]
        for i in range(0, len(selected), COHORT):
            yield selected[i:i + COHORT]


def http_cohort(url, rows, model, output, max_output_tokens, timeout, request_id):
    body = request_body(rows, model, max_output_tokens, request_id)
    submitted = time.time_ns()
    envelope = {"request_id": request_id, "ids": [r["id"] for r in rows], "request_body_sha256": sha(canonical(body)),
                "client_submit_unix_ns": submitted, "client_response_unix_ns": None,
                "server_start_unix_ns": None, "server_start_note": "HTTP API does not expose server scheduling times"}
    atomic(output.with_suffix(".request.json"), {**envelope, "body": body})
    try:
        request = urllib.request.Request(url.rstrip("/") + "/v1/completions", canonical(body), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            raw = error.read()
            with output.with_suffix(".response.json").open("xb") as f:
                f.write(raw)
            envelope.update(client_response_unix_ns=time.time_ns(), response_sha256=sha(raw), response_bytes=len(raw), http_status=error.code)
            raise
        # Preserve exact response bytes even if JSON/schema/score validation fails.
        with output.with_suffix(".response.json").open("xb") as f:
            f.write(raw)
        envelope.update(client_response_unix_ns=time.time_ns(), response_sha256=sha(raw), response_bytes=len(raw))
        parsed = json.loads(raw)
        scored = validate_response(rows, parsed)
        require(all(len(c["token_ids"]) <= body["max_tokens"] for c in parsed["choices"]), "Output exceeds frozen generation cap")
        atomic(output.with_suffix(".scored.json"), {**envelope, "complete": True, "records": scored})
        return scored
    except BaseException as error:
        atomic(output.with_suffix(".error.json"), {**envelope, "complete": False, "error_type": type(error).__name__, "error": str(error)})
        raise


def rank_evidence(folder, arm, ep, source, checkpoint=None):
    values = [read(p) for p in Path(folder).glob("rank*_pid*.json")]
    require(len(values) == ep and {v.get("ep_rank") for v in values} == set(range(ep)), "Need one ready record per physicalEP rank")
    keep, penalty = ARMS[arm]
    for value in values:
        require(value.get("complete") is True and value.get("arm") == arm and value.get("ep_world_size") == ep,
                "Wrong/incomplete worker arm/world")
        require(value.get("probe_sha256") == digest(Path(__file__).with_name("task_bench_probe.py")), "Actual worker probe source differs")
        require(value.get("deployment_proof_sha256") == source["proof_sha256"], "Worker deployment proof differs")
        if checkpoint is not None:
            require(Path(value.get("model_path", "")).resolve() == Path(checkpoint["model_path"]).resolve(),
                    "Actual served model path differs from SHA-verified checkpoint")
        require(value.get("placement_keep") == keep and value.get("load_penalty") == penalty, "Worker routing parameters differ")
        require(value.get("load_tile") == 16 and value.get("source_sha256") == source["files"], "Wrong actual imported kernel/source")
        require(value.get("registered_layers") == 48 and value.get("observed_router_objects") == 48,
                "Worker lacks all48 model routers")
        require(value.get("physical_dp") == ep and value.get("tp") == 1 and value.get("backend") == "deepep_high_throughput"
                and value.get("graphs") == "NONE" and value.get("dtype") == "torch.bfloat16"
                and value.get("prefix_caching") is False and value.get("ubatching") is False
                and value.get("pp") == 1 and value.get("hidden_size") == 2048
                and value.get("enable_expert_parallel") is True and value.get("expert_placement_strategy") == "linear"
                and value.get("max_model_len") == 8192 and value.get("max_num_seqs") == 256
                and value.get("max_num_batched_tokens") == 16384 and value.get("seed") == 0,
                "Worker runtime protocol mismatch")
        require(value.get("all_routers_expected") is True, "Router fallback/unexpected semantic contract")
    return values


def run_stage(args, preflight_only):
    metadata, full, probes = load_prepared(args.prepared)
    require(args.ep in (4, 8), "Only physicalEP4/8 supported")
    # Recheck current model/source/proof instead of accepting an old preparation alone.
    require(source_identity(args.source_proof, args.production_root) == metadata["source"], "Source proof changed")
    require(checkpoint_identity(args.checkpoint_proof, args.model) == metadata["checkpoint"], "Checkpoint/proof stats changed")
    if not preflight_only:
        prior = read(args.preflight / "result.json")
        require(prior.get("complete") is True and prior.get("preflight_pass") is True and prior.get("arm") == args.arm
                and prior.get("ep") == args.ep and prior.get("preparation_sha256") == digest(args.prepared / "preparation.json"),
                "Full scoring requires same-arm passed preflight")
        before = rank_evidence(args.probe_dir, args.arm, args.ep, metadata["source"], metadata["checkpoint"])
        require(prior.get("worker_pids") == sorted(v["pid"] for v in before), "Preflight was on another server")
    own_directory(args.output)
    state = {"complete": False, "mode": "preflight" if preflight_only else "full", "arm": args.arm, "ep": args.ep,
             "started_unix_ns": time.time_ns(), "preparation_sha256": digest(args.prepared / "preparation.json"),
             "harness_sha256": digest(__file__), "preparation": metadata, "cohorts_completed": 0}
    atomic(args.output / "result.json", state)
    records = []
    try:
        for index, rows in enumerate(cohorts(probes if preflight_only else full)):
            request_id = f"task-{args.ep}-{args.arm}-{'preflight' if preflight_only else 'score'}-{index:05d}"
            records.extend(http_cohort(args.url, rows, args.served_model, args.output / f"cohort{index:05d}",
                                       metadata["protocol"]["gsm8k"]["max_output_tokens"], args.request_timeout, request_id))
            state["cohorts_completed"] += 1
            atomic(args.output / "result.json", state)
            print(f"cohort {index}: {rows[0]['task']} complete {len(records)} requests", flush=True)
        evidence = rank_evidence(args.probe_dir, args.arm, args.ep, metadata["source"], metadata["checkpoint"])
        state.update(worker_pids=sorted(v["pid"] for v in evidence), worker_evidence=evidence,
                     finished_unix_ns=time.time_ns(), records=records, requests_completed=len(records))
        if preflight_only:
            gsm = [r for r in records if r["task"] == "gsm8k"]
            # Training-only fail-closed protocol checks; no accuracy threshold/tuning.
            passed = (len(gsm) == 8 and all(not r["empty"] and not r["parse_failed"] and not r["truncated"] for r in gsm)
                      and len([r for r in records if r["task"] == "hellaswag"]) == 16)
            state.update(preflight_pass=passed, complete=passed,
                         preflight_note="Eight GSM training questions: natural stop, nonempty, parseable, within frozen budget; fourHS training items actual-token span checks. No test/validation accuracy tuning.")
            atomic(args.output / "result.json", state)
            require(passed, "Training preflight failed; do not launch full evaluation")
        else:
            require(len(records) == len(full), "Incomplete request set")
            state.update(complete=True, task_summary=summarize_records(records))
            atomic(args.output / "result.json", state)
    except BaseException as error:
        state.update(complete=False, finished_unix_ns=time.time_ns(), error_type=type(error).__name__, error=str(error))
        atomic(args.output / "result.json", state)
        raise


def wilson(hits):
    n, k = len(hits), sum(hits)
    require(n > 0, "No accuracy items")
    z, p = 1.959963984540054, k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"n": n, "correct": k, "accuracy": p, "ci95_wilson": [center - radius, center + radius]}


def item_scores(records):
    gsm = sorted((r for r in records if r["task"] == "gsm8k"), key=lambda r: r["item_index"])
    hs = {}
    for row in records:
        if row["task"] == "hellaswag":
            hs.setdefault(row["item_index"], []).append(row)
    hs_items = []
    for item, values in sorted(hs.items()):
        require(len(values) == 4 and {x["choice"] for x in values} == set(range(4)) and len({x["gold"] for x in values}) == 1,
                "Missing/repeated HellaSwag choice")
        values.sort(key=lambda x: x["choice"])
        raw = max(range(4), key=lambda c: values[c]["loglikelihood"])
        norm = max(range(4), key=lambda c: values[c]["normalized_loglikelihood"])
        hs_items.append({"item_index": item, "gold": values[0]["gold"], "prediction": raw,
                         "normalized_prediction": norm, "correct": raw == values[0]["gold"],
                         "normalized_correct": norm == values[0]["gold"]})
    return {"gsm8k": gsm, "hellaswag": hs_items}


def summarize_records(records):
    items = item_scores(records)
    require(len(items["gsm8k"]) == 1319 and len(items["hellaswag"]) == 10042, "Accuracy requires all items")
    require([x["item_index"] for x in items["gsm8k"]] == list(range(1319))
            and [x["item_index"] for x in items["hellaswag"]] == list(range(10042)), "Missing/repeated item indices")
    gsm = items["gsm8k"]
    return {"gsm8k": {**wilson([r["correct"] for r in gsm]),
                      "strict": wilson([r["strict_correct"] for r in gsm]),
                      "truncated": sum(r["truncated"] for r in gsm), "empty": sum(r["empty"] for r in gsm),
                      "parse_failed": sum(r["parse_failed"] for r in gsm),
                      "finish_reasons": dict(Counter(r["finish_reason"] for r in gsm))},
            "hellaswag": {**wilson([r["correct"] for r in items["hellaswag"]]),
                          "normalized": wilson([r["normalized_correct"] for r in items["hellaswag"]])}}


def paired(base, arm):
    require(len(base) == len(arm) and len(base) > 1, "Paired items do not match")
    delta = [int(a) - int(b) for b, a in zip(base, arm)]
    mean, se = statistics.mean(delta), statistics.stdev(delta) / math.sqrt(len(delta))
    gains = sum(not b and a for b, a in zip(base, arm))
    losses = sum(b and not a for b, a in zip(base, arm))
    m, small = gains + losses, min(gains, losses)
    # Stable exact two-sided binomial McNemar, even thousands of discordants.
    if m:
        logs = [math.lgamma(m + 1) - math.lgamma(i + 1) - math.lgamma(m - i + 1) - m * math.log(2)
                for i in range(small + 1)]
        high = max(logs)
        p = min(1., 2 * math.exp(high) * math.fsum(math.exp(x - high) for x in logs))
    else:
        p = 1.
    return {"n": len(delta), "delta_accuracy": mean, "paired_se": se,
            "ci95_paired_normal": [mean - 1.959963984540054 * se, mean + 1.959963984540054 * se],
            "gains": gains, "losses": losses, "discordant": m, "mcnemar_exact_two_sided_p": p,
            "scope": "Item-paired conditional on observed server launches; CI is not between-launch uncertainty"}


def analyze(args):
    results = [read(p) for p in args.results]
    by_arm = {}
    for result in results:
        require(result.get("complete") is True and result.get("mode") == "full", "Incomplete/full result required")
        require(result["arm"] not in by_arm, "Duplicate arm")
        by_arm[result["arm"]] = result
    require(set(by_arm) == set(ARMS), "Primary analysis requires all six predeclared arms")
    require(len({r["preparation_sha256"] for r in by_arm.values()}) == 1 and len({r["ep"] for r in by_arm.values()}) == 1,
            "Do not pool different inputs/protocols/physicalEPs")
    items = {a: item_scores(r["records"]) for a, r in by_arm.items()}
    summaries = {a: summarize_records(r["records"]) for a, r in by_arm.items()}
    comparisons = {}
    for name, task, key in (("gsm8k", "gsm8k", "correct"), ("hellaswag", "hellaswag", "correct"),
                            ("hellaswag_normalized", "hellaswag", "normalized_correct")):
        base = [r[key] for r in items["stock"][task]]
        comparisons[name] = {a: paired(base, [r[key] for r in items[a][task]]) for a in ARMS if a != "stock"}
    controls = {}
    for a in ("stock_repeat", "keep8"):
        controls[a] = {
            "gsm8k_extracted_answer_disagreements": sum(x["answers"]["flexible"] != y["answers"]["flexible"]
                                                        for x, y in zip(items["stock"]["gsm8k"], items[a]["gsm8k"])),
            "hellaswag_raw_prediction_disagreements": sum(x["prediction"] != y["prediction"]
                                                         for x, y in zip(items["stock"]["hellaswag"], items[a]["hellaswag"])),
            "hellaswag_normalized_prediction_disagreements": sum(x["normalized_prediction"] != y["normalized_prediction"]
                                                                for x, y in zip(items["stock"]["hellaswag"], items[a]["hellaswag"])),
            "interpretation": "Observed native repeat/plugin control; inspect alongside treatment differences; no equivalence threshold inferred"}
    value = {"complete": True, "ep": next(iter(by_arm.values()))["ep"], "summaries": summaries,
             "paired_vs_stock": comparisons, "input_result_sha256": {str(p): digest(p) for p in args.results},
             "control_diagnostics": controls,
             "claims": "New canonical Instruct-2507 evaluation of public tile16λ8 source. Does not establish exact historical paper checkpoint/prompt identity. HellaSwag λ8 results are teacher-forced batch-conditioned classification scores with future-token routing coupling, not strictly causal generative loglikelihood. No latency/speed claim.",
             "multiplicity": "Unadjusted per-comparison95% intervals and exact tests; report all predeclared contrasts. No significance filtering.",
             "controls": "Stock repeat and production keep8 are measured accuracy controls, not forced equal; no nonsignificance-as-equivalence inference."}
    require(not args.output.exists(), "Refusing to overwrite analysis")
    atomic(args.output, value)
    print(json.dumps({"complete": True, "arms": list(summaries), "output": str(args.output)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest="mode", required=True)
    prepare_p = subs.add_parser("prepare")
    prepare_p.add_argument("--datasets", type=Path, required=True)
    prepare_p.add_argument("--max-output-tokens", type=int, default=2048)
    prepare_p.add_argument("--max-model-len", type=int, default=8192)
    for mode in ("preflight", "score"):
        child = subs.add_parser(mode)
        child.add_argument("--prepared", type=Path, required=True)
        child.add_argument("--url", default="http://127.0.0.1:8000")
        child.add_argument("--served-model", default="task-model")
        child.add_argument("--arm", choices=ARMS, required=True)
        child.add_argument("--ep", type=int, choices=(4, 8), default=4)
        child.add_argument("--probe-dir", type=Path, required=True)
        child.add_argument("--request-timeout", type=int, default=1800)
        if mode == "score":
            child.add_argument("--preflight", type=Path, required=True)
    for name in ("prepare", "preflight", "score"):
        child = subs.choices[name]
        child.add_argument("--model", type=Path, required=True)
        child.add_argument("--checkpoint-proof", type=Path, required=True)
        child.add_argument("--source-proof", type=Path, required=True)
        child.add_argument("--production-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    analysis_p = subs.add_parser("analyze")
    analysis_p.add_argument("--results", nargs="+", type=Path, required=True)
    analysis_p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "prepare":
        require(args.max_output_tokens == 2048 and args.max_model_len == 8192, "Frozen protocol uses max_output2048/model8192")
        prepare(args)
    elif args.mode in ("preflight", "score"):
        run_stage(args, args.mode == "preflight")
    else:
        analyze(args)


if __name__ == "__main__":
    main()
