#!/usr/bin/env python3
"""Prepare/run a bounded paper-style Qwen quality check; no cluster operations.

The Table2 source JSONs name an unavailable multilingual stream. Historical mode
therefore requires its independently recovered SHA256. Wikitext is an explicit
corpus substitution, never an exact Table2 reproduction. Preparation reads only
tokenizer/config/token-array files and model file metadata, not model tensors.
The GPU evaluator runs only under --mode run, after a frozen preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NWIN, LENGTH, TOP_K = 32, 2048, 8
NEED = (NWIN + 1) * LENGTH
EXPECTED_KERNEL = "46fe57767d6b4230f311bc40f529bf04e1608bab68d627efd5dcf08fc7a75ff2"
SOURCE_FILES = ("experiments/paper_quality.py", "experiments/accuracy.py", "experiments/moe.py",
                "experiments/placement.py", "src/placement_router/__init__.py",
                "src/placement_router/kernel.py")
PAPER_ARMS = ("placement_6", "placement_4", "random_6", "random_4")
ADDITIONAL_ARMS = ("placement_6p1", "placement_4p1")
CONTROLS = ("baseline", "baseline2", "noop", "production_noop")
ARM_SPEC = {"baseline": ("baseline", 0, 0.), "baseline2": ("baseline", 0, 0.),
    "noop": ("placement", 8, 0.), "production_noop": ("production_noop", 8, 0.),
    "placement_6": ("placement", 6, 0.), "placement_4": ("placement", 4, 0.),
    "placement_6p1": ("placement", 6, 1.), "placement_4p1": ("placement", 4, 1.),
    "random_6": ("random", 6, 0.), "random_4": ("random", 4, 0.)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def source_identity():
    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    require(hashes["src/placement_router/kernel.py"] == EXPECTED_KERNEL,
            "Quality requires pristine public git2d8f519 kernel.py; experimental/default-modified source is refused")
    return hashes


def protocol(args):
    return {"model_id": MODEL_ID, "logical_ep_size": args.logical_ep,
        "cuda_visible_devices": args.cuda_visible_devices,
        "physical_gpu_count_expected": len(args.cuda_visible_devices.split(",")),
        "dtype": "bfloat16", "engine": "HuggingFace Transformers, device_map=auto",
        "distributed_expert_parallel_execution": False,
        "windows": NWIN, "raw_window_length": LENGTH, "batch_size": 4,
        "scored_targets_per_window": LENGTH - 1, "scored_targets_per_arm": NWIN * (LENGTH - 1),
        "window_protocol": "contiguous nonoverlapping first32 windows; no BOS; score raw positions1..2047",
        "historical_token_preparation": "load first3*((32+1)*2048) reference IDs; decode(skip_special_tokens=True); model-tokenize(add_special_tokens=False); require33 windows, score first32",
        "paper_comparable_arms": list(PAPER_ARMS), "additional_balanced_arms": list(ADDITIONAL_ARMS),
        "controls": list(CONTROLS),
        "penalty": "absolute raw-logit units, mean selected rank slots across a 16-token tile",
        "balanced_score_interpretation": "teacher-forced pseudo-NLL, not causal generative NLL",
        "paper_estimator": "mean paired per-window delta NLL; sample standard error ddof1/sqrt32",
        "additional_ci": "paired-window percentile bootstrap95%,20000 draws,seed0",
        "corpus_mode": args.corpus_mode,
        "exact_table2_reproduction_claim": False,
        "limitations": ["Original multilingual inputs are required to reproduce Table2 numerically; Wikitext is an explicitly different corpus.",
            "The current audited HF tuple-output adapter replaces selected IDs/weights with production select; the historical adapter masked logits. Native BF16 ties/weight arithmetic may differ.",
            "Native-preserving noop and actual production keep8 noop are separate controls; neither proves balanced quality.",
            "Logical EP changes the rank map; HF device_map is not an expert-parallel throughput experiment.",
            "Window uncertainty is conditional on this single corpus sample; balanced arms were not quality-measured in the paper."]}


def corpus_proof(args, original):
    observed = digest(args.stream_file)
    if args.corpus_mode == "historical":
        expected = args.historical_stream_sha256
        require(expected and re.fullmatch(r"[0-9a-f]{64}", expected),
                "Original multilingual stream is unavailable: historical mode requires an independently recovered --historical-stream-sha256; do not substitute Wikitext")
        require(observed == expected, "Recovered historical stream SHA256 differs")
        require(observed != original["stream"]["source_sha256"],
                "Known Wikitext source cannot be relabeled as the historical multilingual corpus")
    else:
        require(observed == original["stream"]["source_sha256"],
                "Wikitext substitution must use the captured original wikitext.npy bytes")
    return {"path": str(args.stream_file), "sha256": observed,
            "bytes": args.stream_file.stat().st_size, "mode": args.corpus_mode,
            "historical_stream_claim": args.corpus_mode == "historical",
            "historical_stream_expected_sha256": args.historical_stream_sha256,
            "paper_table2_stream": "multilingual", "original_wikitext_generation": original["stream"].get("dataset_source_from_script")}


def checkpoint_proof(args, original):
    proof = read(args.checkpoint_proof)
    require(proof.get("canonical_identity_verified") is True and proof.get("model_id") == MODEL_ID,
            "Canonical Instruct2507 checkpoint proof is required")
    require(proof.get("revision") == original.get("download_revision"),
            "Canonical checkpoint differs from captured original PVC revision")
    require(Path(proof.get("local_model_path") or "").resolve() == args.model,
            "Checkpoint proof does not identify the actual model path")
    require(proof.get("weights_verified") == proof.get("safetensors") and len(proof.get("weights_verified", [])) == 16,
            "All16 canonical checkpoint shards must have been SHA256-verified by the model preparation step")
    require(digest(args.model / "config.json") == proof.get("canonical_config_sha256"), "Model config SHA256 differs")
    config = read(args.model / "config.json")
    require((config.get("num_experts"), config.get("num_experts_per_tok"), config.get("num_hidden_layers"), config.get("hidden_size")) == (128, 8, 48, 2048),
            "Unexpected Qwen model dimensions")
    signatures = {}
    for item in proof["weights_verified"]:
        path = args.model / item["filename"]
        require(path.is_file() and path.stat().st_size == item["bytes"], "Missing/truncated verified shard: " + item["filename"])
        signatures[item["filename"]] = [path.stat().st_size, path.stat().st_mtime_ns]
    tokenizer_hashes = {}
    for folder in {args.model, args.reference_tokenizer}:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            expected = original["copied_model_files"][name]["sha256"]
            require(digest(folder / name) == expected, "Tokenizer differs from original PVC: " + str(folder / name))
            tokenizer_hashes[str(folder / name)] = expected
    return {"model_id": MODEL_ID, "revision": proof["revision"], "model_path": str(args.model),
            "canonical_weight_hashes_preverified": proof["weights_verified"], "file_stat_signatures": signatures,
            "proof_sha256": digest(args.checkpoint_proof), "tokenizer_sha256": tokenizer_hashes,
            "original_download_revision_matches": True,
            "scope": "Canonical full-shard hashes were verified by root's model preparation; this wrapper rechecks proof, sizes/mtime, config and tokenizer, without reading model tensors."}


def prepare_tokens(args):
    import numpy as np
    from transformers import AutoTokenizer
    raw = np.load(args.stream_file, allow_pickle=False, mmap_mode="r")
    require(raw.ndim == 1 and raw.dtype.kind in "iu", "Stream must be a one-dimensional integer .npy")
    reference = AutoTokenizer.from_pretrained(str(args.reference_tokenizer), trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    prefix = np.asarray(raw[:3 * NEED])
    text = reference.decode(prefix.tolist(), skip_special_tokens=True)
    encoded = np.asarray(tokenizer(text, add_special_tokens=False).input_ids, dtype=np.int64)
    require(len(encoded) >= NEED and encoded.min() >= 0, f"Historical preparation gave{len(encoded)} tokens; need{NEED}")
    windows = encoded[:NWIN * LENGTH].reshape(NWIN, LENGTH)
    path = args.output / "prepared-token-ids.npy"
    np.save(path, windows.reshape(-1), allow_pickle=False)
    return {"path": str(path), "sha256": digest(path), "source_reference_tokens_available": int(len(raw)),
        "reference_prefix_tokens_decoded": int(len(prefix)), "reference_prefix_dtype": str(raw.dtype),
        "reference_prefix_raw_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        "decoded_utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "retokenized_available_tokens": int(len(encoded)), "historical_required_tokens": NEED,
        "input_token_sha256": hashlib.sha256(windows.astype("<i8").tobytes()).hexdigest(),
        "scored_target_token_sha256": hashlib.sha256(windows[:, 1:].astype("<i8").tobytes()).hexdigest(),
        "reference_tokenizer": str(args.reference_tokenizer), "reference_tokenizer_class": type(reference).__name__,
        "model_tokenizer_class": type(tokenizer).__name__}


def environment(args, prepared):
    env = os.environ.copy()
    quality_keys = {"MODEL", "TAG", "GPUS", "SEQ_LEN", "NWIN", "BATCH", "KEEPS", "PENS", "STREAM", "STREAMS", "TOKREF", "TEXT_FILE", "TOKEN_IDS_FILE", "BOS_TOKEN_ID", "OUT", "KERNEL_ARMS", "PRODUCTION_NOOP", "PARITY_TOKENS", "TRACE_TOKENS", "BOOTSTRAPS", "NOOP_ATOL"}
    for key in list(env):
        if key in quality_keys or key.startswith(("PLACEMENT_", "PAPER_PROBE_")):
            env.pop(key, None)
    env.update(MODEL=str(args.model), TAG=args.tag, GPUS=str(args.logical_ep), SEQ_LEN=str(LENGTH), NWIN=str(NWIN),
        BATCH="4", KEEPS="6,4", PENS="0,1", TOKEN_IDS_FILE=prepared["tokens"]["path"],
        OUT=str(args.output / "quality"), PRODUCTION_NOOP="1", PARITY_TOKENS="128", TRACE_TOKENS="16384",
        BOOTSTRAPS="20000", NOOP_ATOL="1e-7", CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT / "experiments"))))
    return env


def validate_result(data, prepared):
    require(data.get("complete") is True and data.get("validation", {}).get("passed") is True, "Quality controls incomplete or failed")
    require(set(data.get("arms", {})) == set(PAPER_ARMS + ADDITIONAL_ARMS + CONTROLS), "Quality arm set differs")
    for name, expected in ARM_SPEC.items():
        arm = data["arms"][name]
        require(tuple(arm.get(key) for key in ("arm", "keep", "load_penalty")) == expected,
                "Quality arm routing parameters differ: " + name)
    require((data.get("n_windows"), data.get("seq_len"), data.get("scored_tokens_per_arm")) == (NWIN, LENGTH, NWIN * (LENGTH - 1)), "Window/target budget differs")
    require(data.get("logical_ep_size") == prepared["protocol"]["logical_ep_size"], "Logical rank map differs")
    require(data.get("visible_cuda_devices") == prepared["protocol"]["physical_gpu_count_expected"], "Visible physical GPU count differs")
    require(data.get("n_moe_layers") == data.get("n_routers_hooked") == 48, "Missing model-router coverage")
    require(data.get("model") == prepared["checkpoint"]["model_path"], "Evaluator loaded a different checkpoint path")
    for key in ("input_token_sha256", "scored_target_token_sha256"):
        require(data.get("data", {}).get(key) == prepared["tokens"][key], "Scored tokens differ: " + key)
    require(data["data"].get("bos_token_id") is None and data["data"].get("model_input_protocol") == "raw_windows", "Unexpected BOS/context protocol")
    require(all(len(data["arms"][arm].get("per_window", [])) == NWIN for arm in data["arms"]), "Missing paired windows")


def summarize(data, prepared):
    validate_result(data, prepared)
    def select(names):
        return {name: {key: data["arms"][name].get(key) for key in ("nll", "dnll", "dnll_se", "dnll_ci95", "changed_frac", "production_parity_by_layer")} for name in names}
    residual = [a - b for a, b in zip(data["arms"]["production_noop"]["per_window"],
                                    data["arms"]["baseline2"]["per_window"])]
    mean, se = statistics.mean(residual), statistics.stdev(residual) / NWIN ** .5
    noop_ok = abs(mean) <= 3 * se + 1e-7 and data["arms"]["production_noop"].get("changed_frac") == 0.
    return {"complete": True, "protocol": prepared["protocol"], "checkpoint": prepared["checkpoint"],
        "corpus": prepared["corpus"], "tokens": prepared["tokens"], "source_sha256": prepared["source_sha256"],
        "paper_comparable_plain_and_random": select(PAPER_ARMS), "additional_balanced_quality": select(ADDITIONAL_ARMS),
        "controls": select(CONTROLS), "native_noop_validation": data["validation"],
        "production_noop_vs_repeat": {"dnll": mean, "sample_se": se,
            "zero_se_nonzero_delta": se == 0. and mean != 0., "selection_and_resolution_check_passed": noop_ok},
        "quality_interpretation_provisional": not noop_ok,
        "production_noop_note": "Separate measured control; its delta/CI must be interpreted, not silently assumed zero."}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("plan", "prepare", "run", "analyze"), default="plan")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--checkpoint-proof", type=Path)
    p.add_argument("--original-provenance", type=Path, default=ROOT / "runs/paper-reproduction-20261017/original-cluster-provenance.json")
    p.add_argument("--stream-file", type=Path)
    p.add_argument("--reference-tokenizer", type=Path)
    p.add_argument("--corpus-mode", choices=("historical", "wikitext-substitution"), default="historical")
    p.add_argument("--historical-stream-sha256")
    p.add_argument("--logical-ep", type=int, choices=(4, 8), default=4)
    p.add_argument("--cuda-visible-devices", default="0,1,2,3")
    p.add_argument("--tag", default="qwen3-instruct2507-paper")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args(argv)
    require(re.fullmatch(r"[0-9]+(,[0-9]+)*", a.cuda_visible_devices) and len(set(a.cuda_visible_devices.split(","))) == len(a.cuda_visible_devices.split(",")), "CUDA devices must be a distinct comma-separated integer list")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag), "Tag must be a filename-safe label")
    for name in ("output", "model", "checkpoint_proof", "original_provenance", "stream_file", "reference_tokenizer"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    if a.reference_tokenizer is None:
        a.reference_tokenizer = a.model
    if a.mode in ("prepare", "run"):
        require(all(getattr(a, n) for n in ("model", "checkpoint_proof", "stream_file")), "--model, --checkpoint-proof and --stream-file are required")
    return a


def main(argv=None):
    args = arguments(argv)
    manifest_path = args.output / "quality-preparation.json"
    result_path = args.output / "quality" / f"acc_{args.tag}.json"
    if args.mode == "plan":
        print(json.dumps({"protocol": protocol(args), "GPU_run_requires_preparation": True,
            "current_exact_corpus_blocker": "Original Table2 artifacts record multilingual; available captured stream is Wikitext. Historical mode requires recovered stream bytes and independent SHA256.",
            "command": shlex.join([args.python, str(ROOT / "experiments/accuracy.py")])}, indent=2))
        return 0
    if args.mode == "analyze":
        save(args.output / "paper-quality-analysis.json", summarize(read(result_path), read(manifest_path)))
        return 0
    original = read(args.original_provenance)
    corpus = corpus_proof(args, original)
    checkpoint = checkpoint_proof(args, original)
    sources = source_identity()
    expected = {"protocol": protocol(args), "checkpoint": checkpoint, "corpus": corpus,
        "source_sha256": sources, "original_provenance_sha256": digest(args.original_provenance), "tag": args.tag}
    if args.mode == "prepare":
        require(not manifest_path.exists(), "Preparation already exists; use a fresh output directory")
        args.output.mkdir(parents=True, exist_ok=True)
        prepared = {**expected, "tokens": prepare_tokens(args), "prepared_unix": time.time()}
        save(manifest_path, prepared)
        env = environment(args, prepared)
        save(args.output / "quality-command.json", {"command": [args.python, str(ROOT / "experiments/accuracy.py")],
            "controlled_environment": {k: env[k] for k in ("MODEL", "TAG", "GPUS", "SEQ_LEN", "NWIN", "BATCH", "KEEPS", "PENS", "TOKEN_IDS_FILE", "OUT", "PRODUCTION_NOOP", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")}, "GPU_launched": False})
        print(f"Prepared {manifest_path}; no GPU evaluator was launched", flush=True)
        return 0
    prepared = read(manifest_path)
    require(all(prepared.get(k) == v for k, v in expected.items()), "Frozen quality preparation differs; choose a fresh output directory")
    require(digest(prepared["tokens"]["path"]) == prepared["tokens"]["sha256"], "Prepared token file changed")
    require(not result_path.exists(), "Result already exists; analyze it or use a fresh output directory")
    status = {"complete": False, "started_unix": time.time(), "preparation_sha256": digest(manifest_path)}
    save(args.output / "quality-status.json", status)
    try:
        with (args.output / "quality.log").open("w", buffering=1) as log:
            subprocess.run([args.python, str(ROOT / "experiments/accuracy.py")], cwd=ROOT,
                           env=environment(args, prepared), stdout=log, stderr=subprocess.STDOUT, check=True)
        save(args.output / "paper-quality-analysis.json", summarize(read(result_path), prepared))
        status.update(complete=True, result_sha256=digest(result_path))
    except BaseException as exc:
        status.update(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["finished_unix"] = time.time()
        save(args.output / "quality-status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
