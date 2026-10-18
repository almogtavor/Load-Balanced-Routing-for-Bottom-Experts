"""Paired natural-text NLL for stock, no-op, placement, balanced, and random routing.

All arms score byte-identical windows. Defaults are 32 x 2048 tokens, keep=6,
PENS=0,1. Set TEXT_FILE to UTF-8 natural text or TOKEN_IDS_FILE to a one-dimensional
.npy array already in MODEL's vocabulary. Without either, the historical Qwen-token
stream is decoded with TOKREF and re-tokenized. Input hashes and counts are saved.

Confidence intervals bootstrap paired window differences, not individual tokens;
they describe this corpus sample, not variation between model families or launches.
With a tile-shared nonzero load penalty, teacher-forced scores are candidate-
conditioned pseudo-NLL; they are not causal generative NLL or perplexity.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform

import numpy as np
import torch
from transformers import AutoTokenizer

import moe
import placement as P

MODEL = os.environ.get("MODEL", "")
TAG = os.environ.get("TAG", "")
N_GPUS = int(os.environ.get("GPUS", 4))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))
N_WIN = int(os.environ.get("NWIN", 32))
BATCH = int(os.environ.get("BATCH", 4))
KEEPS = tuple(int(x) for x in os.environ.get("KEEPS", "6").split(","))
PRODUCTION_NOOP = os.environ.get("PRODUCTION_NOOP", "0") == "1"
BOS_TOKEN_ID = int(os.environ["BOS_TOKEN_ID"]) if "BOS_TOKEN_ID" in os.environ else None
PENS = tuple(float(x) for x in os.environ.get("PENS", "0,1").split(","))
STREAM = os.environ.get("STREAM", "multilingual")
STREAMS = os.environ.get("STREAMS", "/results/streams")
TOKREF = os.environ.get("TOKREF", "/models/qwen_tok")
TEXT_FILE = os.environ.get("TEXT_FILE")
TOKEN_IDS_FILE = os.environ.get("TOKEN_IDS_FILE")
TRACE_TOKENS = int(os.environ.get("TRACE_TOKENS", 16384))
PARITY_TOKENS = int(os.environ.get("PARITY_TOKENS", 128))
BOOTSTRAPS = int(os.environ.get("BOOTSTRAPS", 20000))
NOOP_ATOL = float(os.environ.get("NOOP_ATOL", "1e-7"))
OUT = pathlib.Path(os.environ.get("OUT", "runs"))


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_ids(need: int) -> tuple[np.ndarray, dict]:
    """Return target-vocabulary tokens and explicit source/tokenizer provenance."""
    if TEXT_FILE and TOKEN_IDS_FILE:
        raise ValueError("choose TEXT_FILE or TOKEN_IDS_FILE, not both")
    tok = None
    if TOKEN_IDS_FILE:
        source = pathlib.Path(TOKEN_IDS_FILE)
        ids = np.load(source, allow_pickle=False, mmap_mode="r")
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            raise ValueError("TOKEN_IDS_FILE must contain a one-dimensional integer array")
        mode = "target_model_token_ids"
    else:
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if TEXT_FILE:
            source = pathlib.Path(TEXT_FILE)
            text = source.read_text(encoding="utf-8")
            mode = "utf8_text"
        else:
            source = pathlib.Path(STREAMS) / f"{STREAM}.npy"
            ref = AutoTokenizer.from_pretrained(TOKREF, trust_remote_code=True)
            raw = np.load(source, allow_pickle=False)[: 3 * need]
            text = ref.decode(raw.tolist(), skip_special_tokens=True)
            mode = "reference_tokens_decoded_and_retokenized"
        ids = np.asarray(tok(text, add_special_tokens=False).input_ids, dtype=np.int64)
    available = len(ids)
    if available < need:
        raise ValueError(f"source gave {available} model tokens, need {need}")
    ids = np.asarray(ids[:need], dtype=np.int64)
    if ids.min() < 0:
        raise ValueError("input contains a negative token id")
    return ids, {
        "mode": mode, "source": str(source.resolve()), "source_sha256": file_sha256(source),
        "source_bytes": source.stat().st_size, "model_tokenizer": MODEL,
        "tokenizer_class": type(tok).__name__ if tok is not None else None,
        "reference_tokenizer": TOKREF if mode.startswith("reference") else None,
        "available_model_tokens": available, "used_input_tokens": len(ids),
        "input_token_sha256": hashlib.sha256(ids.astype("<i8", copy=False).tobytes()).hexdigest(),
        "token_hash_encoding": "contiguous little-endian int64",
        "token_ids_require_model_vocabulary": mode == "target_model_token_ids",
    }


def paired_summary(values, baseline, bootstraps: int = BOOTSTRAPS) -> dict:
    """Mean paired delta, sample SE, and percentile bootstrap 95% CI over windows."""
    d = np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("paired inference needs at least two finite window differences")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    if bootstraps < 1000:
        raise ValueError("BOOTSTRAPS must be at least 1000")
    rng = np.random.default_rng(0)
    sampled = d[rng.integers(len(d), size=(bootstraps, len(d)))].mean(1)
    ci = np.quantile(sampled, [0.025, 0.975]).tolist()
    # Nonzero constant differences have zero SE; reporting t=0 would falsely pass a no-op.
    t = mean / se if se > 0 else (0.0 if mean == 0 else None)
    return {"dnll": mean, "dnll_se": se, "dnll_ci95": ci, "t": t,
            "zero_se_nonzero_delta": bool(se == 0 and mean != 0)}


@torch.no_grad()
def per_window_nll(model, win: np.ndarray, bos_token_id: int | None = None) -> list[float]:
    """Mean NLL on raw target positions 1..L-1, optionally with a BOS context.

    BOS prepends to the model input without replacing any raw token. Its own first
    prediction is excluded: logits[:, 1:-1] still scores exactly raw[:, 1:].
    """
    out = []
    for s in range(0, len(win), BATCH):
        b = torch.from_numpy(win[s:s + BATCH]).to(model.device)
        model_input = b if bos_token_id is None else torch.cat([
            torch.full((len(b), 1), bos_token_id, dtype=b.dtype, device=b.device), b], dim=1)
        logits = model(input_ids=model_input).logits.float()
        predictions = logits[:, :-1] if bos_token_id is None else logits[:, 1:-1]
        loss = torch.nn.functional.cross_entropy(
            predictions.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
            reduction="none").view(b.shape[0], -1)
        out.extend(loss.mean(1).tolist())
        print(f"  {min(s + BATCH, len(win))}/{len(win)} windows", flush=True)
    return out


def sanity_checks(results: dict) -> dict:
    noop, floor = results["noop"], results["baseline2"]
    residual = paired_summary(noop["per_window"], floor["per_window"])
    bound = 3 * residual["dnll_se"] + NOOP_ATOL
    checks = {
        "noop_selection_exact": noop["changed_frac"] == 0.0,
        "noop_vs_repeat": residual,
        "noop_absolute_tolerance": NOOP_ATOL,
        "noop_nll_within_resolution": abs(residual["dnll"]) <= bound,
    }
    interventions = {name: r["changed_frac"] for name, r in results.items()
                     if r["arm"] in ("placement", "random") and name != "noop"}
    checks["interventions_fired"] = {name: f > 0.05 for name, f in interventions.items()}
    # Monotonicity is a property of the unpenalized placement rule only.
    plain = [results[f"placement_{k}"]["changed_frac"] for k in sorted(KEEPS, reverse=True)
             if f"placement_{k}" in results]
    checks["plain_placement_monotone"] = all(x <= y for x, y in zip(plain, plain[1:]))
    checks["passed"] = (checks["noop_selection_exact"] and checks["noop_nll_within_resolution"]
                        and all(checks["interventions_fired"].values())
                        and checks["plain_placement_monotone"])
    return checks


def main() -> None:
    if not MODEL or not TAG:
        raise ValueError("MODEL and TAG are required")
    if N_WIN < 2 or SEQ_LEN < 2 or BATCH < 1 or TRACE_TOKENS < 1:
        raise ValueError("need NWIN >= 2, SEQ_LEN >= 2, BATCH >= 1, TRACE_TOKENS >= 1")
    cfg, n_experts, top_k = moe.routing_cfg(MODEL, N_GPUS)
    arms = moe.arms_of(top_k, KEEPS, PENS)
    if PRODUCTION_NOOP:
        arms.insert(3, ("production_noop", "production_noop", top_k, 0.0))
    ids, data = token_ids(N_WIN * SEQ_LEN)
    if ids.max() >= int(cfg.vocab_size):
        raise ValueError(f"token {ids.max()} exceeds model vocabulary {cfg.vocab_size}")
    win = ids.reshape(N_WIN, SEQ_LEN)
    if BOS_TOKEN_ID is not None and not 0 <= BOS_TOKEN_ID < int(cfg.vocab_size):
        raise ValueError("BOS_TOKEN_ID is outside the model vocabulary")
    model_inputs = win if BOS_TOKEN_ID is None else np.concatenate([
        np.full((N_WIN, 1), BOS_TOKEN_ID, dtype=np.int64), win], axis=1)
    data["model_input_protocol"] = "raw_windows" if BOS_TOKEN_ID is None else "prepend_bos_per_window"
    data["bos_token_id"] = BOS_TOKEN_ID
    data["model_input_seq_len"] = model_inputs.shape[1]
    data["model_input_token_sha256"] = hashlib.sha256(model_inputs.astype("<i8").tobytes()).hexdigest()
    data["scored_target_token_sha256"] = hashlib.sha256(win[:, 1:].astype("<i8").tobytes()).hexdigest()
    print(f"{TAG}: E={n_experts} top_k={top_k} G={N_GPUS}, input={data['input_token_sha256']}", flush=True)
    model = moe.load(MODEL)
    state, handles = moe.install(model, n_experts, top_k, int(cfg.hidden_size), N_GPUS,
                                 trace_tokens=TRACE_TOKENS, parity_tokens=PARITY_TOKENS)
    results = {}
    payload = {
        "tag": TAG, "model": MODEL, "stream": STREAM, "n_experts": n_experts,
        "top_k": top_k, "n_gpus": N_GPUS, "logical_ep_size": N_GPUS,
        "visible_cuda_devices": torch.cuda.device_count(),
        "physical_device_map": getattr(model, "hf_device_map", None),
        "n_routers_hooked": state["n_routers"],
        "n_windows": N_WIN, "seq_len": SEQ_LEN, "batch_size": BATCH,
        "scored_tokens_per_arm": N_WIN * (SEQ_LEN - 1), "data": data,
        "model_input_tokens_per_arm": int(model_inputs.size),
        "keeps": KEEPS, "load_penalties": PENS, "arm_order": [a[0] for a in arms],
        "ci_method": "paired-window percentile bootstrap, 95%, seed=0",
        "bootstrap_samples": BOOTSTRAPS, "torch_version": torch.__version__,
        "placement_implementation": "production select (Triton on CUDA; torch reference otherwise)",
        "noop_implementation": "native-tie-preserving selection, native weight arithmetic",
        "production_noop_enabled": PRODUCTION_NOOP,
        "parity_tokens_per_layer": PARITY_TOKENS,
        "parity_comparison": "native-tie-preserving reference versus production selector",
        "python_version": platform.python_version(), "arms": results, "complete": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"acc_{TAG}.json"

    def save():
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
        tmp.replace(dest)

    gpu_cpu = P.gpu_of(n_experts, N_GPUS)
    try:
        for name, arm, keep, penalty in arms:
            torch.manual_seed(0)
            state.update(arm=arm, keep=keep, load_penalty=penalty, trace={}, parity={}, changed=0, rows=0)
            print(f"[{name}] arm={arm} keep={keep} load_penalty={penalty:g}", flush=True)
            pw = per_window_nll(model, win, BOS_TOKEN_ID)
            tr = torch.stack([torch.cat(state["trace"][k])[:TRACE_TOKENS]
                              for k in sorted(state["trace"])])
            payload["n_moe_layers"] = tr.shape[0]
            fan = P.fanout(tr.reshape(-1, top_k), gpu_cpu, N_GPUS).float().mean().item()
            summary = paired_summary(pw, results["baseline"]["per_window"] if results else pw)
            results[name] = {
                "arm": arm, "keep": keep, "load_penalty": penalty,
                "nll": float(np.mean(pw)), "per_window": pw, **summary,
                "fanout": fan, "fanout_ratio": fan / results["baseline"]["fanout"] if results else 1.0,
                "changed_frac": state["changed"] / max(state["rows"], 1),
                "router_rows": state["rows"], "traced_tokens_per_layer": tr.shape[1],
                "production_parity_by_layer": state["parity"],
                "router_logit_sources": dict(state["raw_sources"]),
            }
            r = results[name]
            print(f"[{name}] nll={r['nll']:.5f} dnll={r['dnll']:+.5f} "
                  f"95%CI={r['dnll_ci95']} fan-out={fan:.3f} ({r['fanout_ratio']:.3f}x) "
                  f"changed={r['changed_frac']:.3f}", flush=True)
            save()
    finally:
        for handle in handles:
            handle.remove()
    payload["noise_floor"] = {k: results["baseline2"][k] for k in
                              ("dnll", "dnll_se", "dnll_ci95", "t")}
    payload["validation"] = sanity_checks(results)
    payload["complete"] = True
    save()
    print(f"WROTE {dest}", flush=True)
    assert payload["validation"]["passed"], f"quality controls failed; inspect {dest}"


if __name__ == "__main__":
    main()
