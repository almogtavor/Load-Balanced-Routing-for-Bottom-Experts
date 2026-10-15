"""What placement-aware re-routing costs in NLL, and the routing it produces.

One process, four GPUs via `device_map="auto"`. For each arm the router logits of every MoE
layer are masked (see `placement.py` for the rule, `moe.py` for the hook) and the model is
scored teacher-forced on byte-identical token windows, so `delta-NLL = NLL_arm - NLL_baseline`
is a paired difference over the same positions and its standard error comes from the per-window
spread rather than from an assumption.

Three things make the numbers here trustworthy rather than merely produced:

* a **repeat baseline** (`baseline2`), which is the same arm run twice and therefore measures
  what this instrument's own floor is. Every other arm is judged against it rather than against
  zero.
* a **no-op arm** (`keep = top_k`, so the allowed GPU set is the one the baseline already
  visits) must reproduce the baseline. It is the only check that the masking does not perturb
  the selection by itself - every family normalizes after the projection this hook sits on, and
  if that reordered the head then the masked arms would be measuring the cost of *dropping*
  good experts, not of substituting bottom ones. Its selection must match the baseline's
  exactly, token for token; its NLL only has to be indistinguishable from the baseline's,
  because on the families whose softmax sits downstream of the hook the mask changes that
  softmax's denominator and the top-k renormalization cancels the change only in exact
  arithmetic. In bf16 it cancels to rounding, so the honest bound is the measurement's own
  resolution rather than a zero that floating point cannot deliver.
* a **fired fraction** per arm, counting tokens whose selection actually changed. A null result
  from a hook that silently did nothing looks exactly like a free lunch.
* a **random control** making the same number of substitutions with no placement benefit, which
  separates "the bottom experts do not matter" from "our choice of replacement is good".

Writes `runs/acc_<tag>.json`.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import torch
from transformers import AutoTokenizer

import moe
import placement as P

MODEL = os.environ["MODEL"]
TAG = os.environ["TAG"]
N_GPUS = int(os.environ.get("GPUS", 4))
SEQ_LEN = int(os.environ.get("SEQ_LEN", 2048))
N_WIN = int(os.environ.get("NWIN", 32))
BATCH = int(os.environ.get("BATCH", 4))
KEEPS = tuple(int(x) for x in os.environ.get("KEEPS", "6,4,2").split(","))
STREAM = os.environ.get("STREAM", "multilingual")
STREAMS = os.environ.get("STREAMS", "/results/streams")
TOKREF = os.environ.get("TOKREF", "/models/qwen_tok")
# Tokens of routing kept per layer for the fan-out count.
TRACE_TOKENS = int(os.environ.get("TRACE_TOKENS", 16384))
OUT = pathlib.Path("runs")


def token_ids(need: int) -> np.ndarray:
    """`need` ids of the fixed stream, in this model's own vocabulary.

    The cached streams are stored in Qwen3's vocabulary, so they are decoded back to text and
    re-encoded - unconditionally, including for the Qwen models, because one code path cannot
    be wrong about which models share a vocabulary. Token *identity* across models does not
    matter here at all: every arm is compared only against its own model's baseline.
    """
    ref = AutoTokenizer.from_pretrained(TOKREF, trust_remote_code=True)
    raw = np.load(f"{STREAMS}/{STREAM}.npy")[: 3 * need]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = np.asarray(tok(ref.decode(raw.tolist(), skip_special_tokens=True),
                         add_special_tokens=False).input_ids, dtype=np.int64)
    assert len(ids) >= need, f"stream gave {len(ids)} ids, need {need}"
    return ids[:need]


@torch.no_grad()
def per_window_nll(model, win: np.ndarray) -> list[float]:
    """Mean NLL of positions `1..L-1` of each window. Paired across arms by window index."""
    out: list[float] = []
    for s in range(0, len(win), BATCH):
        b = torch.from_numpy(win[s : s + BATCH]).to(model.device)
        logits = model(input_ids=b).logits.float()
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
            reduction="none").view(b.shape[0], -1)
        out.extend(loss.mean(1).tolist())
        print(f"  {min(s + BATCH, len(win))}/{len(win)} windows", flush=True)
    return out


def main() -> None:
    cfg, n_experts, top_k = moe.routing_cfg(MODEL, N_GPUS)
    print(f"{TAG}: E={n_experts} top_k={top_k} G={N_GPUS}", flush=True)
    model = moe.load(MODEL)
    state, handles = moe.install(model, n_experts, top_k, int(cfg.hidden_size), N_GPUS,
                                 trace_tokens=TRACE_TOKENS)

    win = token_ids((N_WIN + 1) * SEQ_LEN)[: N_WIN * SEQ_LEN].reshape(N_WIN, SEQ_LEN)
    results: dict[str, dict] = {}
    n_layers = 0
    gpu_cpu = P.gpu_of(n_experts, N_GPUS)
    for name, arm, keep in moe.arms_of(top_k, KEEPS):
        # The random control draws its substitutes from the global generator, so without this
        # the control arm is a different draw on every rerun and the two numbers a reader
        # compares are not the two numbers that were compared here.
        torch.manual_seed(0)
        state.update(arm=arm, keep=keep, trace={}, changed=0, rows=0)
        print(f"[{name}] arm={arm} keep={keep}", flush=True)
        pw = per_window_nll(model, win)

        tr = torch.stack([torch.cat(state["trace"][k])[:TRACE_TOKENS]
                          for k in sorted(state["trace"])])          # [layers, tokens, top_k]
        n_layers = tr.shape[0]
        fan = P.fanout(tr.reshape(-1, top_k), gpu_cpu, N_GPUS).float().mean().item()

        d = np.array(pw) - np.array(results["baseline"]["per_window"]) if results else np.zeros(N_WIN)
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if results else 0.0
        results[name] = {
            "nll": float(np.mean(pw)), "per_window": pw,
            "dnll": float(d.mean()), "dnll_se": se,
            "t": float(d.mean() / se) if se > 0 else 0.0,
            "fanout": fan, "fanout_ratio": fan / results["baseline"]["fanout"] if results else 1.0,
            "changed_frac": state["changed"] / max(state["rows"], 1),
        }
        r = results[name]
        print(f"[{name}] nll={r['nll']:.5f} dnll={r['dnll']:+.5f}+-{r['dnll_se']:.5f} "
              f"(t={r['t']:+.1f}) fan-out={fan:.3f} ({r['fanout_ratio']:.3f}x) "
              f"changed={r['changed_frac']:.3f}", flush=True)

    for h in handles:
        h.remove()

    # The no-op arm masks nothing the baseline would have selected, so it must reproduce the
    # baseline's selection token for token - that one is exact, and a non-zero `changed_frac`
    # means the hook perturbs the selection on its own and every other arm on this model is
    # uninterpretable. Its NLL is held to the paired test's own resolution instead: `t` is the
    # delta in units of its standard error, so `|t| < 3` says the residual is not separable
    # from zero at this window count. Both are hard stops, not warnings.
    noop, floor = results["noop"], results["baseline2"]
    assert noop["changed_frac"] == 0.0, f"no-op arm changed {noop['changed_frac']:.4f} of rows"
    assert abs(noop["t"]) < 3, (f"no-op arm moved NLL by {noop['dnll']:+.2e} "
                                f"+-{noop['dnll_se']:.2e} (t={noop['t']:+.1f}), floor "
                                f"{floor['dnll']:+.2e} +-{floor['dnll_se']:.2e}")
    # A null result from a hook that silently did nothing looks exactly like a free lunch, so
    # every intervention arm has to fire.
    for arm in ("placement", "random"):
        f = [results[f"{arm}_{k}"]["changed_frac"] for k in sorted(KEEPS, reverse=True)]
        assert min(f) > 0.05, f"{arm}: intervention barely fired {f}"
    # The placement arm's fired fraction must also *rise* as `keep` falls, which is the rule's
    # own monotonicity: fewer protected ranks means more slots free to be re-placed. The bar is
    # not "most tokens" - at `keep = top_k - 2` the bottom picks are often already on a GPU the
    # head forces a visit to, so there is nothing to substitute and the arm correctly leaves
    # most tokens alone. The random control gets no such check: it redraws its substitutes from
    # all `E` experts, so at E=128 it changes essentially every token at every `keep` and its
    # fired fractions sit at 0.9999, 1.0000, 1.0000 - an ordering that is float noise rather
    # than a property, and asserting on it only tests the saturation.
    f = [results[f"placement_{k}"]["changed_frac"] for k in sorted(KEEPS, reverse=True)]
    assert all(x <= y for x, y in zip(f, f[1:])), f"placement: not monotone in keep {f}"

    OUT.mkdir(exist_ok=True)
    (OUT / f"acc_{TAG}.json").write_text(json.dumps({
        "tag": TAG, "model": MODEL, "stream": STREAM, "n_experts": n_experts,
        "top_k": top_k, "n_gpus": N_GPUS,
        # Hooked and fired differ when a family carries a router that never runs during
        # scoring, such as Qwen3.5's multi-token-prediction head.
        "n_routers_hooked": state["n_routers"], "n_moe_layers": n_layers,
        "n_windows": N_WIN, "seq_len": SEQ_LEN,
        # The repeat baseline, promoted out of `arms` so a reader cannot miss what every delta
        # here has to be larger than to mean anything.
        "noise_floor": {"dnll": floor["dnll"], "dnll_se": floor["dnll_se"], "t": floor["t"]},
        "arms": results,
    }, indent=1) + "\n")
    print(f"WROTE {OUT}/acc_{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
