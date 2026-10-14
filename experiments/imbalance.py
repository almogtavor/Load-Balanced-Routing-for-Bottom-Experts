#!/usr/bin/env python3
"""Per-rank load imbalance, measured on the real router's logits inside a running engine.

`fanout.py` predicts that the placement rule trades dispatch volume for a hotter busiest
rank, from Gaussian logits. This measures the same three quantities on the logits an actual
model emits, which is the only way to know whether the prediction's sign and size carry over.

Stock routing decides the forward: `PLACEMENT_KEEP` stays unset, so the plugin patches
nothing and the engine serves exactly what it would without this file. The probe only *reads*
each router call's logits and scores every arm on them, so one engine load produces the whole
`(keep, load_penalty)` sweep on identical inputs - a paired comparison, with no run-to-run
variance between arms at all.

Three numbers per arm, all relative to the rank total:

* `work` - expert slots assigned to the busiest rank, over the mean. On the critical path:
  every rank waits for the slowest expert GEMM.
* `disp` - tokens dispatched to the busiest rank, over the mean. The all-to-all's tail.
* `vol` - summed tokens over all ranks, the quantity the rule is designed to cut.

Run through `run.sh` with `PLACEMENT_PROBE` pointing here; `plugin.register()` loads it inside
every worker, which is the only place the patch can reach the model.
"""

from __future__ import annotations

import json
import os
import pathlib

import torch

RANKS = int(os.environ.get("RANKS", 4))      # the ep_size being reported for
CALLS = int(os.environ.get("CALLS", 240))    # router calls to score, then dump and go quiet
TAG = os.environ.get("TAG", "imb")
WARMUP = int(os.environ.get("WARMUP", 480))  # router calls to discard before measuring
ARMS = [(8, 0.0), (4, 0.0), (4, 0.5), (4, 1.0), (4, 2.0), (4, 4.0), (2, 0.0), (2, 1.0)]

_acc: dict[str, torch.Tensor] = {}
_calls = 0
_seen = 0
_declined: dict[str, int] = {}
_dumped = False


def _hist(ids, n_experts):
    """`(work[G], dispatch[G])` for one selection, which must be free of padding rows."""
    rank = (torch.arange(n_experts, device=ids.device) * RANKS // n_experts)[ids.long()]
    work = torch.zeros(RANKS, device=ids.device, dtype=torch.float64).scatter_add_(
        0, rank.ravel(), torch.ones(rank.numel(), device=ids.device, dtype=torch.float64))
    hot = torch.zeros(ids.shape[0], RANKS, dtype=torch.bool, device=ids.device)
    hot.scatter_(1, rank, True)
    return work, hot.sum(0).to(torch.float64)


def _decline(why: str) -> None:
    """Why a call was skipped, counted. An empty result is then a diagnosis, not a mystery."""
    _declined[why] = _declined.get(why, 0) + 1


def probe(logits, topk_ids, k) -> None:
    global _calls, _seen
    from placement_router.kernel import rewrite_torch

    if _calls >= CALLS:
        return
    # A capture records kernels instead of running them, and every host sync below - the
    # padding filter, the per-arm reduction - is illegal inside one. vLLM captures its decode
    # graphs at startup over dummy inputs, which is not traffic worth measuring anyway.
    if torch.cuda.is_current_stream_capturing():
        return
    # Startup also does an eager profiling forward on dummy hidden states, whose logits are not
    # the model routing anything. One forward is `n_layers` calls, so discard several forwards'
    # worth before believing what the router emits.
    _seen += 1
    if _seen <= WARMUP:
        return
    e = logits.shape[-1]
    if e % RANKS or e // RANKS < k:
        return _decline(f"{e} experts over {RANKS} ranks leaves no substitutes")
    # vLLM pads unused slots with -1, which under an unsigned index dtype reads as 4294967295;
    # either way such a row would be indexed out of bounds below, so drop the row.
    ids = topk_ids.view(torch.int32) if topk_ids.dtype == torch.uint32 else topk_ids
    live = (ids >= 0).all(1)
    if not bool(live.any()):
        return _decline("every row padded")
    logits, ids = logits[live], ids[live]
    _calls += 1
    for keep, pen in ARMS:
        sel = ids if keep >= k else rewrite_torch(logits.float(), ids, keep, RANKS, pen)
        w, d = _hist(sel, e)
        row = torch.stack([w.max() / w.mean(), d.max() / d.mean(), d.sum(), w.sum()])
        key = f"{keep}p{pen:g}"
        _acc[key] = row if key not in _acc else _acc[key] + row
    if _calls == CALLS:
        dump()


def dump() -> None:
    """Report and write. Called on reaching `CALLS` rather than at exit, which vLLM may not
    reach: it tears workers down with a signal, and an `atexit` hook would never run."""
    global _dumped
    if _dumped:
        return
    _dumped = True
    rank = int(os.environ.get("VLLM_DP_RANK", 0))
    if not _acc:
        print(f"[imb] NOTHING MEASURED in {_seen} calls past warm-up, "
              f"declined={_declined or 'never called'}"
              f" - the hook did not reach the router", flush=True)
        return
    out = {k: (v / _calls).tolist() for k, v in _acc.items()}
    base = out[f"{max(k for k, _ in ARMS)}p0"]
    print(f"[imb] ranks={RANKS} calls={_calls} seen={_seen} declined={_declined}", flush=True)
    for key, (work, disp, vol, wk) in out.items():
        # Total expert slots is `rows * top_k` in every arm by construction. If it moved, the
        # rule dropped or duplicated a slot and every ratio above is measuring that instead.
        assert abs(wk / base[3] - 1) < 1e-9, f"{key} moved total work {wk} vs {base[3]}"
        print(f"[imb] {key:<7} work max/mean {work:.4f} ({work / base[0] - 1:+.2%})"
              f"   disp max/mean {disp:.4f} ({disp / base[1] - 1:+.2%})"
              f"   vol {vol / base[2] - 1:+.2%}", flush=True)
    p = pathlib.Path("runs") / f"imbalance_{TAG}_{rank}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"ranks": RANKS, "calls": _calls, "arms": out}, indent=1))
    print(f"[imb] wrote {p}", flush=True)


def install() -> None:
    """Wrap stock `FusedTopKRouter._compute_routing` read-only. Loaded by `plugin.register()`
    inside each worker; patching from the driver process would never reach the model."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    inner = FusedTopKRouter._compute_routing

    def _compute_routing(self, hidden_states, router_logits, indices_type, **kw):
        out = inner(self, hidden_states, router_logits, indices_type, **kw)
        if self.scoring_func != "softmax":
            _decline(f"scoring_func={self.scoring_func!r}")
        elif not self.renormalize:
            _decline("no renormalization")
        elif self.eplb_state is not None:
            _decline("EPLB breaks the expert-to-rank map")
        else:
            probe(router_logits, out[1], self.top_k)
        return out

    FusedTopKRouter._compute_routing = _compute_routing
    print(f"[imb] probe installed, ranks={RANKS} calls={CALLS} warmup={WARMUP} arms={ARMS}",
          flush=True)


def main() -> None:
    """Load an engine under stock routing and generate until the probe has its calls."""
    from vllm import LLM, SamplingParams

    os.environ.pop("PLACEMENT_KEEP", None)
    os.environ["PLACEMENT_PROBE"] = str(pathlib.Path(__file__).resolve())
    n_req = int(os.environ.get("N_REQ", 32))
    prompt_len = int(os.environ.get("PROMPT_LEN", 4096))
    llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=int(os.environ.get("GPUS", 4)),
              enable_expert_parallel=True, trust_remote_code=True,
              max_model_len=int(os.environ.get("MAX_LEN", 8192)), seed=0,
              enable_prefix_caching=False,
              # The probe reads logits and syncs on them, which a graph capture forbids, and a
              # replayed graph would skip the Python wrapper entirely. Eager costs time this run
              # does not care about and changes nothing about which experts get selected.
              enforce_eager=True,
              gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.85)))
    g = torch.Generator().manual_seed(0)
    vocab = llm.get_tokenizer().vocab_size
    toks = torch.randint(0, vocab, (n_req, prompt_len), generator=g).tolist()
    llm.generate([{"prompt_token_ids": t} for t in toks],
                 SamplingParams(max_tokens=int(os.environ.get("GEN_LEN", 8)),
                                temperature=0.0))
    print("[imb] generation done; the per-worker [imb] lines above carry the result", flush=True)


if __name__ == "__main__":
    main()
