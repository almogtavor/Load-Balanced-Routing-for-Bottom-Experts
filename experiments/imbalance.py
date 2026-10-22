#!/usr/bin/env python3
"""Dispatch fan-out and load, measured on the real router's logits inside a running engine.

Default vLLM routing decides the forward: `PLACEMENT_KEEP` stays unset, so the plugin patches nothing
and the engine serves exactly what it would without this file. The probe only *reads* each
router call's logits and scores every setting on them, so one engine load produces the whole
`(ep, keep, load_penalty, tile)` grid on identical inputs - a paired comparison with no run-to-run
variance between settings at all.

The expert-to-rank map is `e * G // E`, arithmetic on the expert id alone, so the grid can
report for any `ep` on whatever GPU count the engine happens to fit on. At `G` ranks over
8-GPU nodes the same arithmetic gives the node map, which makes the `G=2` and `G=4` rows the
inter-node (RDMA) fan-out of a 16- and 32-GPU deployment.

Four numbers per setting:

* `vol`  - summed (token, rank) dispatch pairs. What the rule is built to cut.
* `disp` - dispatch pairs sent to the busiest rank. The all-to-all's tail.
* `work` - expert assignments on the busiest rank, over the mean. Every rank waits for the slowest.
* `exp`  - assignments on the busiest *expert*, over the mean. Their total is fixed, so this is
  the grouped GEMM's own tail: identical work spread over a longer per-expert block is slower.
* `lighter` - the share of calls whose busiest rank does less work than default vLLM's on the
  same call. A mean can hide a minority of calls the rule made worse.

Run through `run.sh` with `PLACEMENT_PROBE` pointing here; `plugin.register()` loads it inside
every worker, which is the only place the patch can reach the model.
"""

from __future__ import annotations

import itertools
import json
import os
import pathlib

import torch


def _ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",")]


def _floats(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",")]


EPS = _ints("EPS", "4,8,16")              # ep sizes to report for, independent of GPU count
KEEPS = _ints("IMB_KEEPS", "6,4,2")       # substitution budgets; default vLLM is added per ep
PENS = _floats("PENS", "0,1")             # load_penalty values per keep
TILES = _ints("TILES", "1,16")            # load-balancing tiles per nonzero load_penalty
CALLS = int(os.environ.get("CALLS", 240))  # router calls to score, then dump and go quiet
TAG = os.environ.get("TAG", "imb")
WARMUP = int(os.environ.get("WARMUP", 480))  # router calls to discard before measuring

_acc: dict[tuple, torch.Tensor] = {}
_calls = 0
_seen = 0
_declined: dict[str, int] = {}
_tokens: list[int] = []  # real rows per measured call: says whether a run saw prefill or decode
_dumped = False


def _hist(ids, n_experts, ranks):
    """`(work[G], dispatch[G], per_expert[E])` for one padding-free selection."""
    ones = torch.ones(ids.numel(), device=ids.device, dtype=torch.float64)
    flat = ids.long().ravel()
    rank = (torch.arange(n_experts, device=ids.device) * ranks // n_experts)[ids.long()]
    work = torch.zeros(ranks, device=ids.device, dtype=torch.float64).scatter_add_(
        0, rank.ravel(), ones)
    per_expert = torch.zeros(n_experts, device=ids.device, dtype=torch.float64).scatter_add_(
        0, flat, ones)
    hot = torch.zeros(ids.shape[0], ranks, dtype=torch.bool, device=ids.device)
    hot.scatter_(1, rank, True)
    return work, hot.sum(0).to(torch.float64), per_expert


def _decline(why: str) -> None:
    """Why a call was skipped, counted. An empty result is then a diagnosis, not a mystery."""
    _declined[why] = _declined.get(why, 0) + 1


def _grid(k):
    """`(ep, keep, load_penalty, tile)` settings. Default vLLM (`keep == k`) once per ep as its baseline."""
    for g in EPS:
        yield g, k, 0.0, 1
        for keep, pen, tile in itertools.product(KEEPS, PENS, TILES):
            if keep < k and (pen or tile == 1):
                yield g, keep, pen, tile


def probe(logits, topk_ids, k) -> None:
    global _calls, _seen
    from placement_router.kernel import rewrite_torch

    if _calls >= CALLS:
        return
    # A capture records kernels instead of running them, and every host sync below - the
    # padding filter, the per-setting reduction - is illegal inside one. vLLM captures its decode
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
    # vLLM sets a padding row's expert ids to -1, which under an unsigned index dtype reads as
    # 4294967295; either way such a row would be indexed out of bounds below, so it is scored
    # out. It stays in the batch the rewrite sees, as it does in the kernel, so the tiles match.
    ids = topk_ids.view(torch.int32) if topk_ids.dtype == torch.uint32 else topk_ids
    live = (ids >= 0).all(1)
    if not bool(live.any()):
        return _decline("every row padded")
    fl = logits.float()
    full = torch.where(live[:, None], ids, fl.topk(k, -1).indices.to(ids.dtype))
    _calls += 1
    _tokens.append(int(live.sum()))
    for g, keep, pen, tile in _grid(k):
        if e % g or e // g < k:
            _decline(f"{e} experts over {g} ranks leaves no substitutes")
            continue
        sel = ids[live] if keep >= k else rewrite_torch(fl, full, keep, g, pen, tile, ~live)[live]
        w, d, x = _hist(sel, e, g)
        if keep >= k:
            top = w.max()                                              # `_grid` yields it first
        row = torch.stack([d.sum(), d.max(), w.max() / w.mean(), x.max() / x.mean(), w.sum(),
                           (w.max() < top).double()])
        key = (g, keep, pen, tile)
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
    print(f"[imb] calls={_calls} seen={_seen} declined={_declined} tokens per call: "
          f"min {min(_tokens)} median {sorted(_tokens)[len(_tokens) // 2]} max {max(_tokens)}", flush=True)
    print(f"[imb] {'setting':<22}{'vol ↓':>10}{'busiest rank recv ↓':>22}{'busiest rank work ↓':>22}"
          f"{'busiest expert ↓':>19}{'calls lighter ↑':>18}", flush=True)
    for g in EPS:
        if (g, _k0(), 0.0, 1) not in out:
            continue
        base = out[(g, _k0(), 0.0, 1)]
        for (gg, keep, pen, tile), (vol, dmax, work, xmax, wk, lighter) in out.items():
            if gg != g:
                continue
            # Total expert assignments are `rows * top_k` in every setting by construction. If it
            # moved, the rule dropped or duplicated one and every ratio here measures that instead.
            assert abs(wk / base[4] - 1) < 1e-9, f"{(gg, keep, pen, tile)} moved work {wk}"
            name = ("default vLLM" if keep >= _k0() else
                    f"keep={keep} pen={pen:g}" + (f" tile={tile}" if pen else ""))
            print(f"[imb] ep={g:<3} {name:<24}"
                  f"{vol / base[0] - 1:>+9.2%}{dmax / base[1] - 1:>+21.2%}"
                  f"{work / base[2] - 1:>+21.2%}{xmax / base[3] - 1:>+18.2%}{lighter:>17.1%}",
                  flush=True)
    p = pathlib.Path("runs") / f"imbalance_{TAG}_{rank}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps({"calls": _calls, "eps": EPS, "tokens": _tokens,
                             "settings": {f"{g}/{keep}/{pen:g}/{tile}": v for (g, keep, pen, tile), v
                                      in out.items()}}, indent=1))
    print(f"[imb] wrote {p}", flush=True)


_TOPK = [0]


def _k0():
    return _TOPK[0]


def install() -> None:
    """Wrap the default routing units read-only. Loaded by `plugin.register()` inside each
    worker; patching from the driver process would never reach the model."""
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    from placement_router.plugin import _LOGIT_TOPK

    # Gemma routes through a function in its own model file rather than the fused unit, so it
    # needs a second wrapper. Only a listed function ranks by gate logit alone, which is the
    # rule's premise; anything else is declined rather than assumed equivalent.
    custom_inner = CustomRoutingRouter._compute_routing

    def _custom_routing(self, hidden_states, router_logits, indices_type, **kw):
        out = custom_inner(self, hidden_states, router_logits, indices_type, **kw)
        name = getattr(self.custom_routing_function, "__qualname__", "?")
        if name not in _LOGIT_TOPK:
            _decline(f"custom routing function {name}")
        elif self.eplb_state is not None:
            _decline("EPLB breaks the expert-to-rank map")
        else:
            _TOPK[0] = self.top_k
            probe(router_logits, out[1], self.top_k)
        return out

    CustomRoutingRouter._compute_routing = _custom_routing

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
            _TOPK[0] = self.top_k
            probe(router_logits, out[1], self.top_k)
        return out

    FusedTopKRouter._compute_routing = _compute_routing
    print(f"[imb] probe installed, eps={EPS} keeps={KEEPS} pens={PENS} tiles={TILES} calls={CALLS} "
          f"warmup={WARMUP}", flush=True)


def main() -> None:
    """Load an engine under default vLLM routing and generate until the probe has its calls."""
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
    # Real text when given: random ids route unlike text. ponytail: the cached streams on the
    # results volume are Qwen3 token ids, so this is Qwen3 only.
    stream = os.environ.get("IMB_STREAM")
    toks = (torch.from_numpy(__import__("numpy").load(stream)[: n_req * prompt_len].astype("int64"))
            .view(n_req, prompt_len).tolist() if stream
            else torch.randint(0, vocab, (n_req, prompt_len), generator=g).tolist())
    llm.generate([{"prompt_token_ids": t} for t in toks],
                 SamplingParams(max_tokens=int(os.environ.get("GEN_LEN", 8)),
                                temperature=0.0))
    print("[imb] generation done; the per-worker [imb] lines above carry the result", flush=True)


if __name__ == "__main__":
    main()
