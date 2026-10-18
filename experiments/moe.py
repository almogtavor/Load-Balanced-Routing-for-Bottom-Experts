"""Teacher-forced MoE quality hooks for Qwen and Gemma router contracts.

The baseline leaves native router outputs untouched. Placement selection reuses the
production selector (Triton on CUDA), including its per-rank load penalty and gate
weights from the original unpenalized distribution. Gemma returns probabilities
rather than raw logits; a child hook captures its native projection for the
penalty (log-probabilities are a fallback); its per-expert factors follow the selected experts.

Routers returning `(logits/probs, weights, ids)` have only their selected weights and
ids replaced. Older plain-logit routers receive an exact-width mask and execute their
native top-k downstream. No-op and repeat baselines validate both contracts.
"""

from __future__ import annotations

from contextlib import nullcontext
import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import placement as P

__all__ = ["arms_of", "install", "load", "routing_cfg", "text_cfg"]


def pick(cfg, *names, default: int | None = None) -> int:
    """The first field the config actually spells, as an int.

    An unknown spelling of a routing field is a hard stop: guessing would mis-size the GPU block
    map. `default` is for fields a family may legitimately not have.
    """
    for n in names:
        v = getattr(cfg, n, None)
        if v is None:
            continue
        return int(v[0] if isinstance(v, (list, tuple)) else v)
    if default is not None:
        return default
    raise SystemExit(f"{type(cfg).__name__}: no field among {names}")


def text_cfg(cfg):
    """The config that actually carries the MoE fields, unwrapping a multimodal shell."""
    return getattr(cfg, "text_config", None) or cfg


def routing_cfg(model_path: str, n_gpus: int):
    """`(cfg, n_experts, top_k)` for a checkpoint, with the shardability preconditions checked."""
    cfg = text_cfg(AutoConfig.from_pretrained(model_path, trust_remote_code=True))
    n_experts = pick(cfg, "num_experts", "n_routed_experts")
    top_k = pick(cfg, "num_experts_per_tok", "moe_topk", "top_k_experts")
    assert n_experts % n_gpus == 0, f"{n_experts} experts do not shard over {n_gpus} GPUs"
    assert n_experts // n_gpus >= top_k, "a single GPU must hold at least top_k experts"
    return cfg, n_experts, top_k


def load(model_path: str):
    """The model on all visible GPUs, in bf16, with expert dropping disabled."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    except ValueError:
        # Gemma 4 and Qwen3.5 register under the image-text-to-text head, not causal-LM.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    return model


def routers(model, n_experts: int, hidden: int) -> list[torch.nn.Module]:
    """Every MoE layer's router: the module owning the `[E, H]` logit projection.

    Matched on weight shape *and* a `gate`/`router` path component. Both filters are
    load-bearing - the name alone also matches `gate_proj`, whose weight is `[I, H]`, and an
    `[E, H]` matrix alone is only probably unique.

    Where the match is a projection held inside a router that does the softmax and the top-k
    itself (Gemma 4), the *parent* is returned, so the hook sits above that arithmetic rather
    than feeding masked scores into it. `per_expert_scale` identifies such a parent; it is the
    one attribute only `Gemma4TextRouter` has.
    """
    mods = dict(model.named_modules())
    out = []
    for n, m in mods.items():
        if not (any(p in ("gate", "router") for p in n.split("."))
                and getattr(m, "weight", None) is not None
                and tuple(m.weight.shape) == (n_experts, hidden)):
            continue
        p = mods.get(n.rpartition(".")[0])
        out.append(p if getattr(p, "per_expert_scale", None) is not None else m)
    return out


def _select_on_device(logits, *args, **kwargs):
    """Launch Triton on the layer's device under HF device_map="auto".

    Tensor operations infer their device from their inputs; a Triton launch uses the
    current CUDA device/stream. Accelerate may call a later layer with CUDA 0 still
    current while its logits live on CUDA 1, so enter the logits' device explicitly.
    """
    context = torch.cuda.device(logits.device) if logits.is_cuda else nullcontext()
    with context:
        return P.select(logits, *args, **kwargs)


def install(model, n_experts: int, top_k: int, hidden: int, n_gpus: int,
            trace_tokens: int = 0, parity_tokens: int = 0):
    """Hook every router. Returns `(state, handles)`; mutate `state` to switch arm.

    `state["arm"]` is one of `baseline` / `placement` / `random` and `state["keep"]` is how many
    top ranks are chosen by score. `state["load_penalty"]` is charged per slot already on
    a rank; `placement` at `keep == top_k` and penalty 0 is the no-op arm. `changed`
    and `rows` accumulate the fired fraction, and `trace`, when a dict, collects up to
    `trace_tokens` of routing per layer for the fan-out count.
    """
    gates = routers(model, n_experts, hidden)
    assert gates, f"found no [{n_experts}, {hidden}] router projection"
    print(f"hooking {len(gates)} routers", flush=True)
    # Per-device block map: `device_map="auto"` spreads layers over GPUs, so the logits of
    # different layers arrive on different devices.
    gpu_maps: dict[torch.device, torch.Tensor] = {}
    state: dict = {"arm": "baseline", "keep": 0, "load_penalty": 0.0,
                   "trace": None, "parity": {}, "raw_sources": {}, "changed": 0, "rows": 0,
                   "n_routers": len(gates)}
    order = {id(g): i for i, g in enumerate(gates)}
    projection_logits = {}
    projection_handles = []
    for gate in gates:
        if getattr(gate, "per_expert_scale", None) is not None and hasattr(gate, "proj"):
            def remember_projection(mod, args, out, router_id=id(gate)):
                projection_logits[router_id] = out
            projection_handles.append(gate.proj.register_forward_hook(remember_projection))

    def hook(mod, args, out):
        scores = out[0] if isinstance(out, tuple) else out
        z = scores.reshape(-1, scores.shape[-1])
        scale = getattr(mod, "per_expert_scale", None)
        # Gemma returns probabilities. Prefer its captured native projection logits,
        # exactly what serving uses. log(p) is a fallback up to a row constant;
        # subtracting a penalty directly from probabilities would change the rule.
        projected = projection_logits.pop(id(mod), None)
        if projected is not None:
            raw = projected.reshape(-1, projected.shape[-1]).float()
            state["raw_sources"][order[id(mod)]] = "native_projection_logits"
        elif scale is not None:
            raw = z.float().clamp_min(torch.finfo(torch.float32).tiny).log()
            state["raw_sources"][order[id(mod)]] = "log_native_probabilities"
        else:
            raw = z.float()
            state["raw_sources"][order[id(mod)]] = "native_router_logits"
        base = out[2].reshape(-1, top_k).long() if isinstance(out, tuple) else z.topk(top_k, -1).indices
        gpu = gpu_maps.setdefault(z.device, P.gpu_of(n_experts, n_gpus, z.device))
        arm, keep = state["arm"], state["keep"]
        penalty = float(state.get("load_penalty", 0.0))
        production_val = None
        if arm == "baseline":
            mask = None
            idx = base
        else:
            if arm == "random":
                mask = P.random_mask(raw, keep, top_k)
            elif arm in ("placement", "production_noop"):
                if keep < top_k or arm == "production_noop":
                    if not getattr(mod, "norm_topk_prob", True):
                        raise ValueError("production placement requires normalized top-k weights")
                    production_val, ids = _select_on_device(raw, top_k, keep, n_gpus, scale=scale,
                                                   load_penalty=penalty)
                    mask = torch.zeros_like(raw, dtype=torch.bool).scatter_(1, ids.long(), True)
                else:
                    # No-op tests native selection and weight arithmetic separately.
                    mask = P.allowed_mask(raw, keep, gpu, n_gpus, top_k, penalty, base)
            else:
                raise ValueError(f"unknown routing arm {arm!r}")
            idx = (P.mask_logits(raw, mask).topk(top_k, -1).indices if production_val is None
                   else ids.long())
            state["changed"] += int((idx.sort(-1).values != base.sort(-1).values).any(1).sum())
        state["rows"] += idx.shape[0]
        if arm in ("placement", "production_noop") and parity_tokens > 0:
            # Quantify native BF16 tie-breaking and weight arithmetic differences,
            # while the treatment itself always uses the actual production selector.
            rec = state["parity"].setdefault(order[id(mod)], {
                "rows": 0, "selection_mismatch_rows": 0,
                "weight_max_abs_on_matching_selection": 0.0,
            })
            count = min(parity_tokens - rec["rows"], len(raw))
            if count > 0:
                ref_ids = P.rewrite_torch(raw[:count], base[:count], keep, n_gpus, penalty).long()
                if production_val is None:
                    prod_w, prod_ids = _select_on_device(raw[:count], top_k, keep, n_gpus,
                                                scale=scale, load_penalty=penalty)
                else:
                    prod_w, prod_ids = production_val[:count], idx[:count]
                same = (ref_ids.sort(-1).values == prod_ids.long().sort(-1).values).all(-1)
                previous_rows = rec["rows"]
                rec["mean_row_logit_std"] = (
                    rec.get("mean_row_logit_std", 0.0) * previous_rows
                    + float(raw[:count].std(-1, unbiased=False).mean().detach()) * count
                ) / (previous_rows + count)
                if scale is not None:
                    rec["per_expert_scale_min"] = float(scale.min().detach())
                    rec["per_expert_scale_max"] = float(scale.max().detach())
                rec["rows"] += count
                rec["selection_mismatch_rows"] += int((~same).sum())
                if bool(same.any()):
                    probs = z[:count] if scale is not None else z[:count].float().softmax(-1)
                    ref_w = probs.gather(1, ref_ids)
                    if getattr(mod, "norm_topk_prob", True):
                        ref_w = ref_w / ref_w.sum(-1, keepdim=True)
                    if scale is not None:
                        ref_w = ref_w * scale[ref_ids]
                    ref_w = ref_w.gather(1, ref_ids.argsort(-1))
                    prod_w = prod_w.gather(1, prod_ids.long().argsort(-1))
                    error = float((ref_w[same] - prod_w[same]).abs().max().detach())
                    rec["weight_max_abs_on_matching_selection"] = max(
                        rec["weight_max_abs_on_matching_selection"], error)
        t = state["trace"]
        if t is not None:
            k = order[id(mod)]
            left = trace_tokens - sum(x.shape[0] for x in t.setdefault(k, []))
            if left > 0:
                t[k].append(idx[:left].to(torch.int16).cpu())

        if mask is None:
            # Stock must run its native arithmetic, including router tie-breaking.
            return out
        if not isinstance(out, tuple):
            return P.mask_logits(z, mask).reshape(scores.shape)
        if production_val is not None:
            return (scores, production_val.to(out[1].dtype).reshape(out[1].shape),
                    idx.to(out[2].dtype).reshape(out[2].shape))
        # Selection changes, but weights always come from the original unpenalized
        # distribution. Preserve the family's native probability and scale arithmetic.
        probs = z if scale is not None else torch.nn.functional.softmax(z, dtype=torch.float, dim=-1)
        val = probs.gather(-1, idx)
        if getattr(mod, "norm_topk_prob", True):
            val = val / val.sum(dim=-1, keepdim=True)
        if scale is not None:
            val = val * scale[idx]
        # The router's full logits/probabilities are still useful for auxiliary outputs;
        # only the selected weights and ids need replacing in the tuple contract.
        return (scores, val.to(out[1].dtype).reshape(out[1].shape),
                idx.to(out[2].dtype).reshape(out[2].shape))

    return state, projection_handles + [g.register_forward_hook(hook) for g in gates]


def arms_of(top_k: int, keeps, penalties=None):
    """Controls then `(name, arm, keep, penalty)` treatment arms.

    Omitting `penalties` preserves the legacy three-field interface for external callers.
    Explicit penalties form a grid for placement; the random control runs once per keep.
    Zero-penalty arm names retain the historical `placement_6` spelling.
    """
    legacy = penalties is None
    penalties = (0.0,) if legacy else tuple(dict.fromkeys(float(p) for p in penalties))
    keeps = tuple(dict.fromkeys(int(k) for k in keeps))
    if any(not 1 <= k < top_k for k in keeps):
        raise ValueError(f"KEEPS must be between 1 and top_k - 1 ({top_k - 1})")
    if any(not math.isfinite(p) or p < 0 for p in penalties):
        raise ValueError("PENS must contain finite, nonnegative penalties")
    arms = [("baseline", "baseline", 0, 0.0), ("baseline2", "baseline", 0, 0.0),
            ("noop", "placement", top_k, 0.0)]
    arms += [(f"placement_{k}" + (f"p{p:g}" if p else ""), "placement", k, p)
             for k in keeps for p in penalties]
    arms += [(f"random_{k}", "random", k, 0.0) for k in keeps]
    return [a[:3] for a in arms] if legacy else arms
