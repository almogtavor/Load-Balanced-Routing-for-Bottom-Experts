"""Finding and overriding a MoE router, across four families that agree on nothing.

`accuracy.py` scores perplexity and `bench.py` scores a benchmark, but both need the exact
same intervention: find every router, and make its top-k return the set `placement.py` chose.
That intervention is subtle enough - three different gate contracts, four config vocabularies -
that having it in one place is the only way both scripts are measuring the same thing.

The families differ in three ways that each need handling:

* **where the config lives.** Qwen3.5 and Gemma 4 ship as multimodal wrappers whose
  `text_config` carries the routing; Qwen3-30B and Hunyuan keep it at the top level.
* **what the router module is.** Hunyuan's is an `nn.Linear` (`mlp.gate.wg`) returning plain
  logits; transformers 5 made Qwen's a `*TopKRouter` holding a bare `nn.Parameter` and
  returning `(logits, scores, indices)`; Gemma 4's `Gemma4TextRouter` returns
  `(probs, weights, indices)` and does its projection in a child `nn.Linear` (`router.proj`).
  The child is the tempting hook - it puts the whole top-k downstream, so the plain-logits path
  appears to cover it - and it is the wrong one: Gemma's own `softmax -> topk -> /= sum ->
  * per_expert_scale` chain then runs on *masked* scores and reaches the same answer by a
  different sequence of divisions, which in bf16 left the no-op arm 0.02 nat off the baseline
  and flipped 12 benchmark items. So the parent is hooked and its tuple rebuilt here, exactly
  as the Qwen routers are.
* **what happens after the projection.** Every family softmaxes, takes top-k and renormalizes
  by the top-k sum. The renormalization is what makes logit masking exact rather than
  approximate: masking changes the softmax denominator, and dividing by the top-k sum divides
  that change straight back out - in exact arithmetic. It cancels to rounding in floating
  point, so on the families whose softmax is downstream of this hook the no-op arm lands a
  fraction of a millinat off zero rather than on it; `accuracy.py` bounds that rather than
  demanding a zero it cannot have. Gemma's extra `per_expert_scale` is gathered at the chosen
  indices, so it follows the selection rather than being perturbed by the mask.

Every family that returns its selection needs it rebuilt here, because masked logits cannot
reach it otherwise: that is the transformers-5 Qwen routers and Gemma 4's.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import placement as P

__all__ = ["arms_of", "install", "load", "moe_shapes", "routing_cfg", "text_cfg"]


def pick(cfg, *names, default: int | None = None) -> int:
    """The first field the config actually spells, as an int.

    Hunyuan stores its MoE fields as single-element lists (one entry per layer group) where the
    others store ints. For the routing fields, guessing wrong would mis-size the GPU block map
    and quietly route to the wrong devices, so an unknown spelling is a hard stop; `default` is
    for fields a family may legitimately not have at all, such as a shared expert.
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


def moe_shapes(cfg) -> dict:
    """The widths Phase B needs to rebuild one expert-parallel layer.

    `shared_inter` is the total width of the **always-on** path, and it is load-bearing rather
    than cosmetic: it adds compute on the token's own GPU and no communication at all, so
    omitting it inflates the communication fraction and overstates this experiment's benefit on
    exactly the models that have one. Hunyuan spells it as a shared expert; Gemma 4 does not
    call it that but every MoE layer still runs a full dense MLP beside the experts, which is
    the same thing for this accounting.
    """
    moe_inter = pick(cfg, "moe_intermediate_size", "intermediate_size")
    shared = (pick(cfg, "shared_expert_intermediate_size", default=0)
              or pick(cfg, "num_shared_expert", "n_shared_experts", default=0) * moe_inter
              or (pick(cfg, "intermediate_size", default=0)
                  if getattr(cfg, "enable_moe_block", False) else 0))
    return {"hidden": int(cfg.hidden_size), "moe_inter": moe_inter, "shared_inter": shared}


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
    # Scoring is teacher-forced everywhere in this repo, so the cache is only ever dead weight
    # - but turning it off is not free on every family, so it is turned off only where it is a
    # workaround. Hunyuan's vendored modelling code calls a `DynamicCache` method transformers
    # has since removed, so it cannot run with one. Gemma 4 can: forcing it off there moved the
    # baseline NLL from 5.854 to 10.750 and the benchmark to chance, because its sliding-window
    # layers build their attention mask down a different path when there is no cache to measure
    # against.
    if "hunyuan" in f"{type(model).__name__} {getattr(model.config, 'model_type', '')}".lower():
        model.config.use_cache = False
    import hunyuan_capacity
    print(f"capacity patch applied: {hunyuan_capacity.no_drop_moe(model)}", flush=True)
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


def install(model, n_experts: int, top_k: int, hidden: int, n_gpus: int,
            trace_tokens: int = 0):
    """Hook every router. Returns `(state, handles)`; mutate `state` to switch arm.

    `state["arm"]` is one of `baseline` / `placement` / `random` and `state["keep"]` is how many
    top ranks are chosen by score; `placement` at `keep == top_k` is the no-op arm. `changed`
    and `rows` accumulate the fired fraction, and `trace`, when a dict, collects up to
    `trace_tokens` of routing per layer for Phase B to replay.
    """
    gates = routers(model, n_experts, hidden)
    assert gates, f"found no [{n_experts}, {hidden}] router projection"
    print(f"hooking {len(gates)} routers", flush=True)
    # Per-device block map: `device_map="auto"` spreads layers over GPUs, so the logits of
    # different layers arrive on different devices.
    gpu_maps: dict[torch.device, torch.Tensor] = {}
    state: dict = {"arm": "baseline", "keep": 0, "trace": None, "changed": 0, "rows": 0,
                   "n_routers": len(gates)}
    order = {id(g): i for i, g in enumerate(gates)}

    def hook(mod, args, out):
        logits = out[0] if isinstance(out, tuple) else out
        z = logits.reshape(-1, logits.shape[-1])
        gpu = gpu_maps.setdefault(z.device, P.gpu_of(n_experts, n_gpus, z.device))
        arm, keep = state["arm"], state["keep"]
        if arm == "baseline":
            m = None
        elif arm == "random":
            m = P.random_mask(z, keep, top_k)
        else:  # placement, and the no-op arm which is placement at keep == top_k
            m = P.allowed_mask(z, keep, gpu, n_gpus, top_k)
        z2 = z if m is None else P.mask_logits(z, m)

        idx = z2.topk(top_k, -1).indices
        if m is not None:
            base = z.topk(top_k, -1).indices
            state["changed"] += int((idx.sort(-1).values != base.sort(-1).values).any(1).sum())
            state["rows"] += idx.shape[0]
        t = state["trace"]
        if t is not None:
            k = order[id(mod)]
            if sum(x.shape[0] for x in t.setdefault(k, [])) < trace_tokens:
                t[k].append(idx[:trace_tokens].to(torch.int16).cpu())

        if not isinstance(out, tuple):
            return z2.reshape(logits.shape)
        # The transformers-5 Qwen routers moved the softmax, the top-k and the renormalization
        # *inside* the router and return the selection, so masking the logits is not enough and
        # it has to be rebuilt here, in the module's own order. Gathering instead of a second
        # top-k is exact: softmax is monotone, so it cannot reorder the head.
        #
        # The weights come off the **unmasked** logits `z`, not `z2`. That is what the rule
        # actually says - the intervention changes which experts a token is sent to, and each
        # one still carries the gate weight the gate itself would have given it - and a real
        # serving implementation would pick different indices rather than mask anything. It is
        # also what keeps the no-op arm bit-exact: `softmax(z2)` renormalized over the top-k is
        # the same number in exact arithmetic, but it reaches it by a different division, and
        # the last-bit disagreement survives the cast to bf16 often enough to move NLL.
        # Gemma's router already returns fp32 probabilities in `out[0]`, so there is no softmax
        # left to apply - and softmax is monotone, so masking, ranking and top-k behave
        # identically on probabilities and on logits. Its per-expert scale is gathered at the
        # chosen indices, so it follows the selection rather than being perturbed by the mask.
        scale = getattr(mod, "per_expert_scale", None)
        val = (z if scale is not None
               else torch.nn.functional.softmax(z, dtype=torch.float, dim=-1)).gather(-1, idx)
        if getattr(mod, "norm_topk_prob", True):
            val /= val.sum(dim=-1, keepdim=True)
        if scale is not None:
            val = val * scale[idx]
        return (z2.reshape(logits.shape), val.to(logits.dtype), idx)

    return state, [g.register_forward_hook(hook) for g in gates]


def arms_of(top_k: int, keeps) -> list[tuple[str, str, int]]:
    """`(name, arm, keep)` for every arm, in the order they must run.

    The baseline comes first because every other arm is reported as a paired difference from
    it, and the no-op second because it is the assertion that the hook is inert when the rule
    asks for nothing.

    `baseline2` repeats the baseline arm verbatim. Nothing about the rule needs it: it is there
    because the no-op arm's residual has to be compared against something, and "bf16 rounding
    is small" is an argument rather than a measurement. A repeat pass re-runs the same kernels
    on the same inputs, so whatever it reports - zero, or the non-determinism of a reduction
    order - is this instrument's own floor, and the no-op arm is only interesting where it
    exceeds that floor.
    """
    return ([("baseline", "baseline", 0), ("baseline2", "baseline", 0),
             ("noop", "placement", top_k)]
            + [(f"placement_{k}", "placement", k) for k in keeps]
            + [(f"random_{k}", "random", k) for k in keeps])
