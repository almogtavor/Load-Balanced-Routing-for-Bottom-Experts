"""Installing the placement-aware router into default vLLM, in place.

vLLM v0.31 routes through a small class hierarchy: `BaseRouter._select_experts` validates the
EPLB state, delegates to a subclass `_compute_routing`, maps logical expert ids to physical
ones and fixes the index dtype. Every fused-MoE layer in the engine goes through it.

Nearly every MoE model served by vLLM routes through `FusedTopKRouter`, and for those this
plugin swaps in `placement_router.vendor.fused_topk_router` - that same unit copied in with a short
insert, where the kernel *replaces* the fused top-k rather than running after it. One Triton
launch per MoE layer, exactly where default vLLM also spends one, so nothing is added to the
forward and `keep == top_k` is a control setting that costs nothing measurable.

`Gemma4Router` is the exception. It routes through `CustomRoutingRouter`, whose routing
function lives in the model file, so the unit itself cannot be swapped - but its
`_compute_routing` is a thin call into that function and can be, which keeps Gemma to one
launch too. The function is a softmax over every expert, masked to the selection and
renormalized over it - which is the selection-scoped softmax the kernel already emits - times
a learned per-expert output scale, which the kernel takes as an argument and folds in.

`PLACEMENT_KEEP` turns the whole thing on; unset, nothing is patched at all. The other settings:

* `PLACEMENT_LOAD_PENALTY`: the tile load penalty, 0 for the plain rule.
* `PLACEMENT_MASS`, `PLACEMENT_MASS_NORM`: the mass rule, a per-token head that holds at least
  `PLACEMENT_KEEP` picks and this share of the softmax (with `MASS_NORM=1`, of default vLLM's
  renormalized top-k weights).
* `PLACEMENT_ANY_RANK=1`: the tail from every rank, picked by the load penalty alone.
* `PLACEMENT_TAU`, `PLACEMENT_RANGE`: the sampled tail of Expert-Sample, the control.
* `PLACEMENT_PREFILL_ONLY=1`: decode tokens keep default routing.
* `PLACEMENT_COACT`, `PLACEMENT_COACT_LOAD=1`: an expert-to-rank placement from
  `experiments/coact_place.py`, for the tail and optionally for loading the experts.
* `PLACEMENT_EPLB=1`: makes vLLM 0.31's Gemma-4 report its load to EPLB (`_gemma_eplb`).

Either path passes the selection through unchanged, once with a warning, wherever the rule does
not apply:

* **EPLB** replicates experts across ranks, so expert id no longer implies a rank and the
  contiguous map the rule depends on is gone. The any-rank tail needs no such map and runs on
  top of EPLB when `PLACEMENT_EPLB=1`.
* **grouped top-k, score-correction bias, zero-expert and unrecognised custom routing** rank
  by something other than the gate logits, so "the top `keep` by logit" is not the head the
  router chose.
* **routing without renormalization** weights each expert by the softmax over all of them,
  which is not recoverable from the selection alone.
* **a rank count of 1** has no all-to-all to shrink, and a non-divisible or too-small expert
  block breaks the guarantee that a hot rank always has a substitute left to offer.

Passing through is the right failure: a routing rule that silently applied where its premise
does not hold would be measured as free and serve garbage.
"""

from __future__ import annotations

import functools
import os

from placement_router.kernel import select

_KEEP: int | None = None
_LOAD_PENALTY = 0.0
_MASS = 0.0
_MASS_NORM = False
_ANY_RANK = False
_PREFILL_ONLY = False
_COACT = None
_COACT_LOAD = False
_SAMPLE: tuple[float, int] | None = None
_EP: int | None = None
_MISS = object()
_warned: set[str] = set()


def _logger():
    from vllm.logger import init_logger

    return init_logger("placement-router")


def _warn(why: str) -> None:
    _once(why, "placement-router: routing left unchanged - %s" % why, warn=True)


def _once(key: str, msg: str, warn: bool = False) -> None:
    if key not in _warned:
        _warned.add(key)
        log = _logger()
        (log.warning if warn else log.info)(msg)


def ep_size() -> int:
    """The expert-parallel world size. Fixed for the engine's life, so read once."""
    global _EP
    if _EP is None:
        from vllm.distributed.parallel_state import get_ep_group

        _EP = get_ep_group().world_size
    return _EP


def keep_for(router, router_logits) -> int | None:
    """`keep` if the rule applies to this router, `None` if it does not.

    Called from inside the routing unit, so per MoE layer per forward. Nothing `_unsupported`
    looks at can change between forwards - top_k, renormalize, the EPLB state, the logit width
    and the rank count are all fixed once the engine is up - so the answer is decided on the
    first forward and cached on the router. After that this is one dict lookup.
    """
    k = router.__dict__.get("_keep_for", _MISS)
    if k is not _MISS:
        return k
    k = _KEEP
    if k is not None:
        why = _unsupported(router, router_logits, k)
        if why is not None:
            _warn(why)
            k = None
        else:
            # At warning level on purpose: this is the one line that says the kernel is
            # actually running, and `logger.info` is below the root logger's level in a default
            # vLLM image, so an info-level version of it is invisible exactly where it matters.
            _once("active", "placement-router: placement-aware routing active, keep=%d of top_k=%d, "
                            "ep_size=%d" % (k, router.top_k, ep_size()), warn=True)
    router._keep_for = k
    return k


def load_penalty() -> float:
    """`PLACEMENT_LOAD_PENALTY`: what a tail pick pays per pick already on its rank, per token of its tile.

    0.0 is the plain rule - minimise summed volume. Anything higher trades none of that volume
    (the tail stays confined to the head's ranks either way) for a flatter spread over them,
    which is what the step actually waits on. The count is over a tile of the batch, not one
    token. See the kernel docstring.
    """
    return _LOAD_PENALTY


def mass() -> float:
    """`PLACEMENT_MASS`: the softmax mass a token's head must hold; `PLACEMENT_KEEP` is its floor.

    0.0 is off, and the head is `keep` picks for every token. Above it a confident token keeps
    few picks and replaces more of its tail, an uncertain one less. See `rewrite_torch`.
    """
    return _MASS


def mass_norm() -> bool:
    """`PLACEMENT_MASS_NORM`: whether `mass()` is over default vLLM's renormalized top-k weights."""
    return _MASS_NORM


def any_rank() -> bool:
    """`PLACEMENT_ANY_RANK=1`: the tail drawn from every rank, by the load penalty alone."""
    return _ANY_RANK


def sampled_tail() -> tuple[float, int] | None:
    """`(PLACEMENT_TAU, PLACEMENT_RANGE)` for Expert-Sample's sampled tail, `None` when off."""
    return _SAMPLE


def decode_rows(t: int):
    """`PLACEMENT_PREFILL_ONLY=1`: which of this forward's `t` rows are decode tokens.

    `None` with the setting off, `True` for an all-decode forward, else a `[t]` bool mask. Under
    P/D disaggregation the decode node runs default vLLM; a single server emulates that by
    giving decode rows (query length 1) default routing. Computed once per forward and cached
    on the forward context; no host sync.
    """
    if not _PREFILL_ONLY:
        return None
    import torch
    from vllm.forward_context import get_forward_context
    ctx = get_forward_context()
    hit = ctx.__dict__.get("_placement_decode", _MISS)
    if hit is not _MISS and hit[0] == t:
        return hit[1]
    md = ctx.attn_metadata
    md = (md[0] if isinstance(md, list) else md) or {}
    # Hybrid models (Qwen3.5) also carry linear-attention metadata without `max_query_len`.
    md = next((m for m in md.values() if hasattr(m, "max_query_len")), None)
    if md is None:
        # Profiling and DP dummy forwards carry no attention metadata and no real tokens.
        return None
    if md.max_query_len == 1:
        dec = True
    else:
        qsl = md.query_start_loc
        rows = torch.arange(t, device=qsl.device, dtype=qsl.dtype)
        req = torch.searchsorted(qsl[1:], rows, right=True).clamp_(max=len(qsl) - 2)
        dec = ((qsl[req + 1] - qsl[req]) == 1) & (rows < qsl[-1])
    ctx._placement_decode = (t, dec)
    return dec


def coact_perm(router, device):
    """`PLACEMENT_COACT=<[layers, E] expert -> rank .npy>`: the rule's ranks from that placement.

    `None` when unset. Else a `[E]` index such that `logits[:, perm]` puts the experts of rank
    g at ids g*E/G..; the kernel then reads the placement off id blocks and `perm[ids]` maps
    its picks back. The weights stay where vLLM loaded them, so this measures the accuracy of
    routing under that placement, not its throughput. Layers are numbered by first call,
    which is the profiling forward's layer order.
    """
    global _COACT
    if not os.environ.get("PLACEMENT_COACT"):
        return None
    perm = router.__dict__.get("_coact_perm")
    if perm is not None:
        return perm
    import numpy as np
    import torch
    if _COACT is None:
        _COACT = [np.load(os.environ["PLACEMENT_COACT"]), 0]
    place, layer = _COACT
    assert layer < len(place), f"more MoE layers than the {len(place)} in PLACEMENT_COACT"
    _COACT[1] += 1
    perm = torch.as_tensor(np.lexsort((np.arange(place.shape[1]), place[layer])), device=device)
    router._coact_perm = perm
    _once("coact", "placement-router: co-activation placement from %s" % os.environ["PLACEMENT_COACT"], warn=True)
    return perm


def coact_loaded() -> bool:
    """`PLACEMENT_COACT_LOAD=1`: the expert weights follow the placement too (`_coact_loader`)."""
    return _COACT_LOAD


def _coact_loader(inner):
    """Load expert e of layer l at id `argsort(perm)[e]`, `perm` as in `coact_perm`.

    Rank g then holds the experts the placement puts on g, at the ids the permuted logits route
    to: the same model with its experts stored in placement order, so a served run measures
    the placement's throughput as well as its accuracy, with no permuted checkpoint on disk.
    """
    import re

    import numpy as np
    place = np.load(os.environ["PLACEMENT_COACT"])
    pos = [np.argsort(np.lexsort((np.arange(place.shape[1]), p))) for p in place]

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        layer = int(re.search(r"layers\.(\d+)\.", self.layer_name)[1])
        return inner(self, param, loaded_weight, weight_name, shard_id, int(pos[layer][expert_id]),
                     return_success)
    return weight_loader


def register() -> None:
    """vLLM `general_plugins` entry point. Idempotent, and a no-op without `PLACEMENT_KEEP`."""
    global _KEEP, _LOAD_PENALTY, _MASS, _MASS_NORM, _ANY_RANK, _PREFILL_ONLY, _COACT_LOAD, _SAMPLE
    # A measurement hook, loaded by path so nothing in this package depends on it. The model
    # runs in worker *processes*; a monkeypatch applied in the driver never reaches them, so an
    # in-engine probe has to be installed from here, which is where every worker arrives.
    if (probe := os.environ.get("PLACEMENT_PROBE")):
        import importlib.util

        spec = importlib.util.spec_from_file_location("placement_probe", probe)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.install()
    if os.environ.get("PLACEMENT_EPLB") == "1":
        _gemma_eplb()
    if os.environ.get("PLACEMENT_KEEP") is None:
        return

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    from placement_router.vendor.fused_topk_router import FusedTopKRouter as PlacementFusedTopKRouter

    if getattr(BaseRouter, "_placement_patched", False):
        return
    _KEEP = int(os.environ["PLACEMENT_KEEP"])
    _LOAD_PENALTY = float(os.environ.get("PLACEMENT_LOAD_PENALTY", 0.0))
    _MASS = float(os.environ.get("PLACEMENT_MASS") or 0.0)
    _MASS_NORM = os.environ.get("PLACEMENT_MASS_NORM") == "1"
    _ANY_RANK = os.environ.get("PLACEMENT_ANY_RANK") == "1"
    _PREFILL_ONLY = os.environ.get("PLACEMENT_PREFILL_ONLY") == "1"
    if int(os.environ.get("PLACEMENT_RANGE") or 0):
        _SAMPLE = (float(os.environ["PLACEMENT_TAU"]), int(os.environ["PLACEMENT_RANGE"]))
    if os.environ.get("PLACEMENT_COACT") and os.environ.get("PLACEMENT_COACT_LOAD") == "1":
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        RoutedExperts.weight_loader = _coact_loader(RoutedExperts.weight_loader)
        _COACT_LOAD = True
    FusedTopKRouter._compute_routing = PlacementFusedTopKRouter._compute_routing
    CustomRoutingRouter._compute_routing = _gemma_compute_routing(
        CustomRoutingRouter._compute_routing)
    BaseRouter._placement_patched = True
    # Only that the patch is in. Whether it *applies* is decided per router in the first
    # forward, and the expert-parallel group does not exist yet at plugin load time.
    _logger().info("placement-router: router patched, keep=%d load_penalty=%g mass=%g "
                   "mass_norm=%d any_rank=%d prefill_only=%d", _KEEP, _LOAD_PENALTY, _MASS,
                   _MASS_NORM, _ANY_RANK, _PREFILL_ONLY)


def _gemma_eplb() -> None:
    """vLLM 0.31's Gemma-4 builds its MoE layers without `enable_eplb`, so EPLB records no load
    and never rebalances. Pass EPLB's settings through, report the physical expert counts, and
    load the redundant copies, which Gemma's fused checkpoint tensors never reach. Only for the
    `eplb` setting (`PLACEMENT_EPLB`); other models pass through the factory and count patches."""
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.models import gemma4

    if getattr(gemma4, "_eplb_patched", False):
        return
    factory, init, loader = (gemma4.FusedMoEFactory, gemma4.Gemma4ForCausalLM.__init__,
                             RoutedExperts.weight_loader)

    def eplb_factory(*args, **kwargs):
        pc = get_current_vllm_config().parallel_config
        if pc.enable_eplb:
            kwargs.update(enable_eplb=True,
                          num_redundant_experts=pc.eplb_config.num_redundant_experts)
        return factory(*args, **kwargs)

    @functools.wraps(init)  # vLLM reads `vllm_config` from the signature
    def eplb_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if self.moe_layers:
            c = self.moe_layers[0].moe_config
            self.num_physical_experts = c.num_experts
            self.num_local_physical_experts = c.num_local_experts
            self.num_redundant_experts = c.num_experts - c.num_logical_experts

    def eplb_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                    return_success=False):
        ok = loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)
        # EplbState's initial map puts the copy of logical expert e at physical E + e.
        c = self.moe_config
        if expert_id < c.num_experts - c.num_logical_experts:
            ok = loader(self, param, loaded_weight, weight_name, shard_id,
                        c.num_logical_experts + expert_id, return_success) or ok
        return ok

    gemma4.FusedMoEFactory = eplb_factory
    gemma4.Gemma4ForCausalLM.__init__ = eplb_init
    RoutedExperts.weight_loader = eplb_loader
    gemma4._eplb_patched = True


def _gemma_compute_routing(inner):
    """Gemma's path: the kernel replaces its routing function, per-expert scale folded in.

    Only for a listed `CustomRoutingRouter` - `FusedTopKRouter` is handled inside the vendored
    routing unit, and anything else passes through. Being listed in `_LOGIT_TOPK` means both
    that the function selects by gate logit alone and that its weights are the
    selection-scoped softmax times `per_expert_scale`, which is what this reproduces.

    No padding mask, unlike the `FusedTopKRouter` path: Gemma's own routing kernel carries
    none and routes padding tokens like any other, so dropping them here would credit this setting
    with dispatch volume default vLLM pays.
    """

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        keep = keep_for(self, router_logits)
        if keep is None:
            return inner(self, hidden_states, router_logits, indices_type,
                         input_ids=input_ids)
        # Read off the router module at call time for the same reason the model file does: a
        # `functional_call` parameter substitution has to reach it.
        scale = self.custom_routing_function.__self__.per_expert_scale
        return select(router_logits, self.top_k, keep, ep_size(), scale=scale,
                      indices_dtype=indices_type, load_penalty=_LOAD_PENALTY,
                      mass=_MASS, mass_norm=_MASS_NORM, any_rank=_ANY_RANK)

    return _compute_routing


# Custom routing functions whose *selection* is a plain top-k over the gate logits, which is
# the rule's premise. Checked by name because that is what the model files expose; anything
# not listed passes through rather than being assumed equivalent. `Gemma4Router` is
# softmax-over-all, top-k, renormalize, with a per-expert output scale folded in afterwards.
_LOGIT_TOPK = frozenset({"Gemma4Router.routing_function"})


def _unsupported(router, router_logits, keep: int) -> str | None:
    """Why the rule does not apply to this router, or `None` if it does."""
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    if isinstance(router, CustomRoutingRouter):
        name = getattr(router.custom_routing_function, "__qualname__", "?")
        if name not in _LOGIT_TOPK:
            return f"custom routing function {name} may not rank by gate logits alone"
    elif not isinstance(router, FusedTopKRouter):
        return f"{type(router).__name__} does not rank by gate logits alone"
    elif router.scoring_func != "softmax":
        return f"scoring_func={router.scoring_func!r}"
    if not router.renormalize:
        # Without it the weights are the full softmax read at the selection, and the kernel's
        # softmax is scoped to the selection alone.
        return "routing without renormalization"
    # `-with-eplb` (`PLACEMENT_EPLB`): any-rank only, which needs no expert-to-rank map for the tail.
    # Its load penalty still counts picks on the logical expert blocks, not on EPLB's replicas.
    with_eplb = router.eplb_state is not None and _ANY_RANK and os.environ.get("PLACEMENT_EPLB") == "1"
    if router.eplb_state is not None and not with_eplb:
        return "EPLB replication breaks the expert-to-rank map"
    # Under EPLB `global_num_experts` counts the redundant copies; the logits hold the logical experts.
    k, e = router.top_k, router_logits.shape[-1] if with_eplb else router.global_num_experts
    # `keep == k` is allowed through on purpose. The rule then asks for no substitution and
    # the selection is provably unchanged, so that setting measures what this plugin costs to run
    # and nothing else - which is the only way to separate the plumbing from the rule.
    if keep > k:
        return f"keep={keep} > top_k={k}"
    if router_logits.shape[-1] != e:
        return f"router_logits width {router_logits.shape[-1]} != {e} experts"
    g = ep_size()
    if g < 2:
        return f"ep_size={g}, no all-to-all to shrink"
    if e % g or e // g < k:
        return f"{e} experts over {g} ranks leaves no substitutes"
    if g > 32:
        # The kernel tracks which ranks the head visits in one int32 lane.
        return f"ep_size={g} exceeds the kernel's 32-rank mask"
    return None
