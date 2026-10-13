"""Installing the placement-aware router into a stock vLLM, in place.

vLLM v0.31 routes through a small class hierarchy: `BaseRouter._select_experts` validates the
EPLB state, delegates to a subclass `_compute_routing`, maps logical expert ids to physical
ones and fixes the index dtype. Every fused-MoE layer in the engine goes through it.

Nearly every MoE model served by vLLM routes through `FusedTopKRouter`, and for those this
plugin swaps in `par.vendor.fused_topk_router` - that same unit copied in with a three-line
insert, so the rewrite happens inside the routing unit and emits its own gate weights. One
Triton launch per MoE layer, and nothing else added to the forward.

`Gemma4Router` is the exception. It routes through `CustomRoutingRouter`, whose routing
function lives in the model file and folds a learned per-expert output scale into its gate
weights, so neither the unit nor the weights can be reproduced here. Gemma keeps the slower
but exact route: rewrite the ids, then hand the selection back to the router's own routing
function as logits with every other expert at the dtype floor. That costs a `[T, E]` mask and
a second routing pass per layer, which is the price of not reimplementing someone's gate.

`PAR_KEEP` turns the whole thing on; unset, nothing is patched at all. Either path passes the
selection through unchanged, once with a warning, wherever the rule does not apply:

* **EPLB** replicates experts across ranks, so expert id no longer implies a rank and the
  contiguous map the rule depends on is gone.
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

import os

import torch

from par.kernel import rewrite

_KEEP: int | None = None
_EP: int | None = None
_MISS = object()
_warned: set[str] = set()


def _logger():
    from vllm.logger import init_logger

    return init_logger("par")


def _warn(why: str) -> None:
    _once(why, "par: routing left unchanged - %s" % why, warn=True)


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


def par_keep(router, router_logits) -> int | None:
    """`keep` if the rule applies to this router, `None` if it does not.

    Called from inside the routing unit, so per MoE layer per forward. Nothing `_unsupported`
    looks at can change between forwards - top_k, renormalize, the EPLB state, the logit width
    and the rank count are all fixed once the engine is up - so the answer is decided on the
    first forward and cached on the router. After that this is one dict lookup.
    """
    k = router.__dict__.get("_par_keep", _MISS)
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
            # actually running, and `logger.info` is below the root logger's level in a stock
            # vLLM image, so an info-level version of it is invisible exactly where it matters.
            _once("active", "par: placement-aware routing active, keep=%d of top_k=%d, "
                            "ep_size=%d" % (k, router.top_k, ep_size()), warn=True)
    router._par_keep = k
    return k


def _shim_remote_code() -> None:
    """Hunyuan's vendored modelling file imports a transformers 4.x symbol that 5.x deleted.

    vLLM 0.31 has no native Hunyuan-MoE implementation and serves it through the Transformers
    backend, so `trust_remote_code` executes that file in every worker process - and a plugin
    entry point is the only hook that runs there too. `False` skips the `torch.fx.wrap` branch
    the symbol guards, which vLLM never traces anyway.
    """
    from transformers.utils import import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False


def register() -> None:
    """vLLM `general_plugins` entry point. Idempotent, and a no-op without `PAR_KEEP`."""
    global _KEEP
    _shim_remote_code()
    if os.environ.get("PAR_KEEP") is None:
        return

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    from par.vendor.fused_topk_router import FusedTopKRouter as ParFusedTopKRouter

    if getattr(BaseRouter, "_par_patched", False):
        return
    _KEEP = int(os.environ["PAR_KEEP"])
    FusedTopKRouter._compute_routing = ParFusedTopKRouter._compute_routing
    BaseRouter._select_experts = _masked_select_experts(BaseRouter._select_experts)
    BaseRouter._par_patched = True
    # Only that the patch is in. Whether it *applies* is decided per router in the first
    # forward, and the expert-parallel group does not exist yet at plugin load time.
    _logger().info("par: router patched, keep=%d", _KEEP)


def _masked_select_experts(inner):
    """Gemma's path: rewrite the ids, let its own routing function weight them.

    Only for a listed `CustomRoutingRouter` - `FusedTopKRouter` is already handled inside the
    vendored routing unit, and anything else passes through. Kept exact rather than fast
    because the gate weights here are not a plain renormalized softmax.
    """
    from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
        CustomRoutingRouter)

    def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *,
                        input_ids=None):
        topk_weights, topk_ids = inner(self, hidden_states, router_logits,
                                       topk_indices_dtype=topk_indices_dtype,
                                       input_ids=input_ids)
        if not isinstance(self, CustomRoutingRouter):
            return topk_weights, topk_ids
        keep = par_keep(self, router_logits)
        if keep is None:
            return topk_weights, topk_ids
        # The kernel's own weights are discarded here: a masked re-run through the router's
        # routing function is the only way to keep Gemma's per-expert output scale. Its top-k
        # can only resolve the mask to `ids`, and renormalization over the selection (guarded
        # in `_unsupported`) divides out the softmax denominator the mask changed.
        ids = rewrite(router_logits, topk_ids, topk_weights, keep, ep_size())[0].long()
        # vLLM marks cudagraph and DP padding by forcing a token's expert ids to -1 so the
        # dispatch and the experts drop it (`VLLM_MOE_SKIP_PADDING`, on by default). The kernel
        # passes that sentinel through; a masked re-run cannot, since its top-k has to return
        # real experts. So clamp for the gather and put the untouched rows back afterwards -
        # routing a padded token would both add dispatch volume this arm does not owe and make
        # `keep == top_k` stop being a no-op.
        pad = (ids.amin(1) < 0).unsqueeze(1)
        safe = ids.clamp_min(0)
        masked = torch.full_like(router_logits, torch.finfo(router_logits.dtype).min)
        masked.scatter_(1, safe, router_logits.gather(1, safe))
        w, out = self._compute_routing(hidden_states, masked, topk_indices_dtype,
                                       input_ids=input_ids)
        out = self._convert_indices_dtype(out, topk_indices_dtype)
        return torch.where(pad, topk_weights, w), torch.where(pad, topk_ids.to(out.dtype), out)

    return _select_experts


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
    if router.eplb_state is not None:
        return "EPLB replication breaks the expert-to-rank map"
    k, e = router.top_k, router.global_num_experts
    # `keep == k` is allowed through on purpose. The rule then asks for no substitution and
    # the selection is provably unchanged, so that arm measures what this plugin costs to run
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
    return None
