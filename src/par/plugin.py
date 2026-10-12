"""Swapping vLLM's MoE routing for the placement-aware one, in place.

vLLM v0.31 routes through a small class hierarchy: `BaseRouter._select_experts` validates the
EPLB state, delegates to a subclass `_compute_routing`, maps logical expert ids to physical
ones and fixes the index dtype. Every fused-MoE layer in the engine goes through it.

So this plugin wraps that one method rather than reimplementing any of it: the model's own
router computes its own top-k, and the wrapper rewrites the bottom `k - keep` ids to experts
on ranks the head already visits (`par.kernel`), then regathers the gate weights at the new
indices. The expert GEMMs, the dispatch, the combine and every other routing variant are
untouched, which is the only reason this is safe to drop into a stock image.

`PAR_KEEP` turns it on; unset, nothing is patched at all. The wrapper also passes the
selection through unchanged, once with a warning, wherever the rule does not apply:

* **EPLB** replicates experts across ranks, so expert id no longer implies a rank and the
  contiguous map the rule depends on is gone.
* **grouped top-k, score-correction bias and zero-expert routing** rank by something other
  than the gate logits, so "the top `keep` by logit" is not the head the router chose.
* **a rank count of 1** has no all-to-all to shrink, and a non-divisible or too-small expert
  block breaks the guarantee that a hot rank always has a substitute left to offer.

Passing through is the right failure: a routing rule that silently applies where its premise
does not hold would be measured as free and serve garbage.
"""

from __future__ import annotations

import os

from par.kernel import gate_weights, rewrite

_warned: set[str] = set()


def _warn(logger, why: str) -> None:
    if why not in _warned:
        _warned.add(why)
        logger.warning("par: routing left unchanged - %s", why)


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
    _shim_remote_code()
    keep = os.environ.get("PAR_KEEP")
    if keep is None:
        return
    keep = int(keep)

    from vllm.logger import init_logger
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    logger = init_logger("par")
    if getattr(BaseRouter, "_par_patched", False):
        return
    inner = BaseRouter._select_experts

    def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *,
                        input_ids=None):
        topk_weights, topk_ids = inner(self, hidden_states, router_logits,
                                       topk_indices_dtype=topk_indices_dtype,
                                       input_ids=input_ids)
        why = _unsupported(self, router_logits, keep)
        if why is not None:
            _warn(logger, why)
            return topk_weights, topk_ids
        ids = rewrite(router_logits, topk_ids, keep, _ep_size())
        return gate_weights(router_logits, ids, self.renormalize).to(topk_weights.dtype), ids

    BaseRouter._select_experts = _select_experts
    BaseRouter._par_patched = True
    logger.info("par: placement-aware routing active, keep=%d of top_k, ep_size=%d",
                keep, _ep_size())
    # Touched so an import error surfaces at load time rather than in the first forward.
    assert issubclass(FusedTopKRouter, BaseRouter)


def _ep_size() -> int:
    from vllm.distributed.parallel_state import get_ep_group

    return get_ep_group().world_size


def _unsupported(router, router_logits, keep: int) -> str | None:
    """Why the rule does not apply to this router, or `None` if it does."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import FusedTopKRouter

    if not isinstance(router, FusedTopKRouter):
        return f"{type(router).__name__} does not rank by gate logits alone"
    if router.scoring_func != "softmax":
        return f"scoring_func={router.scoring_func!r}"
    if router.eplb_state is not None:
        return "EPLB replication breaks the expert-to-rank map"
    k, e = router.top_k, router.global_num_experts
    if keep >= k:
        return f"keep={keep} >= top_k={k}, nothing to re-place"
    if router_logits.shape[-1] != e:
        return f"router_logits width {router_logits.shape[-1]} != {e} experts"
    g = _ep_size()
    if g < 2:
        return f"ep_size={g}, no all-to-all to shrink"
    if e % g or e // g < k:
        return f"{e} experts over {g} ranks leaves no substitutes"
    return None
