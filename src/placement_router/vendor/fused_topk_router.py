# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored verbatim from vLLM v0.31.0:
#   vllm/model_executor/layers/fused_moe/router/fused_topk_router.py
#
# The only change is the imports and `FusedTopKRouter._compute_routing`, marked `placement`
# below: the lines that hand the routing to `placement_router.kernel.select` instead of
# `fused_topk`, which picks the top `keep` by logit and replaces the rest from ranks the head
# already visits (or, with the load penalty and `any_rank`, from any rank).
# Everything else is upstream's, and the file is kept as a copy rather than a patch so
# `git diff` against the upstream path is the whole story.
#
# It *replaces* the top-k rather than running after it, which is the whole performance story:
# one Triton launch per MoE layer where default vLLM also spends one, so `keep == top_k` is a
# control setting that costs nothing measurable. Following the fused top-k instead would cost a
# second launch per MoE layer per forward - ~3% of decode throughput at 48 layers, which
# cudagraphs cannot amortise away because DeepEP with data parallelism disables them.
#
# It is a copy because the substitution has to happen *here*: the router is the last place
# that still has the full `[T, E]` logits, which is what choosing a substitute needs, and the
# dispatch is the first place that pays for fan-out.
from collections.abc import Callable

import torch

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

from placement_router.kernel import sample_tail, select                                  # placement
from placement_router.plugin import (any_rank, coact_loaded, coact_perm, decode_rows,     # placement
                                     ep_size, keep_for, load_penalty, mass, mass_norm,    # placement
                                     sampled_tail)                                        # placement


def _get_padding_mask(num_tokens: int) -> torch.Tensor | None:
    if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
        is_padding = get_forward_context().is_padding
        return is_padding[:num_tokens] if is_padding is not None else None
    return None


def vllm_topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def vllm_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> tuple[torch.Tensor, ...]:
    ops.topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        is_padding=_get_padding_mask(topk_indices.shape[0]),
    )

    return topk_weights, topk_indices


def dispatch_topk_softmax_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_softmax
    return vllm_topk_softmax


def dispatch_topk_sigmoid_func(
    use_rocm_aiter: bool = False,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    if use_rocm_aiter:
        return rocm_aiter_ops.topk_sigmoid
    return vllm_topk_sigmoid


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, _ = hidden_states.size()

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device,
    )
    token_expert_indices = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    if scoring_func == "softmax":
        topk_func = dispatch_topk_softmax_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    elif scoring_func == "sigmoid":
        topk_func = dispatch_topk_sigmoid_func(
            use_rocm_aiter=rocm_aiter_ops.is_fused_moe_enabled()
        )
        topk_weights, topk_ids = topk_func(
            topk_weights, topk_ids, token_expert_indices, gating_output, renormalize
        )

        return topk_weights, topk_ids, token_expert_indices
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")


class FusedTopKRouter(BaseRouter):
    """Default router using standard fused top-k routing."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
        eplb_state: EplbLayerState | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return get_routing_method_type(
            scoring_func=self.scoring_func,
            top_k=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=None,
            has_e_score_bias=False,
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using standard fused top-k."""
        # placement: the placement-aware kernel *is* the routing here, not a pass over its output,
        # so this setting spends the one launch the fused top-k below would have spent. `keep_for`
        # is `None` - one attribute read - unless the rule's premise holds for this router.
        perm = coact_perm(self, router_logits.device)                           # placement
        if perm is not None and coact_loaded():
            # placement: the experts were loaded in placement order, so route in those ids.
            router_logits, perm = router_logits[:, perm], None
        keep = keep_for(self, router_logits)
        dec = None if keep is None else decode_rows(router_logits.shape[0])     # placement
        if keep is not None and (st := sampled_tail()):                         # placement
            return sample_tail(router_logits, self.top_k, keep, *st, indices_type)
        if keep is not None and dec is not True:
            w, ids = select(router_logits if perm is None else router_logits[:, perm],
                            self.top_k, keep, ep_size(),
                            _get_padding_mask(router_logits.shape[0]),
                            indices_dtype=indices_type, load_penalty=load_penalty(),
                            mass=mass(), mass_norm=mass_norm(), any_rank=any_rank())  # placement
            if perm is not None:
                j = ids.long()                                     # -1 (or unsigned) on padding
                ok = (j >= 0) & (j < len(perm))
                ids = torch.where(ok, perm[j.clamp(0, len(perm) - 1)], j).to(ids.dtype)
            if dec is None:
                return w, ids
            # placement: P/D emulation, decode rows keep default vLLM's routing.
            dw, dids, _ = fused_topk(hidden_states, router_logits, self.top_k, self.renormalize,
                                     indices_type, self.scoring_func)
            return torch.where(dec[:, None], dw, w), torch.where(dec[:, None], dids, ids)

        topk_weights, topk_ids, token_expert_indices = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            scoring_func=self.scoring_func,
        )

        return topk_weights, topk_ids
