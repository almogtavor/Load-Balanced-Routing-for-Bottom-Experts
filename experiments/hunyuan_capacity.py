"""Hunyuan's capacity-dropping gate, replaced in place. No-op for every other family.

Hunyuan's vendored `topkgating` sets `expert_capacity = max(topk, topk * n_tokens //
num_experts)` - *exactly the mean load per expert*, with no capacity factor (the untruncated
line is present directly above it, commented out). Routing is never uniform, so every expert
above its mean share silently discards the overflow: the token gets a zero combine weight for
that expert and its contribution is simply missing from the sum. `HunYuanTopKGate` does read
`config.moe_drop_tokens` (false in this checkpoint) but never passes it here, so the flag has
no effect.

Unpatched this leaves the model near NLL 7.0 on the multilingual stream, perplexity around
1100, where the Qwen models sit near 1.6. That matters doubly here: placement-aware re-routing
*concentrates* load onto fewer experts by design, so an arm would be penalised for triggering
more capacity drops rather than for its routing - the intervention and the artifact point the
same way, which is the one situation where a confound cannot be argued away.

The fix is one line: capacity becomes the largest load any expert actually has, the smallest
value that drops nothing. Everything else is the vendored source unchanged. The dense
`dispatch_mask` is `[tokens, experts, capacity]`, so memory grows with the true maximum load -
which is why this model is evaluated at batch 1.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

__all__ = ["no_drop_moe", "topkgating_no_drop"]


def topkgating_no_drop(logits: torch.Tensor, topk: int):
    """`topkgating` with the capacity limit raised to the real maximum load."""
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts ** 2 * torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert)

    gates_s = torch.clamp(
        torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1),
        min=torch.finfo(gates.dtype).eps)
    router_probs = gates / gates_s
    expert_index = torch.transpose(expert_index, 0, 1).reshape(-1)
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # The one changed line. `exp_counts` has to be computed before it, which is why this sits
    # here rather than where the original put it.
    expert_capacity = max(topk, int(exp_counts.max()))

    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    token_priority = token_priority.reshape((topk, -1, num_experts))
    token_priority = torch.transpose(token_priority, 0, 1)
    token_priority = torch.max(token_priority, dim=1)[0]

    valid_mask = torch.logical_and(token_priority >= 0, token_priority < expert_capacity)
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts

def no_drop_moe(model) -> bool:
    """Patch the gate in the module `trust_remote_code` loaded. Returns whether it applied.

    The function has to be replaced in that module, not rebound on the model, because the
    already-constructed gate modules call it by module-global lookup. Returns False without
    complaint for a model that has no such gate: every other family here routes without a
    capacity limit, so there is nothing to undo.
    """
    for m in model.modules():
        mod = sys.modules.get(type(m).__module__)
        if mod is not None and hasattr(mod, "topkgating"):
            if getattr(mod.topkgating, "__name__", "") == "topkgating_no_drop":
                return True
            mod.topkgating = topkgating_no_drop
            return True
    return False
