import torch
from alignment.compute_naive_policy_gradient_loss import compute_naive_policy_gradient_loss
from alignment.compute_grpo_clip_loss import compute_grpo_clip_loss


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to the appropriate policy gradient loss function.

    Args:
        policy_log_probs: (batch_size, sequence_length)
        loss_type: one of "no_baseline", "reinforce_with_baseline", "grpo_clip"
        raw_rewards: (batch_size, 1) — used for "no_baseline"
        advantages: (batch_size, 1) — used for "reinforce_with_baseline" and "grpo_clip"
        old_log_probs: (batch_size, sequence_length) — used for "grpo_clip"
        cliprange: float — used for "grpo_clip"

    Returns:
        loss: (batch_size, sequence_length) per-token loss
        metadata: dict with any loss-specific metadata
    """
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}

    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}

    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
