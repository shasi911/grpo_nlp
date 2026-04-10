import torch
from alignment.compute_policy_gradient_loss import compute_policy_gradient_loss
from alignment.masked_mean import masked_mean


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute GRPO policy gradient loss, backprop, and return loss + metadata.

    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps

    Args:
        policy_log_probs: (batch_size, sequence_length)
        response_mask: (batch_size, sequence_length)
        gradient_accumulation_steps: int
        loss_type: "no_baseline" | "reinforce_with_baseline" | "grpo_clip"
        raw_rewards: (batch_size, 1) — required for "no_baseline"
        advantages: (batch_size, 1) — required for "reinforce_with_baseline" / "grpo_clip"
        old_log_probs: (batch_size, sequence_length) — required for "grpo_clip"
        cliprange: float — required for "grpo_clip"

    Returns:
        loss: scalar tensor (detached)
        metadata: dict from the underlying loss function
    """
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )

    loss = masked_mean(per_token_loss, response_mask) / gradient_accumulation_steps
    loss.backward()

    return loss.detach(), metadata
