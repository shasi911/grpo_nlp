import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip (PPO-style clipped) per-token loss.

    loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)

    where ratio = exp(policy_log_probs - old_log_probs).

    Args:
        advantages: (batch_size, 1) — advantage for each sequence.
        policy_log_probs: (batch_size, sequence_length) — current policy log-probs.
        old_log_probs: (batch_size, sequence_length) — old policy log-probs.
        cliprange: float ε — clip ratio to [1-ε, 1+ε].

    Returns:
        loss: (batch_size, sequence_length) per-token loss.
        metadata: dict with "clip_fraction" (fraction of tokens where clip activated).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    loss = -torch.min(surr1, surr2)

    clip_fraction = (ratio != clipped_ratio).float().mean()
    metadata = {"clip_fraction": clip_fraction}

    return loss, metadata
