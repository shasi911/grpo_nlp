import torch


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute the naive policy gradient per-token loss.

    Maximising E[R * log π] is equivalent to minimising -R * log π.

    Args:
        raw_rewards_or_advantages: (batch_size, 1) — scalar reward or advantage
            for each sequence in the batch.
        policy_log_probs: (batch_size, sequence_length) — per-token log-probs
            from the current policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length):
            per-token policy gradient loss (before masking / aggregation).
    """
    return -raw_rewards_or_advantages * policy_log_probs
