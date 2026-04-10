import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """Compute per-rollout rewards and normalize within each group (GRPO).

    Args:
        reward_fn: Callable[[str, str], dict] — returns dict with "reward" key.
        rollout_responses: list[str] of length rollout_batch_size.
        repeated_ground_truths: list[str] of length rollout_batch_size.
        group_size: int, number of rollouts per prompt.
        advantage_eps: float, added to std to avoid division by zero.
        normalize_by_std: bool, whether to divide by within-group std.

    Returns:
        normalized_rewards: torch.Tensor of shape (rollout_batch_size,)
        raw_rewards: torch.Tensor of shape (rollout_batch_size,)
        metadata: dict[str, float]
    """
    # Compute scalar reward for each rollout
    raw = [
        reward_fn(resp, gt)["reward"]
        for resp, gt in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(raw, dtype=torch.float32)

    # Reshape into (n_groups, group_size) for group-wise statistics
    n_groups = len(raw_rewards) // group_size
    grouped = raw_rewards.view(n_groups, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)          # (n_groups, 1)
    normalized = grouped - group_mean

    if normalize_by_std:
        # unbiased=True uses N-1 denominator, matching the snapshot
        group_std = grouped.std(dim=1, keepdim=True, unbiased=True)
        normalized = normalized / (group_std + advantage_eps)

    normalized = normalized.view(-1)

    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "frac_nonzero": (raw_rewards != 0).float().mean().item(),
    }

    return normalized, raw_rewards, metadata
