import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages

    loss = -torch.min(surr1, surr2)

    clip_fraction = (ratio != clipped_ratio).float().mean()
    metadata = {"clip_fraction": clip_fraction}

    return loss, metadata
