import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict]:

    batch_size = policy_log_probs.shape[0]
    total_norm = batch_size * gradient_accumulation_steps * normalize_constant

    loss = -(policy_log_probs * response_mask).sum() / total_norm
    loss.backward()

    return loss.detach(), {}
