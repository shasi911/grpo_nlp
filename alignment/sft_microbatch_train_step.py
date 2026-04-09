import torch


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute SFT loss and backprop gradients for a microbatch.

    Loss = -sum(log_probs * mask) / (batch_size * gradient_accumulation_steps * normalize_constant)

    Args:
        policy_log_probs: (batch_size, seq_length) log-probs from the policy.
        response_mask: (batch_size, seq_length) 1 for response tokens, 0 otherwise.
        gradient_accumulation_steps: number of microbatches per optimizer step.
        normalize_constant: optional per-token normalizer (e.g. sequence length in Dr.GRPO).

    Returns:
        (loss, metadata) where loss is the scalar loss before backward.
    """
    batch_size = policy_log_probs.shape[0]
    total_norm = batch_size * gradient_accumulation_steps * normalize_constant

    loss = -(policy_log_probs * response_mask).sum() / total_norm
    loss.backward()

    return loss.detach(), {}
