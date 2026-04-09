import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension).

    Args:
        logits: torch.Tensor of shape (..., vocab_size)

    Returns:
        torch.Tensor of shape (...): entropy over the last dimension.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)
