import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)
