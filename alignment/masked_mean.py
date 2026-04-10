import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:

    mask = mask.float()
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    return masked.sum(dim=dim) / mask.sum(dim=dim)
