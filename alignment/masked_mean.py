import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """Compute the mean of tensor considering only elements where mask == 1.

    Positions where all mask values are False along the reduction dim
    produce NaN (0 / 0), consistent with standard behaviour.

    Args:
        tensor: torch.Tensor to average.
        mask: torch.Tensor (same shape), 1 for positions to include.
        dim: dimension to reduce over. None reduces over all elements.

    Returns:
        Mean tensor with the specified dimension removed (or scalar if dim=None).
    """
    mask = mask.float()
    masked = tensor * mask
    if dim is None:
        return masked.sum() / mask.sum()
    return masked.sum(dim=dim) / mask.sum(dim=dim)
