import torch


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor to sum and normalize.
        mask: torch.Tensor; elements with mask=0 don't contribute to the sum.
        dim: dimension to sum along. If None, sum over all elements.
        normalize_constant: constant to divide the sum by.

    Returns:
        torch.Tensor with the normalized sum.
    """
    masked = tensor * mask
    if dim is None:
        return masked.sum() / normalize_constant
    return masked.sum(dim=dim) / normalize_constant
