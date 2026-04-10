import torch


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict:

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (batch, seq, vocab)

    log_probs_all = torch.log_softmax(logits, dim=-1)  # (batch, seq, vocab)

    # Gather log-prob of the actual next token at each position
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        entropy = -(torch.exp(log_probs_all) * log_probs_all).sum(dim=-1)  # (batch, seq)
        result["token_entropy"] = entropy

    return result
