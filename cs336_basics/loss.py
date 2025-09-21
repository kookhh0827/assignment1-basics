import torch


# cross_entropy
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # inputs: (..., vocab_size)
    # targets: (...,) with class indices
    # return: scalar mean loss over all batch-like dims
    assert inputs.dim() >= 1, "inputs must have at least one dimension"
    assert inputs.shape[:-1] == targets.shape, "targets must match inputs shape except last vocab dimension"

    # For numerical stability: subtract max over vocab dimension
    shifted = inputs - torch.amax(inputs, dim=-1, keepdim=True)

    # log(sum(exp(shifted))) - shifted[range, target]
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    gathered = torch.gather(shifted, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    nll = logsumexp - gathered

    return nll.mean()