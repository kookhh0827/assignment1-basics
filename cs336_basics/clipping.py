from collections.abc import Iterable
import torch


# gradient clipping
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
  # Compute global L2 norm over all parameter gradients
  total_sq = None
  for p in parameters:
    if p.grad is None:
      continue
    g = p.grad
    sq = torch.sum(g.detach() * g.detach())
    if total_sq is None:
      total_sq = sq
    else:
      total_sq = total_sq + sq

  if total_sq is None:
    return

  total_norm = torch.sqrt(total_sq) 
  if total_norm <= max_l2_norm:
    return

  scale = max_l2_norm / (total_norm + eps)
  for p in parameters:
    if p.grad is None:
      continue
    p.grad.mul_(scale)