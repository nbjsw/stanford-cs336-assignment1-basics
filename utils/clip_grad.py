import torch

from typing import Iterable


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    if max_l2_norm <= 0:
        return
    # When a tensor is marked to require gradient computation (i.e., its attribute requires_grad=True),
    # the system will automatically create a .grad attribute for it after the backward pass
    # (loss.backward()) runs.
    total_norm = torch.tensor(0.0, device=parameters[0].grad.device)
    for p in parameters:
        # if p.grad:
        # THIS WILL CAUSE THE ERROR if p.grad has multiple elements
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).pow(2)
    total_norm = torch.sqrt(total_norm)
    norm_coeff = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(norm_coeff)
