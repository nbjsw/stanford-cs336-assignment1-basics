import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Stability
    # dim指向谁，谁就被压缩成 1
    # then it's easy to figure out what value to keep for that "1"
    max_vals, _ = torch.max(x, dim=dim, keepdim=True)
    z_shifted = x - max_vals

    # Numerator
    numerator = torch.exp(z_shifted)

    # Denominator
    denominator = torch.sum(numerator, dim=dim, keepdim=True)

    return numerator / denominator
