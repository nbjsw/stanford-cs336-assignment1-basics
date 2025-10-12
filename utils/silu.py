import torch


class SiLU(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigmoid_x = torch.sigmoid(x)
        return x * sigmoid_x
