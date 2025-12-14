import torch


class Linear(torch.nn.Module):
    """
    Equivalent to torch.nn.Linear(in_features, out_features, bias=False)
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        std = (2.0 / (in_features + out_features)) ** 0.5

        torch.nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=std, 
            a=-3.0 * std, 
            b=3.0 * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
