import torch


class Linear(torch.nn.Module):
    """
    Equivalent to torch.nn.Linear(in_features, out_features, bias=False)
    """
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        num_in_features_tensor = torch.tensor(in_features, device=device, dtype=torch.float32)
        std = 1.0 / torch.sqrt(num_in_features_tensor)
        torch.nn.init.normal_(self.weight, mean=0.0, std=std)
        a_tensor = torch.tensor(-2.0 * std.item(), device=device, dtype=dtype)
        b_tensor = torch.tensor(2.0 * std.item(), device=device, dtype=dtype)
        self.weight.data.clamp_(min=a_tensor, max=b_tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

