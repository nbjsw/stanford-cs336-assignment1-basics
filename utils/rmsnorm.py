import torch


class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        """"
        Root Mean Square Layer Normalization

        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
            device: Device to store the parameters on
            dtype: Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = torch.nn.Parameter(torch.ones((d_model,), **factory_kwargs))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape"""
        # (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        in_dtype = x.dtype
        # prevent overflow when you square the input
        x = x.to(torch.float32)
        mean_square = torch.sum(torch.square(x), dim=-1, keepdim=True) / self.d_model
        rms = torch.sqrt(mean_square + self.eps)
        x = x / rms * self.weight
        return x.to(in_dtype)

