import torch

from . import linear, silu


class SwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.w1 = linear.Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = linear.Linear(d_ff, d_model, **factory_kwargs)
        self.w3 = linear.Linear(d_model, d_ff, **factory_kwargs)

        self.act_fn = silu.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The full feed-forward block (expand → gate → project back)
        # SwiGLU(x)=(SiLU(xW1​)⊙(xW3))W2

        # up projection and silu activation, for gating, shape: [... d_ff]
        gating = self.act_fn(self.w1(x))
        # up projection, for main information, shape: [... d_ff]
        value = self.w3(x)
        # down projection
        return self.w2(gating * value)

