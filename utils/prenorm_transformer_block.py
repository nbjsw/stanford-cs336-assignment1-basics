import torch

from . import attention, linear, swiglu, rmsnorm


class PreNormTransformerBlock(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.norm1 = rmsnorm.RMSNorm(d_model, **factory_kwargs)
        self.attn = attention.CausalMultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, **factory_kwargs)

        self.norm2 = rmsnorm.RMSNorm(d_model, **factory_kwargs)
        self.ffn = swiglu.SwiGLU(d_model, d_ff, **factory_kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attentioned_x = x + self.attn(self.norm1(x))
        out = attentioned_x + self.ffn(self.norm2(attentioned_x))
        return out

