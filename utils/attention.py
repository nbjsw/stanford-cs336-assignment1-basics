import torch

from . import linear, rope, softmax
from einops import rearrange


def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
         实现缩放点积注意力 (Scaled Dot-Product Attention)

    Args:
        query: 查询矩阵 Q。形状: (batch_size, ..., seq_len_q, d_k)
        key: 键矩阵 K。形状: (batch_size, ..., seq_len_k, d_k)
        value: 值矩阵 V。形状: (batch_size, ..., seq_len_k, d_v)
        mask: 可选的布尔掩码 M。形状: (seq_len_q, seq_len_k) 或可广播到注意力分数形状。
              True 表示允许信息流动 (attend)，False 表示掩盖。

    Returns:
         输出张量。形状: (batch_size, ..., seq_len_q, d_v)
    """
    # 一个完整的 Transformer 模型（包含编码器和解码器）使用了三种不同的注意力类型
    # 自注意力 (Self-Attention) seq_len_q = seq_len_k
    # 掩码自注意力 seq_len_q = seq_len_k
    # 编码器-解码器注意力：交叉注意力 (Cross-Attention)
    # $\text{seq\_len\_q}$（解码器长度）和 $\text{seq\_len\_k}$（编码器长度）通常是不同的。
    d_k = key.size(-1)
    scaled_scores = (query @ key.transpose(-2, -1)) / (d_k ** 0.5)
    
    if mask is not None:
        scaled_scores.masked_fill_(~mask, -1e9)

    attention_weights = softmax.softmax(scaled_scores, dim=-1)
    output = attention_weights @ value
    return output


class CausalMultiheadSelfAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 max_seq_len: int = 0, # 新增 RoPE 参数，默认 0 表示禁用 RoPE
                 theta: float = 10000.0,
                 device=None,
                 dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.q_proj = linear.Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = linear.Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = linear.Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = linear.Linear(d_model, d_model, **factory_kwargs)

        self.score = self.d_head ** (-0.5)

        if max_seq_len > 0:
            self.rope = rope.RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len
            )
        else:
            self.rope = None # 不进行旋转

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, ' ... s (h x_h) -> ... h s x_h', h=self.num_heads, x_h=self.d_head)
        return x
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, ' ... h s x_h -> ... s (h x_h)')
        return x

    def forward(self,  x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        _, seq_len, d_model = x.shape
        
        # 1. 线性投影 (Q = x W^Q)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 2. 分割多头: (B, H, L, D)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # 3. 应用 RoPE (检查 self.rope 是否存在)
        if self.rope is not None:
            if token_positions is None:
                # 自动生成位置索引
                token_positions = torch.arange(seq_len, device=x.device).long()

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 4. 因果掩码 (Boolean), scaled_dot_product_attention uses (~M) so we use tril not triu
        # 形状: (L, L)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        # 5. 调用 scaled_dot_product_attention 来完成缩放、点积、掩码、Softmax和V乘法
        # output 形状: (B, H, L, D_v)
        output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # 6. 合并多头
        output = self._combine_heads(output)
        
        # 7. 最终线性投影 (WO)
        output = self.output_proj(output)
        
        return output

