import torch

from . import softmax


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

