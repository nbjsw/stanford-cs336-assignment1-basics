import torch

from einops import einsum, rearrange


class RotaryPositionalEmbedding(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None, dtype: torch.dtype = None):
        """Applies RoPE to the input tensor.

           Args:
               theta: value for the RoPE
               d_k: dimension of query and key vectors
               max_seq_len: Maximum sequence length that will be inputted
               device: Device to store the buffer on
               dtype: Data type
        """
        super().__init__()

        # θ_i,k = i /const_Θ ^ ((2k−2)/d)
        # 这个公式表明了每个词元的旋转角度 θ 是由两个因素共同决定的：
        # 词元的绝对位置 i (决定了旋转的幅度)。
        # 维度对的索引 k (决定了旋转的频率/速度)。k ∈ {1, . . . , d/2}

        # 1. 计算 m 向量 (频率/角度系数)
        # m = [0, 0, 2/d, 2/d, 4/d, 4/d, ..., (d-2)/d, (d-2)/d]
        # (2k-2)/d for k in {1, ..., d/2}，这里我们使用 PyTorch 索引从 0 开始
        # 2 is step
        m = torch.arange(0, d_k, 2).float() / d_k

        # 计算频率：freq = 1 / (theta ** m)
        # 形状: (d_k // 2,)
        inv_freq = theta ** (-m)

        # 2. 计算位置索引 (pos)
        # 形状: (max_seq_len,)
        pos = torch.arange(max_seq_len, dtype=torch.float)

        # 3. 计算角度张量 (t * freq)
        # torch.einsum('i, j -> ij', pos, inv_freq) 
        # 等价于 (max_seq_len, 1) * (1, d_k // 2)
        # 形状: (max_seq_len, d_k // 2)
        angles = einsum(pos, inv_freq, 'i, j -> i j')

        # 4. 扩展为 cos/sin 缓冲区 (将 d_k // 2 扩展为 d_k)
        # 我们需要 [cos(a1), cos(a1), cos(a2), cos(a2), ...]
        # 形状: (max_seq_len, d_k)
        cos = angles.repeat_interleave(2, dim=-1)
        sin = angles.repeat_interleave(2, dim=-1)

        factory_kwargs = {'device': device, 'dtype': dtype}

        # 5. 注册缓冲区 (不进行梯度学习)
        cos_data = torch.cos(cos)
        sin_data = torch.sin(sin)
        self.register_buffer('cos', cos_data.to(**factory_kwargs), persistent=False)
        self.register_buffer('sin', sin_data.to(**factory_kwargs), persistent=False)      


    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将输入向量 x 的后半部分移到前半部分，并取负号。
                         等价于对每一对分量 [x, y] 执行 [-y, x] 转换。
        """
        # x形状: (..., seq_len, d_k)

        # 1. 抽取偶数索引维度的值 (x1, x3, x5, ...)
        # 对应数学上的 'x' 部分
        # start:end:step
        x_even = x[..., ::2] 

        # 2. 抽取奇数索引维度的值 (x2, x4, x6, ...)
        # 对应数学上的 'y' 部分
        x_odd = x[..., 1::2]

        # 3. 构造 [-y, x] 结构
        # torch.stack 沿着新维度(-1)堆叠两个张量：[-x_odd, x_even]
        # 形状: (..., seq_len, d_k // 2, 2)

        # torch.stack([a, b], dim=0)
        # we get s = [[a ....]
        #             [b ....]]
        # but
        # torch.stack([a, b], dim=-1) is equivalent to s.T
        # s' = [[a1, b1],
        #       [a2, b2],
        #       [a3, b3],
        #       ....    ]
        # the following stack help us get [[-y1, x1], [-y2, x2] ..]
        # shape: (..., seq_len, d_k // 2, 2)
        x_rotated = torch.stack([-x_odd, x_even], dim=-1)

        # 4. 展平回原始特征维度 d_k
        # 将倒数第二个维度及其之后的所有维度展平。
        # 形状: (..., seq_len, d_k)
        return x_rotated.flatten(-2)  


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
                  应用 RoPE 旋转。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)。
            token_positions (torch.Tensor): 词元位置索引张量，形状为 (..., seq_len)。

        Returns:
            torch.Tensor: 应用旋转后的张量，形状与 x 相同。
        """
        # --- 步骤 1: 提取 cos/sin 缓冲区 (查表) ---
        # token_positions 形状: (..., seq_len)
        # self.cos/self.sin 形状: (max_seq_len, d_k)

        # 目的: 根据 token_positions 沿着序列维度切片，并恢复原始形状 (..., seq_len, d_k)。
        # token_positions 必须是 LongTensor

        # 展平位置索引，并从缓冲区中选择对应的行
        # 假设输入张量 x 有一个批次维度（B）、一个序列维度（S）和一个特征维度（D=d_k )
        # token_positions.flatten 所有词元的位置索引串联成一个长长的一维列表 because index_select accepts only 1-D
        # reshape *token_positions.shape： 取出原始的批次维度和序列维度 (B, S), -1： 告诉 PyTorch 自适应地填补最后一个维度（即特征维度 D）
        cos_sliced = self.cos.index_select(0, token_positions.flatten()).reshape(*token_positions.shape, -1)
        sin_sliced = self.sin.index_select(0, token_positions.flatten()).reshape(*token_positions.shape, -1)

        # --- 步骤 2: 计算旋转辅助向量 x_rot ---

        # Rotation [2 * 2] 矩阵 是经典2维平面旋转公式，可以通过欧拉公式推导的
        # x' = x * cos - y * sin
        # y' = y * cos + x * sin
        # => [x,y] = [x, y] * cos + [-y, x] * sin
        # this so-called x_rot is [-y, x]

        # x_rot 实现了 [x, y] -> [-y, x] 的转换
        # x_rot 形状: (..., seq_len, d_k)
        x_rot = self._rotate_half(x)

        # 原始形状: (B, L, D_h)
        # 目标形状: (B, 1, L, D_h)
        if cos_sliced.ndim == 3:
            # 使用 einops 优雅地插入一个维度
            cos_sliced = rearrange(cos_sliced, 'b l d -> b 1 l d')

            # 对 sin_sliced 做同样处理 (假设它也用于 RoPE 乘法)
            sin_sliced = rearrange(sin_sliced, 'b l d -> b 1 l d')

        # --- 步骤 3: 应用 RoPE 旋转公式 (元素乘加) ---
        # 公式等价于: x' = x * cos(theta) + R(x) * sin(theta)
        # R(x) 即是 x_rot

        # x * cos(theta)
        first_term = x * cos_sliced

        # R(x) * sin(theta)
        second_term = x_rot * sin_sliced

        x_out = first_term + second_term

        return x_out

