import torch


class Linear(torch.nn.Module):
    """
    Equivalent to torch.nn.Linear(in_features, out_features, bias=False)
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}

        # 1. 构造并存储权重矩阵 W
        # 权重 W 的形状应为 (out_features, in_features)
        # 这样在前向传播时，W @ x.T 的结果维度是正确的。
        # PyTorch 的 nn.Linear 默认以 (out_features, in_features) 存储权重。
        # 线性变换为 y = x @ W.T (如果 W 形状是 (in, out)) 或 y = W @ x (如果 W 形状是 (out, in) 且 x 是列向量)。
        # 在 PyTorch 中，对于输入 x (..., in_features) 和 W (out_features, in_features)，
        # 运算是 x @ W.T，结果是 (..., out_features)。
        # 因此，我们需要 W 的形状是 (out_features, in_features)。
        self.W = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        # 2. 初始化权重
        # 使用 torch.nn.init.trunc_normal_ 进行初始化
        # 截断正态分布初始化
        # 推荐使用 gain=1.0（默认）
        # 计算标准差 (std) 的常见方法是基于输入维度，如 Kaiming 或 Xavier
        # 这里为了符合 PyTorch nn.Linear 的默认初始化方式（但它用的是 Kaiming/Uniform），
        # 我们需要一个合理的 std。对于 trunc_normal_，可以设置一个较小的 std。
        # 遵循 PyTorch nn.Linear 的 Kaiming Uniform 逻辑，std = 1 / sqrt(in_features) 是一个合理的范围：
        std = 1.0 / torch.sqrt(torch.tensor(in_features, dtype=torch.float32))

        # 使用截断正态分布初始化
        torch.nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-2 * std, b=2 * std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里的操作是输入张量 x 和权重 W 的转置 W.T 进行矩阵乘法。
        # x: (..., in_features)
        # W: (out_features, in_features)
        # W.T: (in_features, out_features)
        # 结果: (..., out_features)
        # 这就是 PyTorch nn.Linear 在不使用偏置项时的计算方式：y = x @ W.T
        return x @ self.W.T

