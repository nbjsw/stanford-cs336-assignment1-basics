import torch
import typing
import os
import numpy as np
import numpy.typing as npt


def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # offset selection
    # (X + Y) -> context_length + 1
    # assume dataset: 10
    # context_length: 4
    # max_start_index: 5
    max_start_index = len(dataset) - context_length - 1
    start_indices = np.random.randint(low=0, high=max_start_index + 1, size=batch_size)

    # advance indexing
    offsets = np.arange(context_length)
    # key step: start_indices (batch_size,) + offsets (1, context_length) => (batch_size, context_length)
    # np.newaxis is similar to tensor.unsqueeze(1)
    # start_indices[:, np.newaxis] shape is (batch_size, 1)
    # offsets (1, context_length)
    # add -> broadcasting => (batch_size, context_length)
    indices_x = start_indices[:, np.newaxis] + offsets
    x = dataset[indices_x]
    indices_y = indices_x + 1
    y = dataset[indices_y]

    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()
    x = x.to(device)
    y = y.to(device)

    return x, y


# 定义用于保存和加载检查点所需的状态字典结构
CHECKPOINT_STATE = typing.TypedDict(
    'CHECKPOINT_STATE',
    {
        'iteration': int,
        'model_state_dict': typing.Dict[str, torch.Tensor],
        'optimizer_state_dict': typing.Dict[str, typing.Any],
    }
)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> None:
    """
    将模型、优化器和当前的迭代次数保存到检查点。

    Args:
        model: 要保存的 nn.Module。
        optimizer: 要保存的 torch.optim.Optimizer。
        iteration: 当前的训练迭代次数 (int)。
        out: 文件路径或类文件对象，用于保存检查点。
    """
    # 1. 构造检查点字典
    checkpoint: CHECKPOINT_STATE = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # 2. 使用 torch.save 将字典转储到文件/流
    torch.save(checkpoint, out)
    # print(f"Checkpoint saved successfully at iteration {iteration} to {out}.") # 调试信息


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点加载状态，恢复模型和优化器，并返回保存的迭代次数。

    Args:
        src: 文件路径或类文件对象，用于加载检查点。
        model: 要恢复状态的 nn.Module。
        optimizer: 要恢复状态的 torch.optim.Optimizer。
        
    Returns:
        保存到检查点中的迭代次数 (int)。
    """
    # 1. 使用 torch.load 从文件/流中加载检查点字典
    # 注意：需要添加 map_location='cpu' 以防加载到错误的设备上
    checkpoint = torch.load(src, map_location='cpu')

    # 2. 使用 load_state_dict 恢复模型状态
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. 使用 load_state_dict 恢复优化器状态
    # 这对于 AdamW 等有状态的优化器（保存了动量等）至关重要
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 4. 返回保存的迭代次数，以便恢复学习率调度器或训练循环
    iteration = checkpoint['iteration']
    # print(f"Checkpoint loaded successfully. Resuming from iteration {iteration}.") # 调试信息
    return iteration

