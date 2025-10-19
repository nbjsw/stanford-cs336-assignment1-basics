import torch
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

