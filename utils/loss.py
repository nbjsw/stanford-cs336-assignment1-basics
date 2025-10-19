import torch


class CrossEntropyLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # cross entropy = -log(logits)
        # log(softmax) -> overflow
        # log sum exp -> log(e^zi / sum(e^z)) = zi - log(sum(e^z))

        # shape: shape: (batch_size, vocab_size)
        # use library:
        # log_sum_exp = torch.logsumexp(inputs, dim=-1, keepdim=True)
        # not use library:
        max_vals, _ = torch.max(inputs, dim=-1, keepdim=True)
        z_shifted = inputs - max_vals
        z_shifted_exp = torch.exp(z_shifted)
        log_sum_exp = torch.log(torch.sum(z_shifted_exp, dim=-1, keepdim=True)) + max_vals    
        log_probs = inputs - log_sum_exp

        # targets_expanded.shape: (batch_size, 1) from (batch_size,)
        targets_expanded = targets.unsqueeze(1)
    
        # l_gathered.shape: (batch_size, 1)
        # the idea is think target is one-hot [0,0,0,1,...,0, 0, 0]
        # cross entrophy loss = -sum(y_i * log(logit_i)) => only the valid one kept
        l_gathered = torch.gather(log_probs, dim=-1, index=targets_expanded)
        l_gathered_squeezed = l_gathered.squeeze(1)
        # cross entrophy loss is negative of log(logits)
        return torch.mean(-l_gathered_squeezed)

