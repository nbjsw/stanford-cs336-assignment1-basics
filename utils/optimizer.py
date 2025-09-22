from __future__ import annotations

import torch
from torch.optim import Optimizer
from jaxtyping import Float, jaxtyped
from torch import Tensor


# math: https://chatgpt.com/s/t_68d094f296ac8191a0c5cf546db0c73f
# code: https://chatgpt.com/s/t_68d0950fbb108191b7ef7545d14442a8
class MyAdamW(Optimizer):

    @jaxtyped
    def __init__(self,
                 params: Float[Tensor, " ..."],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        """
        Initialize the MyAdamW optimizer.

        Args:
            params (iterable of Tensor): Iterable of parameters to optimize 
                or list of dicts specifying parameter groups.
            lr (float, optional): Learning rate. Default: 1e-3.
            betas (Tuple[float, float], optional): Coefficients used for computing 
                running averages of gradient and its square. Default: (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Default: 1e-8.
            weight_decay (float, optional): Decoupled weight decay (L2 penalty). Default: 0.01.

        Notes:
            This optimizer implements the AdamW algorithm (Adam with decoupled weight decay).
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(MyAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    @jaxtyped
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        # e.g., nn.Linear -> 2 groups: weight and bias
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad: Float[torch.Tensor, " ..."] = p.grad.data

                # p is torch.nn.Parameter and is mutable
                # However, Parameter has an unique id as the hash key
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # momentum / moving average of gradients
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # uncentered variance / moving average of squared gradients 
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update
                # m_t​=β1 * ​m_(t−1) ​+ (1−β1​) * g_t​
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                # v_t​=β2 * ​v_(t−1​) + (1−β2​)*(g_t)^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                # Bias Correction
                # m_t​_bc =​m_t / (1 - β1^t)​​
                # v_t_bc ​= v_t / (1 − β2^t)
                # Δθ​ = lr * m_t_bc / (sqrt(v_t_bc) + eps)
                # for computation optimization
                # Δθ = lr * m_t / (1 - β1^t)​​ / (sqrt(v_t / (1 − β2^t)) + eps)
                # Δθ = (lr / bias_correction_1 ) * m_t / sqrt(exp_avg_sq / bia_correction_2) + eps)
                bias_correction_1 = 1 - beta1 ** state['step']
                bias_correction_2 = 1 - beta2 ** state['step']
                step_size = lr / bias_correction_1
                # DO NOT use in-place sqrt on exp_avg_sq, the state need to be kept for next run.
                # so we need a new tensor to store sqrt, but add eps could be in place.
                denom = (exp_avg_sq.sqrt() / bias_correction_2 ** 0.5).add_(eps)

                # torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None) =>
                # out = input + value * tensor1 / tensor2​
                # p.data - Δθ = p.data - step_size * m_t / denom 
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # decoupled weight decay
                p.data.add_(p.data, alpha=-lr*weight_decay)

        return loss

