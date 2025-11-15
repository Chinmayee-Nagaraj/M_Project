import torch
from torch.optim import Optimizer


class DynamicLRScheduler:
    """
    Modified Vaswani et al. Learning Rate Scheduler.

    lr =
        k1 * d_model^(-0.5) * n * warmup^(-1.5),     if n <= warmup
        k2 * 0.98^(epoch / 2),                       if n > warmup

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        d_model (int): Transformer input feature size.
        warmup_steps (int): Number of warmup steps.
        k1 (float): Scale factor during warmup.
        k2 (float): Base learning rate after warmup.
    """

    def __init__(self,
                 optimizer: Optimizer,
                 d_model: int = 64,
                 warmup_steps: int = 4000,
                 k1: float = 0.2,
                 k2: float = 4e-4):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup_steps
        self.k1 = k1
        self.k2 = k2
        self.step_num = 0
        self.epoch = 0

    def step(self):
        """Update learning rate each training step."""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def epoch_step(self):
        """Call once per epoch."""
        self.epoch += 1

    def get_lr(self):
        """Compute current learning rate."""
        n = self.step_num
        if n <= self.warmup:
            lr = self.k1 * (self.d_model ** -0.5) * n * (self.warmup ** -1.5)
        else:
            lr = self.k2 * (0.98 ** (self.epoch / 2.0))
        return lr
