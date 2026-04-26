import torch
import torch.nn as nn

from .base import BaseSubKernel

class RBFSubKernel(BaseSubKernel):
    """
    RBF (Gaussian) kernel: K[i,j] = exp(-gamma * (x[i] - x[j])^2)
    """
    def __init__(self, initial_gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(initial_gamma))
    
    def forward(self, x_col: torch.Tensor,
                y_col: torch.Tensor = None) -> torch.Tensor:
        if y_col is None:
            y_col = x_col
        # x_col shape (n,), y_col shape (m,) → diff shape (n, m)
        diff = x_col.unsqueeze(1) - y_col.unsqueeze(0)
        return torch.exp(-self.gamma * diff ** 2)
