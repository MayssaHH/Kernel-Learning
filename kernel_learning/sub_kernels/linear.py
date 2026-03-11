import torch
import torch.nn as nn

from .base import BaseSubKernel

class LinearSubKernel(BaseSubKernel):
    """
    Linear kernel: K[i,j] = sigma^2 * x[i] * x[j]
    Equivalent to: K = sigma^2 * outer(x, x)
    """
    def __init__(self):
        super().__init__()

    
    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        # Vectorized: outer product. #NOTE: this is the same as doing x_col = x.unsqueeze(1)   # shape (n, 1); x_row = x.unsqueeze(0)   # shape (1, n); x_col @ x_row            
        K = torch.outer(x_col, x_col)
        return K
