import torch
import torch.nn as nn

from .base import BaseSubKernel

class RBFSubKernel(BaseSubKernel):
    """
    RBF (Gaussian) kernel: K[i,j] = exp(-gamma * (x[i] - x[j])^2)
    """
    def __init__(self, initial_gamma: float = 1.0, **kwargs):
        super().__init__()
        is_random = kwargs.get("random", False)
        if is_random:
            self.gamma= nn.Parameter(torch.randn(1).abs() + 0.1)  # Random positive gamma
        else:
            self.gamma = nn.Parameter(torch.tensor(initial_gamma))
    
    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        # Vectorized: broadcast to compute all pairwise differences
        diff = x_col.unsqueeze(1) - x_col.unsqueeze(0) #NOTE: unsqueeze(1) adds a dimension at position 1, making it (n, 1); unsqueeze(0) adds a dimension at position 0, making it (1, n) #NOTE: you might ask how can we do (n,1) - (1,n)? broadcasting allows this operation by expanding the dimensions of the tensors to be compatible for element-wise operations. meaning we will take each element in (n,1) and subtract it from each element in (1,n), resulting in a (n,n) matrix where each entry (i,j) is the difference between x_col[i] and x_col[j].
        K = torch.exp(-self.gamma * diff ** 2)
        return K
