import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

# ============================================================================
# SUB-KERNELS
# ============================================================================

class BaseSubKernel(nn.Module, ABC):
    """
    Base class for sub-kernels.
    Each sub-kernel operates on a single feature dimension k (for all samples)
    and produces a kernel matrix.
    
    Supports two modes:
        - Self-kernel: forward(x_col) -> (n, n) matrix
        - Cross-kernel: forward(x_col, y_col) -> (n1, n2) matrix
    """

    @abstractmethod
    def forward(self, x_col: torch.Tensor, y_col: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_col: Feature vector of shape (n1,) for a single feature dimension
            y_col: Optional feature vector of shape (n2,) for cross-kernel computation.
                   If None, computes self-kernel K(x, x).
        Returns:
            K: Kernel matrix of shape (n1, n1) if y_col is None,
               or (n1, n2) if y_col is provided.
        """
        pass
