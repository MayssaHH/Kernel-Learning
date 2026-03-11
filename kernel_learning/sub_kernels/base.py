import torch
import torch.nn as nn
from abc import ABC, abstractmethod

#NOTE: ABC It lets you define: a required interface for subclasses, without providing an implementation 
 
# ============================================================================
# SUB-KERNELS
# ============================================================================

class BaseSubKernel(nn.Module, ABC):
    """
    Base class for sub-kernels.
    Each sub-kernel operates on a single feature dimension k (#NOTE for all samples) and produces
    an (n, n) kernel matrix.
    """
    @abstractmethod # Every subclass of BaseSubKernel must implement forward with this signature.
    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_col: Feature vector of shape (n,) for a single feature dimension
            
        Returns:
            K: Kernel matrix of shape (n, n)
        """
        pass

# the following are example sub-kernels:
