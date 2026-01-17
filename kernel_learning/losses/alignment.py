import torch

from .base import BaseLoss

#This is an example of a loss that uses different inputs.
class AlignmentLoss(BaseLoss):
    """
    TODO: for ahmad, design this loss as you are expecting it and do a PR on main
    Kernel alignment loss (placeholder - implement based on your needs).
    
    """
    def forward(self, K_pred: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # Example: alignment with label-based kernel
        # K_y[i,j] = 1 if y[i] == y[j] else 0
        n = len(y)
        y_mat = y.unsqueeze(1).expand(n, n)
        K_y = (y_mat == y_mat.T).float()
        
        # Centered kernel alignment
        K_pred_centered = K_pred - K_pred.mean(dim=0, keepdim=True) - K_pred.mean(dim=1, keepdim=True) + K_pred.mean()
        K_y_centered = K_y - K_y.mean(dim=0, keepdim=True) - K_y.mean(dim=1, keepdim=True) + K_y.mean()
        
        alignment = torch.sum(K_pred_centered * K_y_centered)
        norm_pred = torch.sqrt(torch.sum(K_pred_centered ** 2) + 1e-8)
        norm_y = torch.sqrt(torch.sum(K_y_centered ** 2) + 1e-8)
        
        # Return negative alignment (we want to maximize alignment, so minimize negative)
        return -alignment / (norm_pred * norm_y)
