import torch

from .base import BaseTrainer
from kernel_learning.kernel_network import KernelNetwork # you can even just do from kernel_learning import KernelNetwork because of __init__.py
from kernel_learning.losses import BaseLoss

class ManualGradientTrainer(BaseTrainer):
    """
    Manual gradient descent trainer
    Updates parameters manually: param -= lr * grad NOTE: here I am assuming my loss function is to minimize the loss, so I subtract the gradient scaled by learning rate from the parameters.
    """
    def __init__(self, lr: float = 0.001, grad_clip: float = 1.0):
        self.lr = lr
        self.grad_clip = grad_clip
    
    def train_epoch(
        self,
        model: KernelNetwork,
        X: torch.Tensor,
        criterion: BaseLoss,
        lambda_lasso: float = 0.0,
        **loss_kwargs
    ) -> float:
        # Forward pass
        K_pred = model(X)
        
        # Compute loss
        loss = criterion(K_pred, **loss_kwargs)
        
        # Add Lasso penalty
        if lambda_lasso > 0:
            loss = loss + model.get_lasso_penalty(lambda_lasso)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        # Manual gradient update
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= self.lr * param.grad
                    param.grad.zero_() # Because PyTorch accumulates gradients by default. #NOTE: this is conceptually the same as optimizer.zero_grad() in the optimizer-based trainer.
        
        return loss.item()  # float value of the loss, used for logging because we have already done loss.backward() and updated the parameters.
