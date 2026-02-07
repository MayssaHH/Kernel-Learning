import torch
## we will use the kernel trained by classification/run.py
## to train a kernel ridge classifier 
## and evaluate the performance on a test set
from kernel_learning import KernelNetwork

class KernelRidgeClassifier:
    def __init__(self, learnt_kernel: KernelNetwork, lambda_ridge: float = 1.0):
        # In kernel ridge classification we solve for dual coefficients alpha.
        # Here alpha will be (n_train, num_classes) to support multi-class.
        self.learnt_kernel = learnt_kernel
        self.lambda_ridge = lambda_ridge
        self.alpha = None
        self.X_train = None
        self.classes_ = None
    
    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        self.classes_, y_indices = torch.unique(y_train, sorted=True, return_inverse=True)
        self.X_train = X_train
        K = self.learnt_kernel(X_train)
        n = K.shape[0]
        num_classes = self.classes_.numel()
        y_one_hot = torch.nn.functional.one_hot(y_indices, num_classes=num_classes).to(device=K.device, dtype=K.dtype)
        eye = torch.eye(n, device=K.device, dtype=K.dtype)
        # Solve: (K + lambda * I) alpha = Y_one_hot
        self.alpha = torch.linalg.solve(K + self.lambda_ridge * eye, y_one_hot)
    
    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        if self.alpha is None or self.X_train is None or self.classes_ is None:
            raise RuntimeError("Call fit() before predict().")

        # KernelNetwork.forward computes K(Z, Z). Build Z=[X_test; X_train] and slice K(test, train).
        X_all = torch.cat([X_test, self.X_train], dim=0)
        K_all = self.learnt_kernel(X_all)
        n_test = X_test.shape[0]
        K_test_train = K_all[:n_test, n_test:]
        scores = K_test_train @ self.alpha  # (n_test, num_classes)
        pred_indices = torch.argmax(scores, dim=1)
        classes = self.classes_.to(pred_indices.device)
        return classes[pred_indices]
    
