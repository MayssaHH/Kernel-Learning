import torch
import torch.nn.functional as F


class KernelRidgeClassifier:
    """
    Kernel Ridge Regression classifier.

    Solves the dual problem: (K + λI) α = Y_one_hot
    Predicts via: scores = K(X_test, X_train) @ α

    Works with any callable kernel: either a KernelNetwork or a plain
    function (X, Y) -> ndarray (for sklearn-style kernels).
    """

    def __init__(self, kernel, lambda_ridge: float = 1e-4):
        self.kernel = kernel
        self.lambda_ridge = lambda_ridge
        self.alpha = None
        self.X_train = None
        self.classes_ = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        self.classes_, y_indices = torch.unique(y_train, sorted=True, return_inverse=True)
        self.X_train = X_train
        K = self._self_kernel(X_train)
        n = K.shape[0]
        num_classes = self.classes_.numel()
        Y = F.one_hot(y_indices, num_classes=num_classes).to(dtype=K.dtype, device=K.device)
        eye = torch.eye(n, device=K.device, dtype=K.dtype)
        self.alpha = torch.linalg.solve(K + self.lambda_ridge * eye, Y)

    def predict(self, X_test: torch.Tensor) -> torch.Tensor:
        K_cross = self._cross_kernel(X_test, self.X_train)
        scores = K_cross @ self.alpha
        pred_indices = torch.argmax(scores, dim=1)
        return self.classes_[pred_indices]

    def _self_kernel(self, X: torch.Tensor) -> torch.Tensor:
        """Compute K(X, X)."""
        if hasattr(self.kernel, 'forward'):
            # KernelNetwork: its forward takes only X and computes self-kernel
            try:
                return self.kernel(X)
            except TypeError:
                pass
        # Callable (X, Y) -> ndarray or tensor
        return self._call_kernel(X, X)

    def _cross_kernel(self, X_test: torch.Tensor, X_train: torch.Tensor) -> torch.Tensor:
        """Compute K(X_test, X_train)."""
        if hasattr(self.kernel, 'forward'):
            # KernelNetwork: concatenate and slice (existing approach)
            X_all = torch.cat([X_test, X_train], dim=0)
            K_all = self.kernel(X_all)
            n_test = X_test.shape[0]
            return K_all[:n_test, n_test:]
        return self._call_kernel(X_test, X_train)

    def _call_kernel(self, X, Y) -> torch.Tensor:
        import numpy as np
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        Y_np = Y.numpy() if isinstance(Y, torch.Tensor) else Y
        result = self.kernel(X_np, Y_np)
        if isinstance(result, np.ndarray):
            return torch.from_numpy(result).float()
        return result
