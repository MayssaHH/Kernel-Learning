"""
Baseline classifiers for the classification-v2 benchmark.

Each baseline exposes a uniform interface:
    fit(X_train_np, y_train_np) -> self
    predict(X_test_np) -> y_pred_np
    score(X_test_np, y_test_np) -> float  (accuracy)
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.svm import SVC


class _SklearnWrapper:
    def __init__(self, clf):
        self.clf = clf

    def fit(self, X_tr, y_tr):
        self.clf.fit(X_tr, y_tr)
        return self

    def predict(self, X_te):
        return self.clf.predict(X_te)

    def score(self, X_te, y_te):
        return float((self.predict(X_te) == y_te).mean())


def make_svm_rbf(C=1.0, gamma="scale"):
    return _SklearnWrapper(SVC(kernel="rbf", C=C, gamma=gamma))


def make_svm_linear(C=1.0):
    return _SklearnWrapper(SVC(kernel="linear", C=C))


def make_svm_poly(degree=3, C=1.0):
    return _SklearnWrapper(SVC(kernel="poly", degree=degree, C=C))


def make_gp_rbf():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
    return _SklearnWrapper(GaussianProcessClassifier(kernel=kernel, random_state=42))


class KRR_Global:
    """
    Kernel Ridge Regression with a *global* kernel over the full feature vector.
    This is the direct 'flat' comparison to our per-feature KernelNetwork.
    """

    def __init__(self, kernel_fn, lambda_ridge: float = 1e-4):
        """
        kernel_fn: callable(X, Y) -> (n1, n2) ndarray
        """
        self.kernel_fn = kernel_fn
        self.lambda_ridge = lambda_ridge
        self.alpha = None
        self.X_train = None
        self.classes_ = None

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray):
        self.classes_ = np.unique(y_tr)
        self.X_train = X_tr
        K = self.kernel_fn(X_tr, X_tr)
        n = K.shape[0]
        labels = np.searchsorted(self.classes_, y_tr)
        num_classes = len(self.classes_)
        Y = np.eye(num_classes)[labels]
        self.alpha = np.linalg.solve(K + self.lambda_ridge * np.eye(n), Y)
        return self

    def predict(self, X_te: np.ndarray) -> np.ndarray:
        K_cross = self.kernel_fn(X_te, self.X_train)
        scores = K_cross @ self.alpha
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, X_te: np.ndarray, y_te: np.ndarray) -> float:
        return float((self.predict(X_te) == y_te).mean())


def rbf_kernel_global(gamma: float = 1.0):
    """Return a global RBF kernel function."""
    def _k(X, Y):
        diff = X[:, None, :] - Y[None, :, :]
        return np.exp(-gamma * (diff ** 2).sum(axis=-1))
    return _k


def linear_kernel_global():
    """Return a global linear kernel function."""
    def _k(X, Y):
        return X @ Y.T
    return _k


def polynomial_kernel_global(degree: int = 3, c: float = 1.0):
    """Return a global polynomial kernel function."""
    def _k(X, Y):
        return (X @ Y.T + c) ** degree
    return _k
