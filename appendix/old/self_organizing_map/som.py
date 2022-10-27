import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class SOM(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, m: int, n: int, seed: int = 0, sigma: float = 1,
                 lr: float = 1, max_iter: int = 3_000) -> None:
        super().__init__()
        self.m = m
        self.n = n
        self.seed = seed
        self.max_iter = max_iter
        self.lr = lr
        self.sigma = sigma
        self.weights = None
        self._rng = np.random.default_rng(seed)
        self._initial_lr = lr
        self._indices = np.indices((self.m, self.n))

    def _neighbourhood(self, idx: int) -> np.ndarray:
        norm = np.linalg.norm(
            np.array(idx)[:, np.newaxis, np.newaxis] - self._indices,
            axis=0)**2
        return np.exp(-norm/self.sigma**2)

    def fit(self, X: np.ndarray, epochs: int = 1) -> "SOM":
        self._validate_data(X)
        self.weights = self._rng.normal(
            size=(self.m, self.n, self.n_features_in_))
        n_samples = X.shape[0]
        n_iterations = 0
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)
        for _ in range(epochs):
            if n_iterations > self.max_iter:
                break
            indices = self._rng.permutation(n_samples)
            for idx in indices:
                x = X[idx]
                diff = x[np.newaxis, np.newaxis, :] - self.weights
                dist = np.linalg.norm(diff, axis=-1)
                bmu = np.unravel_index(np.argmin(dist), dist.shape)
                neighbourhood = self._neighbourhood(bmu)[..., None]
                self.weights += self.lr * neighbourhood * diff
                n_iterations += 1
                self.lr = (1 - (n_iterations / total_iterations)) * \
                    self._initial_lr
        return self

    def predict(self, X: np.ndarray):
        n_samples = X.shape[0]
        dist = np.linalg.norm(
            X[:, np.newaxis, np.newaxis, :] - self.weights, axis=-1)
        return np.argmin(dist.reshape(n_samples, -1), axis=-1)
