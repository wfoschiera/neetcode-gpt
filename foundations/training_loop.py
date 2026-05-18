import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        w = np.zeros(shape=X.shape[1])
        beta = 0
        n = y.size
        for _ in range(epochs):
            y_hat = X @ w + beta

            dw = (2/n) * X.T @ (y_hat-y)
            db = (2/n) * sum(y_hat - y)

            w = w - lr * dw
            beta = beta - lr * db
            MSE = (1/n) * sum((y_hat - y)**2)
        
        return (np.round(w, 5), round(beta, 5))
