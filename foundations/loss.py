import numpy as np
from numpy.typing import NDArray


class Solution:
    EPSILON = 0.0000001

    def binary_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return round(loss, 4)

    def categorical_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        y_pred = np.clip(y_pred, self.EPSILON, 1 - self.EPSILON)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        return round(loss, 4)
