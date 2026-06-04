import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        l = 0.0
        for i in range(n):
            l += y_true[i] * np.log(y_pred[i])
            l += (1-y_true[i]) * np.log(1 - y_pred[i])
        l = l/n
        l = l * -1
        return round(l, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n = len(y_true)
        c = len(y_true[0])
        l = 0.0
        for i in range(n):
            for j in range(c):
                l += (y_true[i][j] * np.log(y_pred[i][j]))
        l = l/n
        l = l * -1
        return round(l, 4)