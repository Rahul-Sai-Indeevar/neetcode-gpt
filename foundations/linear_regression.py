import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # n = len(X)
        # m = len(X[0])
        pred = np.dot(X, weights)
        return np.round(pred, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        diff = model_prediction - ground_truth
        e = np.square(diff)
        mse = np.mean(e)
        return round(float(mse), 5)