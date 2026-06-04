import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = [1/(1 + np.exp(-x)) for x in z]
        return [np.round(x, 5) for x in z]

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        return [np.round(max(0.0, x), 5) for x in z]