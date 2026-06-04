import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x,w) + b
        if activation == "relu":
            z = max(0.0, z)
        elif activation == "sigmoid":
            z = 1 / (1 + np.exp(-z))
        return round(z, 5)