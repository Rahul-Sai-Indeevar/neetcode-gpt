import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        m = max(z);
        z = [np.exp(x-m) for x in z]
        d = sum(z)
        return [round(x / d, 4) for x in z]