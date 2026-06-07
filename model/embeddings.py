import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        ans = []
        for i in token_ids:
            ans.append([round(x, 5) for x in embeddings[i]])
        return ans