import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:

        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        e = z2 - y_true
        L = np.mean(e*e)
        n = len(y_true)
        dy = (2/n) * e
        dW2 = np.outer(dy, a1)
        db2 = dy
        da1 = W2.T @ dy
        dz1 = da1 * (z1 > 0.0)
        dW1 = np.outer(dz1, x)
        db1 = dz1
        dW1 = np.round(dW1, 4)
        db1 = np.round(db1, 4)
        dW2 = np.round(dW2, 4)
        db2 = np.round(db2, 4)
        return {
            "loss": round(float(L), 4),
            "dW1": dW1.tolist(),
            "db1": db1.tolist(),
            "dW2": dW2.tolist(),
            "db2": db2.tolist()
        }