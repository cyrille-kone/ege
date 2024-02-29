# coding=utf-8
r"""
PyCharm Editor
Author @git cyrille-kone
"""
import numpy as np
def load(ds="C")->np.ndarray:
    if ds=='C':
        return np.load("data/cov_boost.npy")
    elif ds=='I1':
        x = 0.75 - 0.55 ** (1 + np.arange(5))
        return np.array([x, x]).T
    elif ds=="I2":
        x = 0.5 - 0.025 * (1 + np.arange(15))
        return np.array([x, x]).T
    elif ds=="I3":
        pass
    elif ds=="I4":
        return np.concatenate([np.random.uniform(0.2, 0.4, size=(10, 2)),
                               np.random.uniform(0.5, 0.75, size=(10, 2))])
    return np.array([])