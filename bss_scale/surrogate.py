# Some norm functions with the weights for MM algorithms
# Copyright (C) 2020  Robin Scheibler
import numpy as np

eps = 1e-15


def lp_norm(E, p=1):
    assert p > 0 and p < 2
    weights = p / np.maximum(eps, 2.0 * np.abs(E) ** (2 - p))
    return weights


def lpq_norm(E, p=1, q=2, axis=1):
    assert p > 0 and q >= p and q <= 2.0

    rn = np.sum(np.abs(E) ** q, axis=axis, keepdims=True) ** (1 - p / q)
    qfn = np.abs(E) ** (2 - q)
    weights = p / np.maximum(eps, 2.0 * rn * qfn)
    return weights
