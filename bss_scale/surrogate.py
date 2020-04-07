# Some norm functions with the weights for MM algorithms
# Copyright (C) 2020  Robin Scheibler
import numpy as np

eps = 1e-15


def lp_norm(E, p=1):
    assert p > 0 and p < 2
    cost = np.sum(np.abs(E) ** p)
    weights = p / np.maximum(eps, 2.0 * np.abs(E) ** (2 - p))
    return cost, weights


def lpq_norm(E, p=1, q=2, axis=1):
    assert p > 0 and q >= p and q <= 2.0

    cost = np.sum(np.sum(np.abs(E) ** q, axis=axis, keepdims=True) ** (p / q))
    rn = np.sum(np.abs(E) ** q, axis=axis, keepdims=True) ** (1 - p / q)
    qfn = np.abs(E) ** (2 - q)
    weights = p / np.maximum(eps, 2.0 * rn * qfn)
    return cost, weights
