"""Shared window definitions and variance filters for the frozen runner.
The module name is retained to preserve its original import.
"""
import numpy as np

WINDOWS = {
    "A_0-31": [0, 10, 15, 23, 25, 31],
    "B_0-15": [0, 10, 15],
    "C_0-50": [0, 10, 15, 23, 25, 31, 35, 50],
    "D_63-100": [63, 65, 75, 83, 90, 100],
    "E_23-54": [23, 25, 31, 35, 50, 54],
}

FILTERS = {"full": 710, "top639": 639, "top533": 533,
           "top355": 355, "top160": 160, "bottom160": 160}

def subsets(y, split, name):
    levels = WINDOWS[name]
    c = (split == "CALIBRATION") & np.isin(y, levels)
    test = np.isin(split, ["VALIDATION", "PREDICTION"])
    sides = {"above": test & (y >= max(levels) + 19),
             "below": test & (y <= min(levels) - 19)}
    return c, {s: mask for s, mask in sides.items() if mask.any()}

def filter_cols(X, method):
    if method == "full":
        return np.arange(X.shape[1])
    order = np.argsort(X.var(axis=0, ddof=1), kind="stable")
    return np.sort(order[:160] if method == "bottom160" else order[-FILTERS[method]:])
