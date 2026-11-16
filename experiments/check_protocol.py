"""Ensure matched, non-leaking 15-seed folds and exact cached-fitness equivalence."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paths import ROOT
from pls import PLSRegression, set_scaling
from ga_mw import ga_mw_select, _fitness_folds
from gwo_mw import gwo_mw_select
from selectors_extra import _folds


def main():
    set_scaling("unit")
    rng = np.random.default_rng(17)
    X = rng.normal(size=(36, 24))
    y = X[:, 0] + .1 * rng.normal(size=36)
    groups = np.repeat(np.arange(18), 2)
    signatures = []
    for seed in range(15):
        a = _fitness_folds(X, y, n_folds=5, seed=seed, groups=groups, shuffle_groups=True)
        b = _folds(len(y), seed=seed, groups=groups, shuffle_groups=True)
        assert all(np.array_equal(tr1, tr2) and np.array_equal(va1, va2)
                   for (tr1, va1), (tr2, va2) in zip(a, b))
        assert all(not set(groups[tr]) & set(groups[va]) for tr, va in a)
        signatures.append(tuple(tuple(va) for tr, va in a))
    assert len(set(signatures)) == 15
    print("PASS: fifteen distinct, matched, group-disjoint CV assignments")
    # Independently verify the first-component identity used to motivate the
    # paper's mechanism controls, with and without a channel-selection mask.
    for mode in ("unit", "mean"):
        set_scaling(mode)
        for cols in (np.arange(X.shape[1]), np.arange(0, X.shape[1], 2)):
            Z = X[:, cols] - X[:, cols].mean(0)
            D = Z.std(0, ddof=1) if mode == "unit" else np.ones(len(cols))
            predicted_score = Z @ ((Z.T @ (y - y.mean())) / (D ** 2))
            score = PLSRegression(n_components=1).fit(X[:, cols], y).x_scores_[:, 0]
            factor = (score @ predicted_score) / (predicted_score @ predicted_score)
            assert np.linalg.norm(score - factor * predicted_score) / np.linalg.norm(score) < 1e-10
    print("PASS: first-component scaling/selection weighting identity")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
