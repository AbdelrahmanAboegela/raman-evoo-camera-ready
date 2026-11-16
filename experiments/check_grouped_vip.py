"""Regression check: VIP must propagate groups into every inner/final LV fit."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vip_cv_select as vip
from plsr_baseline import fit_plsr


def main():
    rng = np.random.default_rng(73)
    g = np.repeat(np.arange(12), 2)
    X = rng.normal(size=(24, 12))
    y = X[:, 0] + rng.normal(scale=0.1, size=24)
    calls = []

    def audited_fit(Xt, yt, max_lv=10, groups=None, cv_seed=0, shuffle_groups=False):
        assert groups is not None, "Group labels lost in nested/final VIP fit"
        assert len(groups) == len(yt)
        for tr, va in GroupKFold(5).split(Xt, yt, groups):
            assert not set(groups[tr]) & set(groups[va])
        calls.append(groups.copy())
        return fit_plsr(Xt, yt, max_lv=max_lv, groups=groups,
                        cv_seed=cv_seed, shuffle_groups=shuffle_groups)

    with patch.object(vip, "fit_plsr", side_effect=audited_fit):
        k, _ = vip.select_k_by_cv(X, y, grid=[5, 8], groups=g, max_lv=2)
        vip.fit_final_vip_plsr(X, y, k, max_lv=2, groups=g)
    assert len(calls) == 11  # two k values x five outer folds + final fit
    assert np.array_equal(calls[-1], g)
    print("PASS: all ten inner LV fits and the final LV fit receive disjoint mixture groups")


if __name__ == "__main__":
    main()
