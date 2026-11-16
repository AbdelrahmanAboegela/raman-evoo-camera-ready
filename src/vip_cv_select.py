"""
The full, leakage-free VIP-PLSR protocol as an actual saved script (this was
previously only run as an inline shell command -- not real, released code).

select_k_by_cv() takes ONLY calibration data (Xc, yc) and returns the k that
minimizes 5-fold CV error computed entirely within that calibration set. It
must never be passed, or reference, the test_in/test_out arrays -- verified
structurally by verify_no_leakage.py, which inspects this function's source.
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from sklearn.model_selection import KFold, GroupKFold

from vip_selection import vip_select
from plsr_baseline import fit_plsr, rmse

GRID = [60, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 250]


def select_k_by_cv(Xc, yc, grid=GRID, cv_seed=0, n_splits=5, groups=None, max_lv=10,
                   shuffle_groups=False):
    """CV entirely within the calibration set Xc/yc. Returns the k (number of
    VIP-ranked wavenumbers to keep) with the lowest CV-RMSE. If groups is
    given, uses GroupKFold so replicate spectra of one mixture never split
    across the train/validation side of a fold."""
    if groups is not None:
        n_splits_eff = min(n_splits, len(np.unique(groups)))
        split_iter = list(GroupKFold(n_splits=n_splits_eff, shuffle=shuffle_groups,
                                    random_state=cv_seed if shuffle_groups else None).split(Xc, yc, groups))
    else:
        split_iter = list(KFold(n_splits=n_splits, shuffle=True, random_state=cv_seed).split(Xc))
    cv_scores = {}
    for k in grid:
        errs = []
        for tr, va in split_iter:
            cols, _ = vip_select(Xc[tr], yc[tr], n_keep=k, max_lv=max_lv)
            m, _ = fit_plsr(Xc[tr][:, cols], yc[tr], max_lv=max_lv,
                            groups=None if groups is None else groups[tr],
                            cv_seed=cv_seed if shuffle_groups else 0, shuffle_groups=shuffle_groups)
            pred = m.predict(Xc[va][:, cols]).ravel()
            errs.append(rmse(yc[va], pred))
        cv_scores[k] = float(np.mean(errs))
    best_k = min(cv_scores, key=cv_scores.get)
    return best_k, cv_scores


def fit_final_vip_plsr(Xc, yc, k, max_lv=10, groups=None, cv_seed=0, shuffle_groups=False):
    """Freeze k, rank on the FULL calibration set, fit the final model."""
    cols, vips = vip_select(Xc, yc, n_keep=k, max_lv=max_lv)
    model, n_lv = fit_plsr(Xc[:, cols], yc, max_lv=max_lv, groups=groups,
                          cv_seed=cv_seed, shuffle_groups=shuffle_groups)
    return model, cols, vips, n_lv


if __name__ == "__main__":
    from data_io import load_raw
    from preprocess import preprocess
    from regimes import get_regime_indices

    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    Xp = preprocess(X).astype(np.float32)
    idx = get_regime_indices(y, split)
    cal_mask = idx["narrow_cal"]
    Xc, yc = Xp[cal_mask], y[cal_mask]

    results = []
    for cv_seed in range(5):
        best_k, cv_scores = select_k_by_cv(Xc, yc, cv_seed=cv_seed)
        model, cols, vips, n_lv = fit_final_vip_plsr(Xc, yc, best_k)

        pred_out = model.predict(Xp[idx["test_out"]][:, cols]).ravel()
        pred_in = model.predict(Xp[idx["test_in"]][:, cols]).ravel()
        r_out = rmse(y[idx["test_out"]], pred_out)
        r_in = rmse(y[idx["test_in"]], pred_in)
        results.append(r_out)
        print(f"cv_seed={cv_seed}  best_k={best_k:4d}  n_lv={n_lv}  "
              f"RMSEP_in={r_in:.3f}  RMSEP_out={r_out:.3f}")

    results = np.array(results)
    print(f"\nVIP-PLSR RMSEP_out across 5 CV-fold choices: "
          f"mean={results.mean():.3f} std={results.std():.3f} "
          f"min={results.min():.3f} max={results.max():.3f}")
