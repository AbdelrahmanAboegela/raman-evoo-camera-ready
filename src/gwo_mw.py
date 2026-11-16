"""
GWO-MW: grey-wolf-optimizer moving-window spectral interval selection + PLSR,
re-implemented from Zhu et al. (2023) Sec 2.5.4 (Eq. 8-18).

The defaults (n=20, iters=40) are the reduced search budget of the study;
the budget published by Zhu et al. (n=20, I_max=1000) is passed explicitly
by the experiment scripts.
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from ga_mw import _windows_to_columns, _cv_rmsecv, _fitness_folds, WINDOW_WIDTH


def gwo_mw_select(X, y, n_windows=30, pop_size=20, n_iters=40, n_components=5, seed=0,
                  verbose=False, groups=None, shuffle_groups=False):
    rng = np.random.default_rng(seed)
    n_wn = X.shape[1]
    max_start = n_wn - WINDOW_WIDTH

    positions = rng.uniform(0, max_start, size=(pop_size, n_windows))
    cv_splits = _fitness_folds(X, y, seed=seed, groups=groups, shuffle_groups=shuffle_groups)

    def fitness_of(pos_vec):
        cols = _windows_to_columns(np.round(pos_vec).astype(int), n_wn)
        return _cv_rmsecv(X, y, cols, n_components, cv_splits=cv_splits)

    fitness = np.array([fitness_of(p) for p in positions])
    order = np.argsort(fitness)
    alpha, beta, delta = positions[order[0]].copy(), positions[order[1]].copy(), positions[order[2]].copy()
    alpha_fit = fitness[order[0]]
    history = [alpha_fit]

    for it in range(n_iters):
        a = 2 - it * (2.0 / n_iters)  # linearly decreases 2 -> 0
        new_positions = np.empty_like(positions)
        for i in range(pop_size):
            X_new = np.zeros(n_windows)
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(n_windows), rng.random(n_windows)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - positions[i])
                X_new += leader - A * D
            new_positions[i] = np.clip(X_new / 3.0, 0, max_start)
        positions = new_positions
        fitness = np.array([fitness_of(p) for p in positions])
        order = np.argsort(fitness)
        if fitness[order[0]] < alpha_fit:
            alpha_fit = fitness[order[0]]
            alpha = positions[order[0]].copy()
        beta, delta = positions[order[1]].copy(), positions[order[2]].copy()
        history.append(alpha_fit)
        if verbose and it % 10 == 0:
            print(f"    gen {it:3d}  best RMSECV={alpha_fit:.4f}")

    best_cols = _windows_to_columns(np.round(alpha).astype(int), n_wn)
    return best_cols, alpha, history


if __name__ == "__main__":
    import time
    from data_io import load_raw
    from preprocess import preprocess
    from regimes import get_regime_indices
    from plsr_baseline import rmse, r2, rpd, fit_plsr

    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    Xp = preprocess(X)
    idx = get_regime_indices(y, split)

    for regime in ["narrow_cal", "full_cal"]:
        cal_mask = idx[regime]
        Xc, yc = Xp[cal_mask], y[cal_mask]
        t0 = time.time()
        cols, positions, hist = gwo_mw_select(Xc, yc, verbose=False)
        dt = time.time() - t0
        print(f"\n== GWO-MW on {regime} (n={cal_mask.sum()}) == "
              f"selected {len(cols)} vars in {dt:.1f}s, final RMSECV={hist[-1]:.4f}")

        model, n_lv = fit_plsr(Xc[:, cols], yc, max_lv=10)
        pred_c = model.predict(Xc[:, cols]).ravel()
        print(f"   final PLSR n_lv={n_lv}  RMSEC={rmse(yc, pred_c):.3f}  R2c={r2(yc, pred_c):.4f}")
        for test_name in ["test_in", "test_out"]:
            tm = idx[test_name]
            Xt, yt = Xp[tm][:, cols], y[tm]
            pred_t = model.predict(Xt).ravel()
            print(f"   {test_name:9s} n={tm.sum():3d}  RMSEP={rmse(yt, pred_t):.3f}  "
                  f"R2p={r2(yt, pred_t):.4f}  RPD={rpd(yt, pred_t):.3f}  bias={float(np.mean(pred_t-yt)):+.3f}")
