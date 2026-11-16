"""
GA-MW: genetic-algorithm moving-window spectral interval selection + PLSR,
re-implemented from Zhu et al. (2023) Sec 2.5.1-2.5.2 (Eq. 5) since no
code/data from that paper is public.

The defaults (pop=40, iters=40) are the reduced search budget of the study;
the budget published by Zhu et al. (pop=100, iters=1000) is passed
explicitly by the experiment scripts.
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from pls import PLSRegression
from sklearn.model_selection import KFold, GroupKFold

WINDOW_WIDTH = 5


def _windows_to_columns(positions, n_wn, width=WINDOW_WIDTH):
    cols = set()
    for p in positions:
        p = int(np.clip(p, 0, n_wn - width))
        cols.update(range(p, p + width))
    return np.array(sorted(cols))


def _fitness_folds(X, y, n_folds=3, seed=0, groups=None, shuffle_groups=False):
    if groups is not None:
        n_splits = min(n_folds, len(np.unique(groups)))
        return list(GroupKFold(n_splits=n_splits, shuffle=shuffle_groups,
                               random_state=seed if shuffle_groups else None).split(X, y, groups))
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))


def _cv_rmsecv(X, y, cols, n_components, n_folds=3, seed=0, groups=None,
               cv_splits=None, shuffle_groups=False):
    if len(cols) < n_components:
        return 1e6
    Xs = X[:, cols]
    split_iter = cv_splits if cv_splits is not None else _fitness_folds(
        X, y, n_folds=n_folds, seed=seed, groups=groups, shuffle_groups=shuffle_groups)
    errs = []
    for tr, va in split_iter:
        m = PLSRegression(n_components=min(n_components, len(tr) - 1))
        m.fit(Xs[tr], y[tr])
        pred = m.predict(Xs[va]).ravel()
        errs.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
    return float(np.mean(errs))


def ga_mw_select(X, y, n_windows=30, pop_size=40, n_iters=40, n_components=5,
                 pc=0.6, pm=0.03, seed=0, verbose=False, groups=None, shuffle_groups=False):
    rng = np.random.default_rng(seed)
    n_wn = X.shape[1]
    max_start = n_wn - WINDOW_WIDTH

    pop = rng.integers(0, max_start + 1, size=(pop_size, n_windows))
    cv_splits = _fitness_folds(X, y, seed=seed, groups=groups, shuffle_groups=shuffle_groups)

    def fitness_of(ind):
        cols = _windows_to_columns(ind, n_wn)
        return _cv_rmsecv(X, y, cols, n_components, cv_splits=cv_splits)

    fitness = np.array([fitness_of(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_ind, best_fit = pop[best_idx].copy(), fitness[best_idx]
    history = [best_fit]

    for it in range(n_iters):
        # roulette-wheel selection on 1/RMSECV (lower RMSECV -> higher prob)
        inv = 1.0 / (fitness + 1e-9)
        probs = inv / inv.sum()
        parents_idx = rng.choice(pop_size, size=pop_size, p=probs)
        parents = pop[parents_idx]

        # crossover: pair consecutive parents, swap genes with prob pc
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            mask = rng.random(n_windows) < pc
            children[i][mask], children[i + 1][mask] = parents[i + 1][mask], parents[i][mask]

        # mutation
        mut_mask = rng.random((pop_size, n_windows)) < pm
        rand_vals = rng.integers(0, max_start + 1, size=(pop_size, n_windows))
        children = np.where(mut_mask, rand_vals, children)

        child_fitness = np.array([fitness_of(ind) for ind in children])

        # elitism: keep best-of-generation, replace worst child with running best
        worst_child = np.argmax(child_fitness)
        children[worst_child] = best_ind
        child_fitness[worst_child] = best_fit

        pop, fitness = children, child_fitness
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fit:
            best_fit = fitness[gen_best_idx]
            best_ind = pop[gen_best_idx].copy()
        history.append(best_fit)
        if verbose and it % 10 == 0:
            print(f"    gen {it:3d}  best RMSECV={best_fit:.4f}")

    best_cols = _windows_to_columns(best_ind, n_wn)
    return best_cols, best_ind, history


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
        cols, positions, hist = ga_mw_select(Xc, yc, verbose=False)
        dt = time.time() - t0
        n_sel = len(cols)
        print(f"\n== GA-MW on {regime} (n={cal_mask.sum()}) == "
              f"selected {n_sel} vars in {dt:.1f}s, final RMSECV={hist[-1]:.4f}")
        print("   selected wavenumbers (cm-1), first/last 5:",
              np.round(wn[cols[:5]], 1).tolist(), "...", np.round(wn[cols[-5:]], 1).tolist())

        model, n_lv = fit_plsr(Xc[:, cols], yc, max_lv=10)
        pred_c = model.predict(Xc[:, cols]).ravel()
        print(f"   final PLSR n_lv={n_lv}  RMSEC={rmse(yc, pred_c):.3f}  R2c={r2(yc, pred_c):.4f}")
        for test_name in ["test_in", "test_out"]:
            tm = idx[test_name]
            Xt, yt = Xp[tm][:, cols], y[tm]
            pred_t = model.predict(Xt).ravel()
            print(f"   {test_name:9s} n={tm.sum():3d}  RMSEP={rmse(yt, pred_t):.3f}  "
                  f"R2p={r2(yt, pred_t):.4f}  RPD={rpd(yt, pred_t):.3f}  bias={float(np.mean(pred_t-yt)):+.3f}")
