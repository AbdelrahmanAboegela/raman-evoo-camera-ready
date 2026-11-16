"""
Additional variable-selection baselines (Review 1: "no deterministic baseline beyond VIP").
All use calibration data only; model selection by k-fold CV inside the calibration set,
matching the protocol used for the other methods.

  ipls_select : forward interval PLS (Norgaard et al., 2000) - deterministic given folds
  spa_select  : successive projections algorithm + MLR (Araujo et al., 2001) - deterministic
  cars_select : competitive adaptive reweighted sampling (Li et al., 2009) - stochastic
  enpls_select: elastic-net variable selection (PLS is fitted on the retained variables by the caller)

Each returns the array of selected column indices.
"""
import numpy as np
from pls import PLSRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold

from plsr_baseline import select_n_components


def _folds(n, n_folds=5, seed=0, groups=None, shuffle_groups=False):
    if groups is not None:
        return list(GroupKFold(n_splits=min(n_folds, len(np.unique(groups))),
                               shuffle=shuffle_groups, random_state=seed if shuffle_groups else None)
                    .split(np.arange(n), groups=groups))
    return list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n)))


def _cv_pls(X, y, folds, n_lv):
    n_lv = max(1, min(n_lv, X.shape[1], min(len(t) for t, _ in folds) - 1))
    err = []
    for tr, va in folds:
        m = PLSRegression(n_components=n_lv).fit(X[tr], y[tr])
        err.append(np.sqrt(np.mean((y[va] - m.predict(X[va]).ravel()) ** 2)))
    return float(np.mean(err))


# ------------------------------------------------------------------ iPLS ----
def ipls_select(X, y, n_intervals=20, max_lv=10, cv_seed=0, groups=None, shuffle_groups=False):
    p = X.shape[1]
    edges = np.linspace(0, p, n_intervals + 1).astype(int)
    intervals = [np.arange(edges[i], edges[i + 1]) for i in range(n_intervals)]
    folds = _folds(len(y), seed=cv_seed, groups=groups, shuffle_groups=shuffle_groups)

    def score(cols):
        return _cv_pls(X[:, cols], y, folds, max_lv)

    chosen, best = [], np.inf
    remaining = list(range(n_intervals))
    while remaining:
        cands = [(score(np.concatenate([intervals[i] for i in chosen + [j]])), j) for j in remaining]
        s, j = min(cands)
        if s < best - 1e-9:
            best, chosen = s, chosen + [j]
            remaining.remove(j)
        else:
            break
    return np.sort(np.concatenate([intervals[i] for i in chosen]))


# ------------------------------------------------------------------- SPA ----
def _spa_chain(Xn, k0, K):
    proj = Xn.copy()
    sel = [k0]
    for _ in range(K - 1):
        v = proj[:, sel[-1]]
        proj = proj - np.outer(v, (proj.T @ v) / (v @ v))
        norms = (proj ** 2).sum(0)
        norms[sel] = -1.0
        sel.append(int(np.argmax(norms)))
    return sel


def _cv_mlr(Xs, y, folds):
    err = []
    for tr, va in folds:
        A = np.c_[np.ones(len(tr)), Xs[tr]]
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred = np.c_[np.ones(len(va)), Xs[va]] @ beta
        err.append(np.sqrt(np.mean((y[va] - pred) ** 2)))
    return float(np.mean(err))


def spa_select(X, y, k_max=30, cv_seed=0, start_stride=1):
    Xc = X - X.mean(0)
    n = Xc.shape[0]
    k_max = min(k_max, n // 3)
    folds = _folds(n, seed=cv_seed)
    best = (np.inf, None)
    for k0 in range(0, X.shape[1], start_stride):
        chain = _spa_chain(Xc, k0, k_max)
        for K in range(2, k_max + 1):
            s = _cv_mlr(Xc[:, chain[:K]], y, folds)
            if s < best[0]:
                best = (s, chain[:K])
    return np.sort(np.array(best[1]))


# ------------------------------------------------------------------ CARS ----
def cars_select(X, y, n_runs=50, max_lv=10, sample_frac=0.8, cv_seed=0, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    a = (p / 2.0) ** (1.0 / (n_runs - 1))
    k = np.log(p / 2.0) / (n_runs - 1)
    keep = np.arange(p)
    folds = _folds(n, seed=cv_seed)
    best = (np.inf, keep)
    for i in range(1, n_runs + 1):
        n_i = max(2, int(round(p * a * np.exp(-k * i))))
        ix = rng.choice(n, size=int(sample_frac * n), replace=False)
        lv = min(max_lv, len(keep), len(ix) - 2)
        m = PLSRegression(n_components=max(1, lv)).fit(X[ix][:, keep], y[ix])
        w = np.abs(m.coef_.ravel())
        if i > 1:  # EDF: force-keep the n_i most important, then ARS among them
            order = np.argsort(-w)[: min(len(keep), max(n_i, 2))]
            keep, w = keep[order], w[order]
            if len(keep) > n_i:
                prob = w / w.sum() if w.sum() > 0 else None
                keep = np.unique(rng.choice(keep, size=n_i, replace=True, p=prob))
        s = _cv_pls(X[:, keep], y, folds, max_lv)
        if s < best[0]:
            best = (s, keep.copy())
    return np.sort(best[1])


# ----------------------------------------------------------------- EN-PLS ----
def enpls_select(X, y, cv_seed=0):
    """Variables with a non-zero elastic-net coefficient (penalty and l1 ratio chosen by 5-fold CV)."""
    cv = list(KFold(n_splits=5, shuffle=True, random_state=cv_seed).split(np.arange(len(y))))
    en = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], alphas=40, cv=cv, max_iter=20000, tol=1e-4).fit(X, y)
    cols = np.where(np.abs(en.coef_) > 0)[0]
    if len(cols) < 2:
        cols = np.argsort(-np.abs(en.coef_))[:2]
    return np.sort(cols)
