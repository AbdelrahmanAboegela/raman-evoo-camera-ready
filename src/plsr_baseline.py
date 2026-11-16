"""Full-spectrum PLSR baseline with CV-selected number of latent variables."""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from pls import PLSRegression
from sklearn.model_selection import KFold, GroupKFold


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rpd(y_true, y_pred):
    sd = y_true.std(ddof=1)
    r = rmse(y_true, y_pred)
    return float(sd / r) if r > 0 else float("inf")


def select_n_components(X, y, max_lv=10, n_folds=5, seed=0, groups=None, shuffle_groups=False):
    n_samples = X.shape[0]
    max_lv = min(max_lv, n_samples - 1, X.shape[1])
    if max_lv < 1:
        return 1
    n_splits = min(n_folds, n_samples)
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits, shuffle=shuffle_groups,
                              random_state=seed if shuffle_groups else None)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X)
    split_iter = list(split_iter)
    cv_rmse = []
    for lv in range(1, max_lv + 1):
        errs = []
        for tr_idx, va_idx in split_iter:
            model = PLSRegression(n_components=lv)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx]).ravel()
            errs.append(rmse(y[va_idx], pred))
        cv_rmse.append(np.mean(errs))
    return int(np.argmin(cv_rmse) + 1), cv_rmse


def fit_plsr(X_train, y_train, max_lv=10, groups=None, cv_seed=0, shuffle_groups=False):
    n_lv, cv_rmse = select_n_components(X_train, y_train, max_lv=max_lv, groups=groups,
                                       seed=cv_seed, shuffle_groups=shuffle_groups)
    model = PLSRegression(n_components=n_lv)
    model.fit(X_train, y_train)
    return model, n_lv


if __name__ == "__main__":
    import time
    from data_io import load_raw
    from preprocess import preprocess
    from regimes import get_regime_indices

    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    print("preprocessing all", X.shape[0], "spectra...")
    t0 = time.time()
    Xp = preprocess(X)
    print("done in", round(time.time() - t0, 1), "s")

    idx = get_regime_indices(y, split)

    for regime in ["narrow_cal", "full_cal"]:
        cal_mask = idx[regime]
        Xc, yc = Xp[cal_mask], y[cal_mask]
        model, n_lv = fit_plsr(Xc, yc)
        print(f"\n== PLSR fit on {regime} (n={cal_mask.sum()}, n_lv={n_lv}) ==")
        pred_c = model.predict(Xc).ravel()
        print(f"  calibration RMSEC={rmse(yc, pred_c):.3f} R2c={r2(yc, pred_c):.4f}")
        for test_name in ["test_in", "test_out"]:
            tm = idx[test_name]
            Xt, yt = Xp[tm], y[tm]
            pred_t = model.predict(Xt).ravel()
            print(f"  {test_name:9s} n={tm.sum():3d}  RMSEP={rmse(yt, pred_t):.3f}  "
                  f"R2p={r2(yt, pred_t):.4f}  RPD={rpd(yt, pred_t):.3f}  bias={float(np.mean(pred_t-yt)):+.3f}")
