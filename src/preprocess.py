"""
Spectral preprocessing pipeline matching Zhu et al. (2023) Sec 2.3:
  1) airPLS baseline correction
  2) SNV (standard normal variate)
  3) Savitzky-Golay smoothing
  4) centring, and by default scaling to unit variance, applied inside the PLS
     model at fit time (not here); see src/pls.py for the scaling switch
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.signal import savgol_filter


def airpls(y, lam=1e5, itermax=15):
    """Adaptive iteratively reweighted penalized least squares baseline (Zhang et al. 2010)."""
    m = y.shape[0]
    w = np.ones(m)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(m, m - 2))
    DD = lam * D.dot(D.T)
    baseline = np.zeros(m)
    for i in range(1, itermax + 1):
        W = sparse.diags(w, 0)
        Z = (W + DD).tocsc()
        baseline = splinalg.spsolve(Z, w * y)
        d = y - baseline
        dn = d[d < 0]
        if len(dn) == 0 or np.abs(dn.sum()) < 1e-3 * np.abs(y).sum():
            break
        t = np.abs(dn).sum() / len(dn)
        # points above the baseline get weight ~0 (they are peaks, not baseline);
        # points below get an exponentially growing weight so the baseline is pulled down to them
        w = np.zeros(m)
        neg_mask = d < 0
        w[neg_mask] = np.exp(i * np.abs(d[neg_mask]) / t)
        w[0] = np.exp(i * np.abs(d).max() / t)
        w[-1] = w[0]
    return baseline


def airpls_correct(X, lam=1e5):
    Xc = np.empty_like(X)
    for i in range(X.shape[0]):
        b = airpls(X[i], lam=lam)
        Xc[i] = X[i] - b
    return Xc


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def sg_smooth(X, window=11, poly=3):
    return savgol_filter(X, window_length=window, polyorder=poly, axis=1)


def preprocess(X, airpls_lam=1e5, sg_window=11, sg_poly=3):
    """Full pipeline: airPLS -> SNV -> Savitzky-Golay. Centring and scaling are left to the estimator."""
    Xb = airpls_correct(X, lam=airpls_lam)
    Xs = snv(Xb)
    Xg = sg_smooth(Xs, window=sg_window, poly=sg_poly)
    return Xg


if __name__ == "__main__":
    import time
    from data_io import load_raw

    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    t0 = time.time()
    Xp = preprocess(X[:20])  # quick smoke test on a subset
    print("preprocessed shape:", Xp.shape, "time for 20 spectra:", round(time.time() - t0, 2), "s")
    print("any NaN:", np.isnan(Xp).any())
