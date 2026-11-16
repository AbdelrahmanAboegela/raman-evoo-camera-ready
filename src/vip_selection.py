"""
VIP (Variable Importance in Projection) variable selection: a deterministic,
seed-free alternative to GA-MW/GWO-MW's stochastic search. Fits full-spectrum
PLSR once, ranks wavelengths by their contribution to the fitted model, keeps
the top-K, refits. No population, no randomness -- if this matches or beats
GA-MW's robustness, it does so with zero run-to-run variance by construction.
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
from pls import PLSRegression

from data_io import load_raw
from preprocess import preprocess
from regimes import get_regime_indices
from plsr_baseline import fit_plsr, rmse, r2


def compute_vip(model):
    t = model.x_scores_           # (n, A)
    w = model.x_weights_          # (p, A)
    q = model.y_loadings_         # (1, A) or (A,1)
    q = np.asarray(q).reshape(-1)
    p, h = w.shape
    s = np.array([(t[:, a] @ t[:, a]) * (q[a] ** 2) for a in range(h)])  # SSY per component
    total_s = s.sum()
    vips = np.zeros(p)
    for j in range(p):
        weight = np.array([(w[j, a] / np.linalg.norm(w[:, a])) ** 2 for a in range(h)])
        vips[j] = np.sqrt(p * (weight @ s) / total_s)
    return vips


def vip_select(X, y, n_keep, max_lv=10):
    # Fit a full-spectrum reference model (max components) purely to rank variables
    ref = PLSRegression(n_components=min(max_lv, X.shape[1], X.shape[0] - 1))
    ref.fit(X, y)
    vips = compute_vip(ref)
    cols = np.argsort(vips)[::-1][:n_keep]
    return np.sort(cols), vips


if __name__ == "__main__":
    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    Xp = preprocess(X).astype(np.float32)
    idx = get_regime_indices(y, split)

    for regime in ["narrow_cal", "full_cal"]:
        cal_mask = idx[regime]
        Xc, yc = Xp[cal_mask], y[cal_mask]

        # match GA-MW's typical selected-variable count (~130-150) for a fair comparison
        cols, vips = vip_select(Xc, yc, n_keep=140)
        model, n_lv = fit_plsr(Xc[:, cols], yc, max_lv=10)

        pred_c = model.predict(Xc[:, cols]).ravel()
        print(f"\n== VIP-PLSR on {regime} (n={cal_mask.sum()}, n_lv={n_lv}, "
              f"n_vars={len(cols)}) ==")
        print(f"  RMSEC={rmse(yc, pred_c):.3f}  R2c={r2(yc, pred_c):.4f}")
        for tn in ["test_in", "test_out"]:
            tm = idx[tn]
            pred_t = model.predict(Xp[tm][:, cols]).ravel()
            yt = y[tm]
            print(f"  {tn:9s} n={tm.sum():3d}  RMSEP={rmse(yt, pred_t):.3f}  R2p={r2(yt, pred_t):.4f}")

        top_bands = wn[cols[np.argsort(vips[cols])[::-1][:10]]]
        print(f"  top-10 VIP wavenumbers: {np.round(top_bands, 1).tolist()}")
