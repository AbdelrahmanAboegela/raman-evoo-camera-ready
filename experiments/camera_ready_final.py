"""Frozen grouped camera-ready analyses; per-run checkpoints enable resuming.

python -u experiments/camera_ready_final.py --stage search --jobs 4
python -u experiments/camera_ready_final.py --stage references --jobs 1
python -u experiments/camera_ready_final.py --stage region --jobs 4
python -u experiments/camera_ready_final.py --stage refit --jobs 4
python -u experiments/camera_ready_final.py --stage timing --jobs 1
All outputs are separate from the original and exploratory result files.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paths import ROOT, DATA_CSV, RESULTS
from data_io import load_raw
from preprocess import preprocess
from groups import get_groups
from pls import PLSRegression, set_scaling
from plsr_baseline import fit_plsr, rmse
from vip_selection import vip_select
from vip_cv_select import GRID, fit_final_vip_plsr
from selectors_extra import ipls_select
from ga_mw import ga_mw_select, _windows_to_columns
from gwo_mw import gwo_mw_select
from camera_ready_diagnostics import WINDOWS, subsets, filter_cols

OUT = RESULTS / "camera_ready_standardized"
WIDE = GRID + [275, 300, 350, 400, 450, 500, 550, 600, 650, 710]
BUDGETS = {"reduced": {"ga": (40, 40), "gwo": (20, 40)},
           "published": {"ga": (100, 1000), "gwo": (20, 1000)}}


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")
    tmp.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    protocol = dict(version=2, study="exploratory controlled follow-up; same fixed dataset",
                    data_sha256=hashlib.sha256(DATA_CSV.read_bytes()).hexdigest(),
                    windows=WINDOWS, gap=19, seeds=list(range(15)), budgets=BUDGETS,
                    scaling=["unit", "mean"], lv_cap=10, fixed_lvs=list(range(1, 16)),
                    cv="15 matched shuffled GroupKFold assignments, random_state=master seed; search fitness 3 folds/5 LV; final LV and all VIP/iPLS selection 5 folds",
                    vip_grid=WIDE, ipls_intervals=20,
                    region=[1000, 1800], region_search_seeds=list(range(15)),
                    timing_repeats=3, timing_seed=0,
                    random_controls="15 master IDs; SeedSequence([20261116,seed,kind]) with kind 0=windows,1=columns160; matched grouped CV seed",
                    variance_filters=["full", "top639", "top533", "top355", "top160", "bottom160"],
                    bootstrap=dict(replicates=10000, family="GA/GWO vs PLSR at each scaling and two interactions", size=6),
                    python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                    sklearn=sklearn.__version__, threads_per_worker=1)
    protocol = json.loads(json.dumps(protocol))  # normalize tuple-valued budgets
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Frozen protocol differs"
    else:
        atomic_json(path, protocol)
    if not (OUT / "data.npz").exists():
        with threadpool_limits(limits=1):
            X, y, split, wn, meta = load_raw(str(DATA_CSV))
            Xp = preprocess(X).astype(np.float32)
        np.savez_compressed(OUT / "data.npz", X=Xp, y=y, split=np.asarray(split, dtype=str),
                            wn=wn, groups=get_groups(meta))
    if not (OUT / "environment.json").exists():
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor).Name"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        atomic_json(OUT / "environment.json", dict(cpu=cpu, os=platform.platform(),
                    recorded=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list((ROOT / "src").glob("*.py")) + [Path(__file__)]}))


def data(region=False):
    name = "region_data.npz" if region else "data.npz"
    with np.load(OUT / name) as z:
        return tuple(z[k] for k in ("X", "y", "split", "wn", "groups"))


def sides_for(y, split, window):
    if window == "full":
        test = np.isin(split, ["VALIDATION", "PREDICTION"])
        return split == "CALIBRATION", {"above": test & (y >= 50), "low": test & (y <= 35)}
    cal, sides = subsets(y, split, window)
    test = np.isin(split, ["VALIDATION", "PREDICTION"])
    sides["inside"] = test & (y >= min(WINDOWS[window])) & (y <= max(WINDOWS[window]))
    if window == "A_0-31":
        sides["low"] = test & (y <= 35)
    return cal, sides


def group_folds(X, y, groups, seed, n_splits=5):
    fs = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups))
    assert all(not set(groups[tr]) & set(groups[va]) for tr, va in fs)
    return fs


def fold_arrays(Xc, yc, groups, cal_mask, seed, search=False):
    rows = np.flatnonzero(cal_mask)
    out = {}
    for count in ([3, 5] if search else [5]):
        for i, (tr, va) in enumerate(group_folds(Xc, yc, groups, seed, count)):
            out[f"fold{count}_{i}_train"] = rows[tr]
            out[f"fold{count}_{i}_validation"] = rows[va]
    return out


def save_run(key, record, arrays):
    arrays = {**arrays}
    path = OUT / "runs" / key
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    atomic_json(path.with_suffix(".json"), dict(key=key, **record))
    return dict(key=key, **record)


def cached(key):
    path = OUT / "runs" / (key + ".json")
    if path.exists() and path.with_suffix(".npz").exists():
        return json.loads(path.read_text())


def evaluate(model, cols, X, y, sides):
    arrays = {"cols": cols}
    scores = {}
    for side, mk in sides.items():
        pred = model.predict(X[mk][:, cols]).ravel()
        assert np.isfinite(pred).all()
        scores[side] = rmse(y[mk], pred)
        arrays["pred_" + side] = pred
    return scores, arrays


def search(mode, method, seed, budget, region=False, window="A_0-31"):
    label = "region" if region else ("full" if window == "full" else "narrow")
    key = f"{label}_{budget}_{mode}_{method}_s{seed:02d}"
    if (r := cached(key)) is not None:
        return r
    with threadpool_limits(limits=1):
        set_scaling(mode)
        X, y, split, wn, groups = data(region)
        c, sides = sides_for(y, split, window)
        pop, iters = BUDGETS[budget][method]
        t0 = time.perf_counter()
        cols, positions, history = (ga_mw_select if method == "ga" else gwo_mw_select)(
            X[c], y[c], seed=seed, groups=groups[c], pop_size=pop, n_iters=iters, shuffle_groups=True)
        ts = time.perf_counter() - t0
        model, lv = fit_plsr(X[c][:, cols], y[c], groups=groups[c], cv_seed=seed, shuffle_groups=True)
        errors, arrays = evaluate(model, cols, X, y, sides)
        arrays.update(positions=positions, fitness_history=np.asarray(history))
        arrays.update(fold_arrays(X[c], y[c], groups[c], c, seed, search=True))
        return save_run(key, dict(mode=mode, method=method, seed=seed, budget=budget,
                        region=region, window=window, n_vars=int(len(cols)), lv=int(lv),
                        rmsep=errors, search_seconds=ts, total_seconds=time.perf_counter() - t0,
                        fitness=float(history[-1])), arrays)


def vip_fit(Xc, yc, groups, seed=0):
    grid = sorted(set(min(k, Xc.shape[1]) for k in WIDE))
    fs = group_folds(Xc, yc, groups, seed)
    ranks = []
    for tr, va in fs:
        assert not set(groups[tr]) & set(groups[va])
        _, vips = vip_select(Xc[tr], yc[tr], n_keep=Xc.shape[1])
        ranks.append(np.argsort(vips)[::-1])
    scores = {}
    for k in grid:
        errs = []
        for (tr, va), rank in zip(fs, ranks):
            cols = np.sort(rank[:k])
            model, _ = fit_plsr(Xc[tr][:, cols], yc[tr], groups=groups[tr],
                                cv_seed=seed, shuffle_groups=True)
            errs.append(rmse(yc[va], model.predict(Xc[va][:, cols]).ravel()))
        scores[k] = float(np.mean(errs))
    k = min(scores, key=scores.get)
    model, cols, _, lv = fit_final_vip_plsr(Xc, yc, k, groups=groups, cv_seed=seed, shuffle_groups=True)
    return model, cols, lv, dict(k=int(k), grid=grid, cv_scores=scores)


def reference(window, mode, method, region=False, seed=0):
    key = f"{'region' if region else window}_reference_{mode}_{method}_s{seed:02d}"
    if (r := cached(key)) is not None:
        return r
    with threadpool_limits(limits=1):
        set_scaling(mode)
        X, y, split, wn, groups = data(region)
        c, sides = sides_for(y, split, window)
        Xc, yc, gc = X[c], y[c], groups[c]
        t0 = time.perf_counter()
        extra = {}
        if method == "vip":
            model, cols, lv, extra = vip_fit(Xc, yc, gc, seed)
        else:
            cols = (ipls_select(Xc, yc, groups=gc, cv_seed=seed, shuffle_groups=True) if method == "ipls"
                    else np.arange(X.shape[1]))
            model, lv = fit_plsr(Xc[:, cols], yc, groups=gc, cv_seed=seed, shuffle_groups=True)
        errors, arrays = evaluate(model, cols, X, y, sides)
        arrays.update(fold_arrays(Xc, yc, gc, c, seed))
        return save_run(key, dict(window=window, mode=mode, method=method, region=region,
                        seed=seed, n_vars=int(len(cols)), lv=int(lv), rmsep=errors,
                        total_seconds=time.perf_counter() - t0, **extra), arrays)


def run_parallel(fn, tasks, jobs):
    for i, r in enumerate(Parallel(n_jobs=jobs, return_as="generator_unordered")(
            delayed(fn)(*task) for task in tasks), 1):
        print(f"{i}/{len(tasks)} {r['key']} high={r.get('rmsep', {}).get('above', '-')}", flush=True)


def control(mode, method, seed):
    key = f"narrow_control_{mode}_{method}_s{seed:02d}"
    if (r := cached(key)) is not None:
        return r
    with threadpool_limits(limits=1):
        set_scaling(mode)
        X, y, split, wn, groups = data()
        c, sides = sides_for(y, split, "A_0-31")
        Xc, yc, gc = X[c], y[c], groups[c]
        random = method.startswith("random_")
        if random:
            kind = 0 if method == "random_windows" else 1
            rng = np.random.default_rng(np.random.SeedSequence([20261116, seed, kind]))
            cols = (_windows_to_columns(rng.integers(0, X.shape[1] - 4, 30), X.shape[1])
                    if kind == 0 else np.sort(rng.choice(X.shape[1], 160, replace=False)))
        else:
            cols = filter_cols(Xc, method)
        fs = group_folds(Xc, yc, gc, seed)
        fold_cols = [cols if random else filter_cols(Xc[tr], method) for tr, va in fs]
        scores = []
        for lv in range(1, 11):
            errors = []
            for (tr, va), selected in zip(fs, fold_cols):
                m = PLSRegression(n_components=lv).fit(Xc[tr][:, selected], yc[tr])
                errors.append(rmse(yc[va], m.predict(Xc[va][:, selected]).ravel()))
            scores.append(float(np.mean(errors)))
        lv = int(np.argmin(scores)) + 1
        model = PLSRegression(n_components=lv).fit(Xc[:, cols], yc)
        errors, arrays = evaluate(model, cols, X, y, sides)
        curves = []
        for fixed_lv in range(1, 16):
            m = PLSRegression(n_components=fixed_lv).fit(Xc[:, cols], yc)
            e, predictions = evaluate(m, cols, X, y, sides)
            curves.append(dict(lv=fixed_lv, rmsep=e))
            arrays[f"fixed_lv{fixed_lv}_above"] = predictions["pred_above"]
        arrays.update(fold_arrays(Xc, yc, gc, c, seed))
        return save_run(key, dict(mode=mode, method=method, seed=seed, window="A_0-31",
                        region=False, n_vars=int(len(cols)), lv=lv, rmsep=errors,
                        cv_scores=scores, curves=curves), arrays)


def refit(mode, method, seed):
    key = f"refit_{mode}_{method}_s{seed:02d}"
    if (r := cached(key)) is not None:
        return r
    source = (f"narrow_published_{mode}_{method}_s{seed:02d}" if method in ("ga", "gwo")
              else f"A_0-31_reference_{mode}_{method}_s{seed:02d}")
    assert cached(source), source
    with threadpool_limits(limits=1):
        X, y, split, wn, groups = data()
        c, sides = sides_for(y, split, "A_0-31")
        cols = np.load(OUT / "runs" / (source + ".npz"))["cols"]
        curves, arrays = {}, {"cols": cols}
        for scoring in ("unit", "mean"):
            set_scaling(scoring)
            rows = []
            for lv in range(1, 16):
                m = PLSRegression(n_components=lv).fit(X[c][:, cols], y[c])
                errors, predictions = evaluate(m, cols, X, y, sides)
                rows.append(dict(lv=lv, rmsep=errors))
                for side in ("above", "low"):
                    arrays[f"{scoring}_lv{lv}_{side}"] = predictions["pred_" + side]
            curves[scoring] = rows
        return save_run(key, dict(mode=mode, method=method, seed=seed, curves=curves), arrays)


def timing():
    records = {}
    for mode in ("unit", "mean"):
        for method in ("plsr", "vip", "ipls", "ga", "gwo"):
            key = f"timing_{mode}_{method}"
            if (r := cached(key)) is not None:
                records[key] = r
                continue
            values = []
            repeats = 3
            with threadpool_limits(limits=1):
                set_scaling(mode)
                X, y, split, wn, groups = data()
                c, _ = sides_for(y, split, "A_0-31")
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    if method == "vip":
                        model, cols, lv, _ = vip_fit(X[c], y[c], groups[c], seed=0)
                    else:
                        if method in ("ga", "gwo"):
                            pop, iters = BUDGETS["published"][method]
                            cols, _, _ = (ga_mw_select if method == "ga" else gwo_mw_select)(
                                X[c], y[c], groups=groups[c], seed=0, pop_size=pop, n_iters=iters, shuffle_groups=True)
                        elif method == "ipls":
                            cols = ipls_select(X[c], y[c], groups=groups[c], cv_seed=0, shuffle_groups=True)
                        else:
                            cols = np.arange(X.shape[1])
                        model, lv = fit_plsr(X[c][:, cols], y[c], groups=groups[c], cv_seed=0, shuffle_groups=True)
                    values.append(time.perf_counter() - t0)
                # Verify standalone timing computes the same deployed model.
                refkey = (f"narrow_published_{mode}_{method}_s00" if method in ("ga", "gwo")
                          else f"A_0-31_reference_{mode}_{method}_s00")
                ref = cached(refkey)
                assert ref and lv == ref["lv"]
                assert np.array_equal(cols, np.load(OUT / "runs" / (refkey + ".npz"))["cols"])
                records[key] = save_run(key, dict(mode=mode, method=method, seconds=values,
                            median=float(np.median(values)), repeats=repeats, lv=int(lv)), {"cols": cols})
                print(key, records[key]["median"], flush=True)
    atomic_json(OUT / "timing.json", records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["search", "references", "region", "full", "controls", "refit", "timing"], required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    prepare()
    start = time.perf_counter()
    if args.stage == "search":
        for budget in ("reduced", "published"):
            tasks = [(mode, method, seed, budget) for seed in range(15)
                     for mode in ("unit", "mean") for method in ("ga", "gwo")]
            run_parallel(search, tasks, args.jobs)
    elif args.stage == "references":
        run_parallel(reference, [(w, m, a, False, s) for s in range(15) for w in WINDOWS
                                for m in ("unit", "mean") for a in ("plsr", "vip", "ipls")], args.jobs)
    elif args.stage == "region":
        if not (OUT / "region_data.npz").exists():
            with threadpool_limits(limits=1):
                X, y, split, wn, meta = load_raw(str(DATA_CSV))
                keep = (wn >= 1000) & (wn <= 1800)
                Xp = preprocess(X[:, keep]).astype(np.float32)
            np.savez_compressed(OUT / "region_data.npz", X=Xp, y=y,
                                split=np.asarray(split, dtype=str), wn=wn[keep], groups=get_groups(meta))
        run_parallel(reference, [("A_0-31", m, a, True, s) for s in range(15) for m in ("unit", "mean")
                                 for a in ("plsr", "vip", "ipls")], args.jobs)
        run_parallel(search, [(m, a, s, "reduced", True) for s in range(15)
                              for m in ("unit", "mean") for a in ("ga", "gwo")], args.jobs)
    elif args.stage == "full":
        run_parallel(reference, [("full", m, a, False, s) for s in range(15)
                                 for m in ("unit", "mean") for a in ("plsr", "vip", "ipls")], args.jobs)
        run_parallel(search, [(m, a, s, "reduced", False, "full") for s in range(15)
                              for m in ("unit", "mean") for a in ("ga", "gwo")], args.jobs)
    elif args.stage == "controls":
        run_parallel(control, [(m, a, s) for s in range(15) for m in ("unit", "mean")
                              for a in ("top639", "top533", "top355", "top160", "bottom160", "random_windows", "random_columns160")], args.jobs)
    elif args.stage == "refit":
        run_parallel(refit, [(m, a, s) for m in ("unit", "mean") for a in ("ga", "gwo", "vip")
                             for s in range(15)], args.jobs)
    else:
        timing()
    print(f"DONE {args.stage} elapsed={time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
