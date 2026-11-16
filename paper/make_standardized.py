"""Audit and generate the paper from the matched 15-seed experiment matrix.

python paper/make_standardized.py --audit-only  # validates completed checkpoints
python paper/make_standardized.py             # requires the entire matrix
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from camera_ready_final import OUT, ROOT, WINDOWS, sides_for, group_folds
from plsr_baseline import rmse
from pls import PLSRegression, set_scaling

PAPER = ROOT / "paper"
MODES = ("unit", "mean")
SEEDS = range(15)


def read(key):
    p = OUT / "runs" / (key + ".json")
    return json.loads(p.read_text()), np.load(p.with_suffix(".npz"))


def expected_keys():
    keys = []
    for seed in SEEDS:
        for mode in MODES:
            keys += [f"cap20_reference_{mode}_plsr_s{seed:02d}"]
            for algo in ("ga", "gwo"):
                keys += [f"narrow_{b}_{mode}_{algo}_s{seed:02d}" for b in ("reduced", "published")]
                keys += [f"full_reduced_{mode}_{algo}_s{seed:02d}", f"region_reduced_{mode}_{algo}_s{seed:02d}"]
            for w in [*WINDOWS, "full", "region"]:
                keys += [f"{w}_reference_{mode}_{m}_s{seed:02d}" for m in ("plsr", "vip", "ipls")]
            keys += [f"narrow_control_{mode}_{m}_s{seed:02d}" for m in
                     ("top639", "top533", "top355", "top160", "bottom160", "random_windows", "random_columns160")]
            keys += [f"refit_{mode}_{m}_s{seed:02d}" for m in ("ga", "gwo", "vip")]
    keys += [f"timing_{mode}_{m}" for mode in MODES for m in ("plsr", "vip", "ipls", "ga", "gwo")]
    assert len(keys) == 1210 and len(set(keys)) == len(keys)
    return keys


def audit(require_complete=True):
    environment = json.loads((OUT / "environment.json").read_text())
    for name, expected_hash in environment["source_sha256"].items():
        assert hashlib.sha256((ROOT / name.replace(chr(92), "/")).read_bytes()).hexdigest() == expected_hash, f"Frozen source changed: {name}"
    addendum = json.loads((OUT / "cap20_addendum.json").read_text())
    assert hashlib.sha256((ROOT / "experiments/lv_cap20_standardized.py").read_bytes()).hexdigest() == addendum["source_sha256"]
    with np.load(OUT / "data.npz") as z:
        X, y, split, wn, g = (z[k] for k in ("X", "y", "split", "wn", "groups"))
    available = {p.stem for p in (OUT / "runs").glob("*.json") if p.with_suffix(".npz").exists()}
    missing = sorted(set(expected_keys()) - available)
    unexpected = sorted(available - set(expected_keys()))
    assert not unexpected, f"Unexpected/mixed-protocol checkpoints: {unexpected[:5]}"
    if require_complete:
        assert not missing, f"Incomplete standardized matrix: {len(missing)} missing; first: {missing[:5]}"
    checked, fold_checks, coverage = 0, 0, {}
    for key in sorted(available):
        r, arrays = read(key)
        assert r["key"] == key
        assert r["mode"] in MODES and f"_{r['mode']}_" in key
        if key.startswith("timing_"):
            assert r["repeats"] == 3 and len(r["seconds"]) == 3
            assert all(np.isfinite(t) and t > 0 for t in r["seconds"])
            coverage[("timing", "A_0-31", r["mode"], r["method"], "published" if r["method"] in ("ga", "gwo") else "none")] = {0, 1, 2}
            arrays.close()
            continue
        assert key.endswith(f"_s{r['seed']:02d}")
        analysis = ("lv_cap_sensitivity" if key.startswith("cap20_") else
                    "fixed_refit" if key.startswith("refit_") else
                    "controls" if key.startswith("narrow_control_") else
                    "region" if key.startswith("region_") else
                    "full_range" if key.startswith("full_") else
                    "narrow_search" if key.startswith("narrow_") else "windows")
        cell = (analysis, r.get("window", "A_0-31"), r["mode"], r["method"], r.get("budget", "none"))
        coverage.setdefault(cell, set()).add(r["seed"])
        if key.startswith("refit_"):
            _, masks = sides_for(y, split, "A_0-31")
            for mode in MODES:
                for row in r["curves"][mode]:
                    pred = arrays[f"{mode}_lv{row['lv']}_above"]
                    assert np.isclose(rmse(y[masks["above"]], pred), row["rmsep"]["above"], atol=1e-9, rtol=0)
                    checked += 1
            arrays.close()
            continue
        seed = r["seed"]
        assert seed in SEEDS
        cal, masks = sides_for(y, split, r["window"])
        assert len(arrays["cols"]) == r["n_vars"]
        assert len(np.unique(arrays["cols"])) == r["n_vars"]
        for side, value in r["rmsep"].items():
            pred = arrays["pred_" + side]
            assert pred.shape == y[masks[side]].shape and np.isfinite(pred).all()
            assert np.isclose(rmse(y[masks[side]], pred), value, atol=1e-9, rtol=0), key
            checked += 1
        for count in ([3, 5] if "fitness" in r else [5]):
            canonical = group_folds(X[cal], y[cal], g[cal], seed, count)
            global_rows = np.flatnonzero(cal)
            for i, (tr, va) in enumerate(canonical):
                assert np.array_equal(arrays[f"fold{count}_{i}_train"], global_rows[tr]), key
                assert np.array_equal(arrays[f"fold{count}_{i}_validation"], global_rows[va]), key
                assert not set(g[global_rows[tr]]) & set(g[global_rows[va]])
                fold_checks += 1
        if "curves" in r:
            for row in r["curves"]:
                pred = arrays[f"fixed_lv{row['lv']}_above"]
                assert np.isclose(rmse(y[masks["above"]], pred), row["rmsep"]["above"], atol=1e-9, rtol=0)
                checked += 1
        arrays.close()
    coverage_rows = []
    for cell, ids in sorted(coverage.items()):
        expected_ids = set(range(3)) if cell[0] == "timing" else set(SEEDS)
        if require_complete:
            assert ids == expected_ids, (cell, sorted(ids))
        coverage_rows.append(dict(analysis=cell[0], window=cell[1], scaling=cell[2], method=cell[3],
                                 budget=cell[4], repetitions=len(ids), expected=len(expected_ids),
                                 complete=ids == expected_ids, ids=" ".join(map(str, sorted(ids)))))
    with (OUT / "coverage.csv").open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["analysis", "window", "scaling", "method", "budget", "repetitions", "expected", "complete", "ids"])
        writer.writeheader(); writer.writerows(coverage_rows)
    if require_complete:
        assert len(coverage_rows) == 90
    result = dict(completed=len(available), expected=1210, missing=len(missing),
                  prediction_vectors_checked=checked, fold_assignments_checked=fold_checks,
                  coverage_cells=len(coverage_rows), all_complete=not missing)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("AUDIT", result, flush=True)
    return X, y, split, wn, g, result


def stat(x):
    a = np.asarray(x, float)
    return dict(mean=float(a.mean()), sd=float(a.std()), min=float(a.min()), max=float(a.max()))


def pm(s):
    return f"{s['mean']:.2f} $\\pm$ {s['sd']:.2f}"


def table(name, rows):
    rows[-1] = rows[-1].rstrip().removesuffix(chr(92) * 2).rstrip()
    (PAPER / "tables" / name).write_text("\n".join(rows) + "\n", encoding="utf8")


def generate(X, y, split, wn, groups, audit_result):
    cal, masks = sides_for(y, split, "A_0-31")
    low_keep = ~np.isin(groups[masks["low"]], groups[cal])
    assert low_keep.sum() == 90 and masks["above"].sum() == 140
    assert not set(groups[cal]) & set(groups[masks["above"]])
    summary, R = {}, {}
    for mode in MODES:
        for method in ("plsr", "vip", "ipls", "ga_reduced", "gwo_reduced", "ga", "gwo"):
            budget = "reduced" if method.endswith("_reduced") else "published"
            algo = method.split("_")[0]
            keys = ([f"narrow_{budget}_{mode}_{algo}_s{s:02d}" for s in SEEDS]
                    if algo in ("ga", "gwo") else [f"A_0-31_reference_{mode}_{method}_s{s:02d}" for s in SEEDS])
            high, low, lv, nv = [], [], [], []
            for key in keys:
                r, a = read(key)
                high.append(a["pred_above"] - y[masks["above"]])
                low.append(rmse(y[masks["low"]][low_keep], a["pred_low"][low_keep]))
                lv.append(r["lv"])
                nv.append(r["n_vars"])
                a.close()
            R[mode, method] = np.vstack(high)
            summary[mode + "/" + method] = dict(n=15, high=stat(np.sqrt((R[mode, method] ** 2).mean(1))),
                    low=stat(low), lv=lv, n_vars=nv, pooled=float(np.sqrt(np.mean(R[mode, method] ** 2))))

    # All methods, including PLSR, share the master-seed draw in each bootstrap.
    # A common mixture draw preserves the pairing of every prediction comparison.
    labels = [(a, mode) for a in ("ga", "gwo") for mode in MODES] + [(a, "interaction") for a in ("ga", "gwo")]
    def contrasts(ix, seeds):
        vals = {}
        for method in ("ga", "gwo"):
            for mode in MODES:
                vals[method, mode] = float(np.sqrt(np.mean(R[mode, method][seeds][:, ix] ** 2)) -
                                           np.sqrt(np.mean(R[mode, "plsr"][seeds][:, ix] ** 2)))
            vals[method, "interaction"] = vals[method, "mean"] - vals[method, "unit"]
        return np.array([vals[label] for label in labels])
    tg = groups[masks["above"]]
    ug = np.unique(tg)
    members = [np.flatnonzero(tg == u) for u in ug]
    point = contrasts(np.arange(len(tg)), np.arange(15))
    rng = np.random.default_rng(20261116)
    samples = np.empty((10000, 6))
    for b in range(len(samples)):
        ix = np.concatenate([members[i] for i in rng.integers(0, len(ug), len(ug))])
        samples[b] = contrasts(ix, rng.integers(0, 15, 15))
    ci = np.percentile(samples, [100*.05/12, 100*(1-.05/12)], axis=0)
    effects = {a + "/" + mode: dict(diff=float(point[i]), ci=ci[:, i].tolist()) for i, (a, mode) in enumerate(labels)}
    np.savez_compressed(OUT / "bootstrap.npz", draws=samples, point=point, labels=np.asarray(["/".join(x) for x in labels]))

    fixed = {}
    for source in MODES:
        for method in ("ga", "gwo", "vip"):
            for scoring in MODES:
                fixed[source + "/" + method + "/" + scoring] = [
                    [row["rmsep"]["above"] for row in read(f"refit_{source}_{method}_s{s:02d}")[0]["curves"][scoring]]
                    for s in SEEDS]
    # The full spectrum at fixed LV has no fitted CV choice and no randomness:
    # fit this common deterministic control once rather than inventing 15 repeats.
    for mode in MODES:
        set_scaling(mode)
        curve = []
        for lv in range(1, 16):
            m = PLSRegression(n_components=lv).fit(X[cal], y[cal])
            curve.append(rmse(y[masks["above"]], m.predict(X[masks["above"]]).ravel()))
        fixed["full/" + mode] = [curve]

    controls = {}
    for mode in MODES:
        for method in ("top639", "top533", "top355", "top160", "bottom160", "random_windows", "random_columns160"):
            rows = [read(f"narrow_control_{mode}_{method}_s{s:02d}")[0] for s in SEEDS]
            controls[mode + "/" + method] = dict(n=15, high=stat([r["rmsep"]["above"] for r in rows]),
                lv=[r["lv"] for r in rows], curves=[[r["rmsep"]["above"] for r in row["curves"]] for row in rows])

    # Main table: exactly 15 matched pipeline runs in every row.
    rows = []
    for mode, label in (("unit", "Unit var."), ("mean", "Mean-cent.")):
        for method, name in (("plsr", "PLSR"), ("vip", "VIP-PLSR"), ("ipls", "iPLS"),
                             ("ga_reduced", "GA-MW"), ("gwo_reduced", "GWO-MW"), ("ga", "GA-MW"), ("gwo", "GWO-MW")):
            r = summary[mode + "/" + method]
            budget = "Reduced" if method.endswith("_reduced") else ("Published" if method in ("ga", "gwo") else "---")
            lv = str(r["lv"][0]) if min(r["lv"]) == max(r["lv"]) else f"{min(r['lv'])}--{max(r['lv'])}"
            nv = f"{np.mean(r['n_vars']):.0f}"
            effect = "---"
            if method in ("ga", "gwo"):
                e = effects[method + "/" + mode]
                effect = f"${e['diff']:+.2f}$ [${e['ci'][0]:+.2f}$, ${e['ci'][1]:+.2f}$]"
            rows.append(f"{label} & {name} & {budget} & {nv} & {lv} & {pm(r['low'])} & {pm(r['high'])} & {effect} " + chr(92)*2)
        if mode == "unit": rows.append("\\midrule")
    table("final_main.tex", rows)

    rows, windows = [], []
    for window, levels in WINDOWS.items():
        wc, wm = sides_for(y, split, window)
        for side in ("above", "below"):
            if side not in wm: continue
            cells, values = [], {}
            for mode in MODES:
                block = {m: stat([read(f"{window}_reference_{mode}_{m}_s{s:02d}")[0]["rmsep"][side] for s in SEEDS])
                         for m in ("plsr", "vip", "ipls")}
                values[mode] = block
                best = min(v["mean"] for v in block.values())
                cells += [pm(v) if v["mean"] != best else f"\\textbf{{{v['mean']:.2f}}} $\\pm$ {v['sd']:.2f}" for v in block.values()]
            rows.append(f"{min(levels)}--{max(levels)} & {int(wc.sum())} & {side} & {int(wm[side].sum())} & " + " & ".join(cells) + " " + chr(92)*2)
            windows.append(dict(window=window, side=side, n_test=int(wm[side].sum()), values=values))
    table("final_windows.tex", rows)

    # The leakage reference is rerun under the same fifteen shuffled grouped
    # assignments; neither its five-run VIP nor its row-wise legacy scores survive.
    fc, fm = sides_for(y, split, "full")
    strict = {side: ~np.isin(groups[mask], groups[fc]) for side, mask in fm.items()}
    assert strict["low"].sum() == 90 and strict["above"].sum() == 110
    leakage, rows = {}, []
    for mode, label in (("unit", "Unit var."), ("mean", "Mean-cent.")):
        for method, name in (("plsr", "PLSR"), ("vip", "VIP-PLSR"), ("ipls", "iPLS"), ("ga", "GA-MW"), ("gwo", "GWO-MW")):
            vals = {side + suffix: [] for side in ("low", "above") for suffix in ("_all", "_strict")}
            for s in SEEDS:
                key = f"full_reduced_{mode}_{method}_s{s:02d}" if method in ("ga", "gwo") else f"full_reference_{mode}_{method}_s{s:02d}"
                r, z = read(key)
                for side in ("low", "above"):
                    pred, target = z["pred_" + side], y[fm[side]]
                    vals[side + "_all"].append(rmse(target, pred))
                    vals[side + "_strict"].append(rmse(target[strict[side]], pred[strict[side]]))
                z.close()
            d = {k: stat(v) for k, v in vals.items()}
            leakage[mode + "/" + method] = d
            rows.append(f"{label} & {name} & " + " & ".join(pm(d[k]) for k in ("low_all", "low_strict", "above_all", "above_strict")) + " " + chr(92)*2)
        if mode == "unit": rows.append("\\midrule")
    table("final_leakage.tex", rows)

    region = {}
    for mode in MODES:
        for method in ("plsr", "vip", "ipls", "ga", "gwo"):
            keys = [f"region_reduced_{mode}_{method}_s{s:02d}" if method in ("ga", "gwo") else
                    f"region_reference_{mode}_{method}_s{s:02d}" for s in SEEDS]
            region[mode + "/" + method] = stat([read(k)[0]["rmsep"]["above"] for k in keys])
    cap20 = {}
    for mode in MODES:
        rows = [read(f"cap20_reference_{mode}_plsr_s{s:02d}")[0] for s in SEEDS]
        cap20[mode] = dict(n=15, high=stat([r["rmsep"]["above"] for r in rows]), lv=[r["lv"] for r in rows])
    timing = json.loads((OUT / "timing.json").read_text())
    assert all(r["repeats"] == 3 for r in timing.values())
    macros = {}
    for mode, cap in (("unit", "Unit"), ("mean", "Mean")):
        for method, label in (("plsr", "PLSR"), ("vip", "VIP"), ("ipls", "IPLS"), ("ga", "GA"), ("gwo", "GWO"),
                              ("ga_reduced", "GARed"), ("gwo_reduced", "GWORed")):
            r = summary[mode + "/" + method]
            macros[cap + label] = f"{r['high']['mean']:.2f}"
            macros[cap + label + "SD"] = f"{r['high']['sd']:.2f}"
            if method in ("plsr", "vip", "ipls", "ga", "gwo"):
                macros[cap + label + "Time"] = f"{timing[f'timing_{mode}_{method}']['median']:.1f}"
                macros["Region" + cap + label] = f"{region[mode + '/' + method]['mean']:.2f}"
        macros[cap + "CapTwenty"] = f"{cap20[mode]['high']['mean']:.2f}"
        macros[cap + "CapTwentySD"] = f"{cap20[mode]['high']['sd']:.2f}"
        macros[cap + "CapTwentyLV"] = f"{min(cap20[mode]['lv'])}--{max(cap20[mode]['lv'])}"
        ks = summary[mode + "/vip"]["n_vars"]
        macros[cap + "VIPK"] = str(min(ks)) if min(ks) == max(ks) else f"{min(ks)}--{max(ks)}"
        macros[cap + "GAFixedTen"] = f"{np.mean(fixed['unit/ga/' + mode], axis=0)[9]:.2f}"
        macros[cap + "FullFixedTen"] = f"{fixed['full/' + mode][0][9]:.2f}"
        macros[cap + "VarianceHalf"] = f"{controls[mode + '/top355']['high']['mean']:.2f}"
        macros[cap + "VarianceHalfFixedTen"] = f"{np.mean(controls[mode + '/top355']['curves'], axis=0)[9]:.2f}"
        for a, label in (("ga", "GA"), ("gwo", "GWO")):
            e = effects[a + "/" + mode]
            macros[cap + label + "Delta"] = f"{e['diff']:+.2f}"
            macros[cap + label + "CI"] = f"[{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]"
        macros[cap + "Random"] = f"{controls[mode + '/random_windows']['high']['mean']:.2f}"
        macros[cap + "RandomSD"] = f"{controls[mode + '/random_windows']['high']['sd']:.2f}"
        macros[cap + "ResidualR"] = f"{np.corrcoef(R[mode, 'ga'].mean(0), R[mode, 'vip'].mean(0))[0,1]:.2f}"
    for algo, label in (("ga", "GA"), ("gwo", "GWO")):
        e = effects[algo + "/interaction"]
        macros[label + "Interaction"] = f"{e['diff']:+.2f}"
        macros[label + "InteractionCI"] = f"[{e['ci'][0]:+.2f},{e['ci'][1]:+.2f}]"
    (PAPER / "final_numbers.tex").write_text("% Generated from the complete standardized 15-seed matrix.\n" +
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k,v in macros.items()) + "\n", encoding="utf8")

    total = X[cal].astype(float).var(0, ddof=1)
    gc = groups[cal]
    Xc = X[cal].astype(float)
    within = sum(((Xc[gc == group] - Xc[gc == group].mean(0)) ** 2).sum(0) for group in np.unique(gc)) / (len(gc) - len(np.unique(gc)))
    repeatability = within / np.maximum(total, np.finfo(float).tiny)
    result = dict(main=summary, effects=effects, windows=windows, leakage=leakage, region=region,
                  controls=controls, fixed=fixed, cap20=cap20, audit=audit_result, timing=timing,
                  low_n=90, high_n=140, repetitions=15, timing_repeats=3)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    np.savez_compressed(OUT / "repeatability.npz", total=total, within=within, ratio=repeatability)
    figures(fixed, controls, summary, total, repeatability)
    print("FINAL", json.dumps(effects, indent=2), flush=True)


def figures(fixed, controls, summary, total, ratio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), sharey=True)
    styles = [("full", "Full spectrum", "#333333"), ("unit/vip", "VIP (UV-selected)", "#6a3d9a"),
              ("unit/ga", "GA (UV-selected)", "#0072b2"), ("mean/ga", "GA (MC-selected)", "#d55e00")]
    for ax, mode in zip(axes, MODES):
        for source, label, color in styles:
            vals = np.array(fixed[source + "/" + mode])
            ax.plot(range(1, 16), vals.mean(0), color=color, label=label, lw=1.5)
        ax.set_title("(a) Unit variance" if mode == "unit" else "(b) Mean centering")
        ax.set_xlabel("Fixed latent variables")
        ax.set_xticks([1, 5, 10, 15]); ax.grid(alpha=.15)
    axes[0].set_ylabel(r"RMSEP$_{\mathrm{high}}$ (EVOO %)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=.5)
    for ext in ("pdf", "png"): fig.savefig(PAPER / "figs" / ("fig_final_fixed." + ext), dpi=180)
    plt.close(fig)

    quartiles = np.array_split(np.argsort(total), 4)
    fig, (a,b) = plt.subplots(1, 2, figsize=(7, 2.55))
    a.boxplot([ratio[q] for q in quartiles], tick_labels=["Q1", "Q2", "Q3", "Q4"],
              showfliers=False, medianprops=dict(color="#0072b2"))
    a.set_xlabel("Calibration-variance quartile (low to high)")
    a.set_ylabel("Within-group / total variance"); a.set_title("(a) Channel repeatability")
    for mode, label, color in (("unit", "Unit variance", "#0072b2"), ("mean", "Mean centered", "#d55e00")):
        methods = ["top160", "top355", "top533", "top639"]
        counts = [160,355,533,639,710]
        vals = [controls[mode + "/" + m]["high"]["mean"] for m in methods] + [summary[mode + "/plsr"]["high"]["mean"]]
        sd = [controls[mode + "/" + m]["high"]["sd"] for m in methods] + [summary[mode + "/plsr"]["high"]["sd"]]
        fixed10 = [np.mean(controls[mode + "/" + m]["curves"], axis=0)[9] for m in methods] + [fixed["full/" + mode][0][9]]
        b.errorbar(counts, vals, yerr=sd, fmt="o-", color=color, label=label + ": LV by CV", ms=3, capsize=2, lw=1)
        b.plot(counts, fixed10, "--", color=color, label=label + ": 10 LV", lw=1)
    b.set_xlabel("Highest-variance channels retained")
    b.set_ylabel(r"RMSEP$_{\mathrm{high}}$ (EVOO %)")
    b.set_title("(b) Calibration-derived filters"); b.legend(frameon=False, fontsize=6.5); b.grid(alpha=.15)
    fig.tight_layout(pad=.5)
    for ext in ("pdf", "png"): fig.savefig(PAPER / "figs" / ("fig_final_variance." + ext), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--audit-only", action="store_true")
    parsed = args.parse_args()
    with threadpool_limits(limits=1):
        inputs = audit(require_complete=not parsed.audit_only)
        if not parsed.audit_only: generate(*inputs)
