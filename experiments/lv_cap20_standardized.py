"""Check a stronger full-spectrum baseline with 15 matched assignments per scaling.

The main 10-LV protocol is unchanged. This conservative sensitivity gives only
the full-spectrum reference a larger LV search range; it is not an all-method
20-LV benchmark. Trigger: all 15 unit-variance baseline choices hit the 10-LV cap.
"""
from threadpoolctl import threadpool_limits

from camera_ready_final import (OUT, data, sides_for, cached, save_run,
                                evaluate, fold_arrays)
from pls import set_scaling
from plsr_baseline import fit_plsr
import numpy as np
import hashlib
import json
from pathlib import Path


def main():
    addendum = dict(name="Full-spectrum LV-cap sensitivity",
        reason="All 15 unit-variance primary baseline choices reached the common 10-LV upper cap.",
        lv_range=[1,20], methods=["plsr"], scalings=["unit","mean"], seeds=list(range(15)),
        main_protocol_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "cap20_addendum.json").write_text(json.dumps(addendum, indent=2))
    X, y, split, wn, groups = data()
    c, sides = sides_for(y, split, "A_0-31")
    for mode in ("unit", "mean"):
        set_scaling(mode)
        for seed in range(15):
            key = f"cap20_reference_{mode}_plsr_s{seed:02d}"
            if cached(key):
                continue
            model, lv = fit_plsr(X[c], y[c], max_lv=20, groups=groups[c],
                                 cv_seed=seed, shuffle_groups=True)
            errors, arrays = evaluate(model, np.arange(X.shape[1]), X, y, sides)
            arrays.update(fold_arrays(X[c], y[c], groups[c], c, seed))
            save_run(key, dict(mode=mode, method="plsr", seed=seed, window="A_0-31",
                     region=False, n_vars=X.shape[1], lv=int(lv), lv_cap=20,
                     rmsep=errors, purpose="higher-cap full-spectrum sensitivity"), arrays)
            print(key, "LV", lv, "high", errors["above"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
