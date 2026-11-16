"""
Calibration-regime slicing for the range-generalization study.

NARROW regime mimics Zhu et al. (2023)'s adulteration-screening design
(discrete levels within their published 5-30% window); FULL regime uses
all available CALIBRATION rows (0-100%). Both are evaluated on the same
held-out VALIDATION+PREDICTION spectra, split into in-range / out-of-range.
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np

NARROW_LEVELS = [0.0, 10.0, 15.0, 23.0, 25.0, 31.0]
IN_RANGE_MAX = 35.0
OUT_RANGE_MIN = 50.0


def get_regime_indices(y, split):
    cal_mask = split == "CALIBRATION"
    test_mask = (split == "VALIDATION") | (split == "PREDICTION")

    narrow_mask = cal_mask & np.isin(y, NARROW_LEVELS)
    full_mask = cal_mask

    test_in_mask = test_mask & (y <= IN_RANGE_MAX)
    test_out_mask = test_mask & (y >= OUT_RANGE_MIN)

    return {
        "narrow_cal": narrow_mask,
        "full_cal": full_mask,
        "test_in": test_in_mask,
        "test_out": test_out_mask,
    }


if __name__ == "__main__":
    from data_io import load_raw

    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    idx = get_regime_indices(y, split)
    for k, m in idx.items():
        print(f"{k:12s} n={m.sum():4d}  EVOO% range=[{y[m].min() if m.any() else '-'}, {y[m].max() if m.any() else '-'}]")
