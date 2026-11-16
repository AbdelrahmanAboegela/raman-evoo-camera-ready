"""
Load and parse the Picouet (2023) Raman_Oils_raw.csv dataset.

File format quirks (verified by inspection):
  - ';'-delimited, decimal commas (French locale export)
  - header: 9 metadata columns then one column per wavenumber (cm-1)
  - metadata columns: Date, Obs/day, Spectrum#, EVOO(%), Sunflower(%),
    Tps(ms), Scans, P(mW), Type (CALIBRATION/VALIDATION/PREDICTION)
"""
from paths import DATA_CSV, RESULTS, FIGS  # noqa: E402
import numpy as np
import pandas as pd

N_META_COLS = 9


def load_raw(csv_path):
    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="latin-1")
    meta_cols = list(df.columns[:N_META_COLS])
    wn_cols = list(df.columns[N_META_COLS:])
    wavenumbers = np.array([float(c.replace(",", ".")) for c in wn_cols])

    meta = df[meta_cols].copy()
    meta.columns = [
        "date", "obs_per_day", "spectrum_num", "evoo_pct", "sunflower_pct",
        "acq_time_ms", "scans", "power_mw", "split",
    ]
    meta["split"] = meta["split"].str.strip().str.upper()
    for col in ("evoo_pct", "sunflower_pct"):
        meta[col] = (
            meta[col].astype(str).str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False).astype(float)
        )
    X = df[wn_cols].to_numpy(dtype=float)
    y = meta["evoo_pct"].to_numpy(dtype=float)
    split = meta["split"].to_numpy()
    return X, y, split, wavenumbers, meta


if __name__ == "__main__":
    X, y, split, wn, meta = load_raw(str(DATA_CSV))
    print("X shape:", X.shape)
    print("wavenumber range:", wn.min(), "-", wn.max(), "cm-1, n=", len(wn))
    print("y range:", y.min(), "-", y.max())
    print("split counts:\n", pd.Series(split).value_counts())
    print("unique EVOO% levels:", sorted(set(y.tolist())))
