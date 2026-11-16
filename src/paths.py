"""Repo-relative paths, so no script depends on where the repo is checked out."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_CSV = DATA / "Raman_Oils_raw.csv"
RESULTS = ROOT / "results"
FIGS = ROOT / "paper" / "figs"
