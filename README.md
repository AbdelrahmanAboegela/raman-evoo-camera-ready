# Beyond the Calibration Range: Mechanisms of Variable-Selection Robustness in Raman Quantification of Olive Oil Adulteration

Final code, numerical evidence and manuscript for the JAC-ECC 2026 paper.

Abdelrahman A. Alshames, Omnia S. Eshra, Zeina Eltannikhy, Moaz H. El Garawany,
and Alaa A. Sanhoury. Faculty of Computer Science and Information Technology,
Egypt-Japan University of Science and Technology (E-JUST), Alexandria, Egypt.

We examine why spectral selection helps Raman PLSR outside its calibration range.
The benefit depends on channel scaling, model complexity and concentration coverage.
The study uses one public EVOO/sunflower dataset; it does not establish transfer to
independent instruments or adulterants.

## Paper and evidence

- [Final paper](paper/paper.pdf)
- [Exact run policy](docs/standardized_protocol.md)
- [Reviewer-response and scientific interpretation](docs/reviewer_response.md)
- [Numerical summary](results/camera_ready_standardized/summary.json)
- [Audit](results/camera_ready_standardized/audit.json) and [coverage matrix](results/camera_ready_standardized/coverage.csv)
- [All per-run predictions and CV indices](results/camera_ready_standardized/checkpoints.zip)

Every fitted condition uses the same **15 master seed IDs (0--14)**, including
deterministic selectors through shuffled grouped CV. Every timing condition uses
**three complete repetitions**, measured separately at seed 0. Static variance
summaries and full-spectrum fixed-LV curves are common deterministic controls.
The 1,210-record matrix passes 8,760 prediction-vector and 6,270 fold checks.

| High-concentration RMSEP, mean +/- population SD | Unit variance | Mean centering |
|---|---:|---:|
| Full-spectrum PLSR | 4.42 +/- 0.00 | 3.09 +/- 0.80 |
| VIP-PLSR | 1.67 +/- 0.36 | 3.10 +/- 0.64 |
| GA-MW, published budget | 1.87 +/- 0.55 | 3.36 +/- 0.73 |
| GWO-MW, published budget | 2.30 +/- 0.84 | 3.67 +/- 1.05 |

Errors are EVOO percentage points. Adjusted interactions support attenuation of
selection benefit under mean centering; the within-scaling intervals do not establish
mean-centered harm. The paper gives paired group/seed bootstrap intervals and controls.

## Validate the released results

Tested with Python 3.11.5 on Windows and the versions pinned in requirements.txt.
For PDF compilation, install a LaTeX distribution with IEEEtran, stfloats and placeins.

```powershell
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py all
```

`check` expands the compact checkpoint archive into Git-ignored files, verifies exact
computational fingerprints, tests grouped fitting and the first-component identity,
then recomputes every saved prediction error and checks the stored folds.
`all` additionally regenerates the numerical tables/figures, compiles LaTeX twice,
checks the PDF, and creates the PDF, LaTeX source and complete research-code ZIPs
with a checksum manifest in `dist/`. Raw data are not needed to audit the archived
predictions or rebuild the manuscript from the included derived caches.

Individual steps are `python reproduce.py figures`, `python reproduce.py build`
and `python reproduce.py package`. These commands do not need ancestor Git commits.

## Independent numerical rerun

In a **fresh clone before expanding the archived checkpoints**, run:

```powershell
python reproduce.py rerun --jobs 4
python reproduce.py build
```

This downloads and hash-verifies the public raw CSV, runs the complete standardized
search/reference/window/region/control/refit matrix, then measures all timings
sequentially. Run on an otherwise idle machine for timings. The published optimizer
budgets make a complete rerun take hours. The original measurement environment is
recorded in `environment.json`; a rerun records its own machine information separately.
The frozen protocol requires the pinned Python/library versions. Hardware-dependent
timings can differ. `rerun` refuses to silently reuse restored checkpoints as new runs.

## Repository layout

| Directory | Contents |
|---|---|
| `src/` | Executed numerical library, preserved byte for byte |
| `experiments/` | Standardized runner, fixed shared helpers, baseline sensitivity and checks |
| `paper/` | Final LaTeX, generated numbers, three tables and two quantitative figures |
| `results/camera_ready_standardized/` | Protocol, execution metadata, caches, summary and compact checkpoint archive |
| `docs/` | Run policy and reviewer-response map |
| `data/` | Dataset attribution and download instructions |

This repository is one release snapshot. It retains the numerical provenance needed
to inspect the findings; exploratory outputs, build logs and previous Git history
are kept outside the release. Original execution identifiers in metadata describe
the measurements and do not require access to the previous repository.

## Data and citation

P. Picouet, *Raman spectra of olive and sunflower oil mixtures (Raman_GRAPPE_Oil1)*,
GRAPPE, ESA/INRAE, Licence Ouverte 2.0, last modified 28 April 2023.
[Dataset and provider](https://www.data.gouv.fr/datasets/spectres-raman-de-melanges-dhuiles-dolive-et-de-tournesol-raman-spectra-of-olive-and-sunflower-oil-mixtures).
See [data/README.md](data/README.md) and [NOTICE.md](NOTICE.md).

A. A. Alshames, O. S. Eshra, Z. Eltannikhy, M. H. El Garawany and A. A. Sanhoury,
"Beyond the Calibration Range: Mechanisms of Variable-Selection Robustness in Raman
Quantification of Olive Oil Adulteration," JAC-ECC 2026.
