# Camera-ready run policy

The final manuscript uses only `results/camera_ready_standardized/`. Earlier exploratory
and fixed-grouping analyses remain available for provenance but do not supply its
reported performance, inference, or runtime numbers.

## Matched repetitions

- Master seed IDs are 0 through 14 for **every fitted condition in the paper**.
- Each ID defines shuffled group-preserving CV folds, matched across methods and
  scalings within the same calibration window. The optimizers also use that ID for
  their random search. Deterministic selectors therefore have 15 meaningful
  fold-assignment runs, rather than 15 duplicate fits under identical folds.
- Search fitness uses three folds and five LV for both optimizers. Final LV selection,
  VIP outer/inner selection and iPLS scoring use five folds. Fold counts differ only
  by their declared purpose, not arbitrarily across competing methods.
- Main above-range predictions always use the same 140 spectra. Main low-concentration
  summaries exclude 15 groups-shared rows and use the same strict set of 90 spectra.
- Narrow reduced and published budgets each have 15 repetitions per optimizer and
  scaling. All five deterministic-method window comparisons, cropped-region checks,
  full-range leakage comparisons, and random/variance controls also have 15 repetitions.
- The full-range and cropped-region optimizer checks use the declared reduced budget;
  they are controls, not published-budget comparisons. For each optimizer, budgets
  are identical under the two scalings and stated in captions/text.
- Every runtime comparison uses three complete repetitions on an idle machine, one
  BLAS thread, and master seed 0. This common timing seed prevents different selected
  model sizes from being mistaken for execution variability.
- Static channel-variance summaries and the fixed-LV full-spectrum reference have no
  random choice or CV-selected parameter. These deterministic controls are calculated
  once and are not counted as independent experimental repetitions.

## Completion and audits

The frozen primary matrix plus the declared baseline-cap sensitivity contains 1,210 checkpoint records:

| Analysis | Records |
|---|---:|
| Narrow search: 2 budgets Ã— 2 scalings Ã— 2 methods Ã— 15 seeds | 120 |
| Five-window references: 5 windows Ã— 2 scalings Ã— 3 methods Ã— 15 seeds | 450 |
| Full-range reference: 2 scalings Ã— 5 methods Ã— 15 seeds | 150 |
| Cropped-region reference: 2 scalings Ã— 5 methods Ã— 15 seeds | 150 |
| Controls: 2 scalings Ã— 7 controls Ã— 15 seeds | 210 |
| Higher-cap PLSR sensitivity: 2 scalings × 15 seeds | 30 |
| Fixed-channel refits: 2 source scalings Ã— 3 selectors Ã— 15 seed-specific sets | 90 |
| Timing: 2 scalings Ã— 5 methods, each containing 3 repetitions | 10 |

Each fixed-channel record contains both refit scalings and all 15 fixed LV counts.
Every predictive checkpoint stores predictions and exact train/validation indices.
`paper/make_standardized.py` refuses final artifact generation if any required record
is missing or any timing cell has fewer than three repetitions. It independently
recomputes prediction errors and checks every stored CV split against the common
seed-specific grouping before generating manuscript macros, tables and figures.

Primary uncertainty uses paired mixture-group and master-seed resampling, including
the corresponding PLSR repetition. Six predeclared GA/GWO contrasts receive
Bonferroni percentile intervals; no seed-level independent-dataset claim is made.

The higher-cap check gives full-spectrum PLSR a 1--20 LV range under both scalings,
using the same 15 assignments. The primary comparison retains its common 1--10 LV
range. This baseline-only sensitivity checks whether the primary upper cap explains
the selection benefit; it is not an all-method 20-LV benchmark.
