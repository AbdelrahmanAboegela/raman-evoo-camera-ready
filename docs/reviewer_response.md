# Scientific revision and reviewer-response map

The paper title remains unchanged. The revision presents one argument: the
extrapolation benefit of selection depends on channel weighting and calibration
coverage, and controlled refits separate these effects from LV count and optimizer
identity. It does not present the sequence of development experiments.

## Contributions and evidence

| Contribution | Main evidence |
|---|---|
| Scaling changes the extrapolation benefit of selection | Main narrow-regime table: both scalings, reduced and published budgets, 15 matched master IDs throughout; six specified effects/interactions with adjusted bootstrap intervals |
| Channel weighting partly explains the benefit | Fixed-channel/fixed-LV interventions; calibration-derived variance filters; random controls; repeatability plot; matched-region check |
| A deterministic alternative is useful under identifiable conditions | VIP and iPLS under both scalings across five windows, with the same 15 assignments; three complete runtime repetitions per method |

The manuscript removes the experimental chronology, most intermediate budgets,
stability and overlap arguments, the arbitrary equivalence-margin headline, and the
optimistic post-selection CV comparison. It retains sufficient controls to justify
each scientific conclusion. Supporting analyses remain in the repository.

## Reviewer feedback

- **R1.1 â€” search budget:** both optimizers are evaluated at the published and reduced
  budgets under both scalings. Every condition uses the same 15 master IDs; budget
  dependence is measured rather than inferred.
- **R1.2 â€” external validity:** five concentration windows and the matched spectral
  region test regime dependence. The manuscript still identifies one instrument,
  one adulterant and one dataset as limits; these controls do not establish external
  instrument/adulterant validation.
- **R1.3 â€” mechanism and VIP agreement:** fixed-channel interventions replace the
  inference that selected-band overlap proves a shared mechanism. Residual agreement
  is descriptive. Formal equivalence is not asserted using a margin chosen from
  observed error.
- **R1.4 â€” cost:** complete calibration time is measured on the stated idle machine
  with one linear-algebra thread and the same timing seed. Every method, including
  published-budget GA/GWO, has three repetitions. Intermediate budget curves move
  outside the main argument.
- **R1.5 â€” replicate leakage:** the full-range table is regenerated under grouped
  development with 15 master IDs for all methods and both scalings. It gives all
  and strict test-group scores. The main low-concentration score also excludes
  calibration-shared groups (90 spectra); the main high test remains 140 spectra.
- **Additional baselines:** iPLS is retained as the classical deterministic comparator
  beyond VIP. Variance and random controls test simpler explanations.
- **Multiple comparisons:** six specified GA/GWO effects/interactions use adjusted
  paired bootstrap intervals. Repeated fits are not treated as new datasets. Window
  and control comparisons are descriptive.
- **R2.1 â€” terminology:** RMSEP_low and RMSEP_high are used consistently; the strict
  low-set definition and out-of-window above/below definitions are explicit.
- **R2.2 â€” execution:** hardware, library versions, CV grouping, seed policy and timing
  policy are stated. The exact fold indices and executable-source hashes are saved.
- **R2.3 â€” data citation:** retain the dataset's direct repository URL, access date,
  provider and licence information.

## Numerical source of truth

Only `results/camera_ready_standardized/` feeds the revised paper. The
[run policy](standardized_protocol.md) specifies the complete matrix and its audit.
`paper/make_standardized.py` requires every expected checkpoint before generating
the numerical macros, tables and figures. Earlier diagnostic numbers are not mixed
with this matrix.

## Complete standardized results

The full 1,210-record matrix passed its audit: 8,760 prediction vectors and 6,270
CV fold assignments checked, with all 90 coverage cells complete. All fitted
conditions use seed IDs 0--14; all ten timing conditions contain three repeats.

- Unit variance, mean ± population SD: PLSR 4.42 ± 0.00, VIP 1.67 ± 0.36,
  published GA-MW 1.87 ± 0.55 and GWO-MW 2.30 ± 0.84 EVOO percentage points.
- Mean centering: PLSR 3.09 ± 0.80, VIP 3.10 ± 0.64,
  published GA-MW 3.36 ± 0.73 and GWO-MW 3.67 ± 1.05.
- Adjusted scaling-interaction intervals exclude zero for GA (+2.71, [1.39,4.25])
  and GWO (+2.59, [1.22,4.20]). Within-scaling mean-centered intervals include
  zero: attenuation is supported, statistically established harm is not.
- Giving only full-spectrum PLSR the larger 1--20 LV range yields 3.87 ± 0.50
  under unit variance and 3.09 ± 0.80 under mean centering, using the same 15
  assignments. The common-cap main comparison is unchanged.
- Unit-variance calibration medians over three repeats: VIP 8.8 s, published GA
  458.0 s and GWO 77.4 s. Prediction costs are not these calibration costs.

This supports a focused paper about conditional robustness and mechanisms. It does
not support a universally superior selector, formal VIP/GA equivalence, or a claim
that mean centering makes selection harmful in general.
