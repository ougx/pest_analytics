# Example run diagnostics from the pages rebuilt from pest_app

2026-10-09. Data: `testdata/example_pst.zip` (Napa2026_04, PEST_HP; 5161 weighted observations: head 4272, flow 889).
Numbers found while building the pages from review item 7.7 (composite scaled sensitivity, weighted residual
distribution, step contribution, regularisation). Scripts: [20261009_example_run_diagnostics_scripts](20261009_example_run_diagnostics_scripts/README.md).

## 1. The flow targets carry nearly all of the sensitivity

`01_css_group_shares.py` (composite scaled sensitivity, CSS = √(Σ(wJ)²/n), split by obsgrp).

- jco.1: the 10 most sensitive parameters (syfn09, kmaxfn06, kmaxfn05, sfrvk106, ...) take 99.9% (median) of their
  squared sensitivity from the 889 flow targets; over all 131 sensitive parameters the median flow share is 97.4%.
  Only 12 of 131 parameters get more than half of it from the 4272 heads.
- jco.6: median flow share 97.6% over the 64 parameters with sensitivity; none is head-dominated.
- So the weighted Jacobian, and with it every Jacobian-based page (identifiability, uncertainty, correlation),
  is effectively a flow analysis; the head data barely inform the parameters at these weights.

## 2. Weighted residuals: a narrow core and heavy tails

`02_weighted_residual_distribution.py` (Residual Analysis, weighted residual distribution).

| Iteration | std | Q-Q quartile-line slope | share of φ in the largest 1% | in the largest 5% |
|---|---|---|---|---|
| 0 | 14.79 | 4.37 | 32.6% | 73.0% |
| 6 | 11.42 | 4.62 | 34.0% | 65.2% |

- The middle half of the weighted residuals has a spread of about 4.5 (the quartile-line slope), the overall
  standard deviation is 11–15: a few targets carry most of φ (52 targets, 1%, carry a third of it).
- Sites among the 200 largest at iteration 6: napar_napa 23, nc-76 23, nc-223d 20, t168mw-6 20, nc-182 16.
- Implication for 7.2: scaling the noise by √(φ/n) (×11.5) is driven by these tails; for most targets the misfit is
  about 4.5 times what the weights imply, not 11.5.

## 3. The linearized model stops predicting the real change after iteration 1

`03_linear_vs_actual_phi.py` (Observation Conflict, step contribution; .pst weights).

| Iteration | first-order reduction (Σ pull × step) | change predicted by the linear model | actual change | actual / linear |
|---|---|---|---|---|
| 1 | 5.97e5 | −3.20e5 | −3.18e5 | 0.99 |
| 2 | 8.16e4 | −7.50e4 | −3.84e4 | 0.51 |
| 3 | 2.04e5 | +2.5e3 | −5.03e4 | −19.8 |
| 4 | 1.22e5 | +3.33e5 | −3.72e4 | −0.11 |
| 5 | 1.84e5 | −1.33e5 | −2.69e4 | 0.20 |
| 6 | 1.29e5 | −9.30e4 | −8.2e3 | 0.09 |

- In iteration 1 the saved Jacobian predicts the actual φ change almost exactly. From iteration 3 on, the linear
  model even predicts an increase where φ fell, and later reductions are 9–20% of what it predicts.
- Consistent with the zero Jacobian columns and the failed model runs (`20261009_linear_uncertainty.md`): the later
  Jacobians do not describe the model over PEST's steps.

## 4. Two preferred-value equations carry most of the regularisation misfit

`04_regularisation_misfit.py` (Regularisation page; prior-information residuals in the .rei files, .pst weights).

- Regularisation misfit by group, iterations 1–6: regulhk 1126 → 1448 (peak 1746 at iteration 4); regulss, regulsy
  and regulvk stay below 9.
- .rei.6: total 1466, of which kdcr01 60.2% and kdfn01 24.1% (84.3% together) — the preferred-value equations added in
  Napa2026_04 with weight 50 ("high weights for regul Kd"). kdcr01 ends a factor 3.9 above its preferred value
  (residual −0.59 in log10), kdfn01 a factor 2.4 (−0.38).
- So the data pull Kd away from the preferred values despite the high weights.

## Caveats

- Weights are those of the `.pst` file; with `weight_orig` the flow / head balance would differ.
- The linear-model prediction uses the saved Jacobian of each iteration, which has zero columns from iteration 2 on.
