# Linear parameter uncertainty on the example data, and the zero Jacobian columns

2026-10-09. Data: `testdata/example_pst.zip` (Napa2026_04, PEST_HP, 6 iterations, 132 adjustable parameters,
5161 weighted observations). Done while building the Parameter Uncertainty page (review item 7.2, `fosm.py`,
`module/parunc.py`). Scripts: [20261009_linear_uncertainty_scripts](20261009_linear_uncertainty_scripts/README.md).

## Questions

1. Can the saved Jacobians be used for uncertainty (and the other Jacobian-based pages) as they are?
2. How much does calibration reduce parameter uncertainty, and how much does that depend on the
   assumed measurement noise?

## 1. Later Jacobians have many all-zero columns

`01_zero_jacobian_columns.py`. Columns that are zero for every weighted observation:

| Jacobian | zero columns (of 132) | parameters on a bound | zero and on a bound |
|---|---|---|---|
| jco.1 | 1 | 10 | 1 |
| jco.2 | 5 | 12 | 0 |
| jco.3 | 27 | 9 | 4 |
| jco.4 | 40 | 9 | 6 |
| jco.5 | 71 | 9 | 4 |
| jco.6 | 68 | 7 | 2 |

- The zero columns are mostly **not** parameters on a bound (bounds checked at the parameters where each
  Jacobian was computed, `.par.(N-1)`).
- The `.rec` file has "Forgive model run failure during Jacobian runs: yes", so the likely cause is failed
  derivative runs that PEST_HP accepted with a zero column. The `.rec` file does not list Jacobian run failures,
  so this is not confirmed; PEST_HP's run-management output would show it.
- **Consequence:** from jco.3 on, a large share of the parameters has no sensitivity in the saved Jacobian, so
  identifiability, uncertainty, correlation, sensitivity and conflict for those parameters say nothing.
  jco.1 is the only nearly complete Jacobian. The Parameter Uncertainty page flags these parameters.

## 2. Uncertainty reduction depends mostly on the assumed noise

`02_uncertainty_reduction.py`. Prior std from the bounds (bounds span 4 std, log10 units for log
parameters), posterior $(J^TQJ + C_{prior}^{-1})^{-1}$; reduction $= 1 - \sigma_{post}/\sigma_{prior}$, median over
the parameters with measurement sensitivity.

| Jacobian | φ/n (residuals where it was computed) | noise from weights | noise scaled by φ/n |
|---|---|---|---|
| jco.1 | 223.3 (noise std ×14.9) | 92.9% | 38.2% |
| jco.6 | 132.1 (noise std ×11.5) | 99.0% (64 parameters) | 89.0% (64 parameters) |

- The weights are far from 1/noise: the misfit implies noise 11–15 times larger than the weights do. With the
  weights taken as they are, nearly every parameter looks well constrained. The page therefore scales the noise
  to the fit by default.
- Adding the regularization prior-information rows changes nothing at this precision (their weights after the
  regularization weight factor are small, about 0.03–0.16). For jco.1 there are none to add: its residuals come from
  the `.hp` file, which holds no prior information.
- jco.6, scaled noise, mean reduction by group (parameters with sensitivity only): sfrk 92.9%, ss 92.1%,
  vk 91.3%, hk 88.2%, sy 85.5%, ghbthick 77.8%. These are high only because the 68 parameters without
  sensitivity are left out.

## Caveats

- Linear analysis around one Jacobian; the bounds-based prior is a convention (pyEMU's), not a statement of
  real prior knowledge.
- Corrections to numbers given in chat on 2026-10-09: φ/n "≈ 132" holds for jco.6; it is 223 for jco.1. Group
  means quoted then for jco.6 (sfrk 65% … ghbthick 46%) counted the parameters without sensitivity as zero
  reduction; the figures above leave them out, as the page now does.
