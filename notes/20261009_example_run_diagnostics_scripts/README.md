# Scripts for 20261009_example_run_diagnostics.md

All scripts read `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\testdata\example_pst.zip` and import the
repository's modules from `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2` (the `REPO` constant at the top of
each script). They print their results; nothing is written to disk.

| Script | Answers | Note section |
|---|---|---|
| `01_css_group_shares.py` | Composite scaled sensitivity for jco.1 and jco.6, top parameters, flow / head share of the squared sensitivity (needs `pst.composite_sensitivity`, commit `f759c84`) | 1 |
| `02_weighted_residual_distribution.py` | Mean, std, Q-Q quartile-line slope of the weighted residuals at iterations 0 and 6; share of φ in the largest residuals; their sites (needs `plotting.normal_scores`, commit `adde485`) | 2 |
| `03_linear_vs_actual_phi.py` | Per iteration: first-order φ reduction from the step, linear-model prediction, actual change (needs `pst.step_contribution`, commit `85398e3`) | 3 |
| `04_regularisation_misfit.py` | Regularisation misfit by group per iteration with the .pst weights, largest equations in .rei.6 (needs `module/regul.py`, commit `fc3f930`, and makepst) | 4 |

Run from any directory with the project's Python environment (pandas, numpy, scipy, makepst):

```bat
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_example_run_diagnostics_scripts\01_css_group_shares.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_example_run_diagnostics_scripts\02_weighted_residual_distribution.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_example_run_diagnostics_scripts\03_linear_vs_actual_phi.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_example_run_diagnostics_scripts\04_regularisation_misfit.py"
```
