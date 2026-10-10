# Scripts for 20261009_linear_uncertainty.md

Both scripts read `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\testdata\example_pst.zip` and import
the repository's modules from `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2` (the `REPO` constant at
the top of each script). They print their results; nothing is written to disk.

| Script | Answers | Note section |
|---|---|---|
| `01_zero_jacobian_columns.py` | For every Jacobian: all-zero columns for the weighted observations, parameters on a bound where it was computed, and the `.rec` run-failure settings | 1 |
| `02_uncertainty_reduction.py` | φ/n and the median uncertainty reduction for jco.1 and jco.6, noise from weights or scaled, with and without regularization rows; mean reduction by group for jco.6 (needs `fosm.py`, commit `1680056`) | 2 |

Run from any directory with the project's Python environment (pandas, numpy):

```bat
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_linear_uncertainty_scripts\01_zero_jacobian_columns.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_linear_uncertainty_scripts\02_uncertainty_reduction.py"
```
