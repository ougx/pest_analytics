# Scripts for 20261009_run_compare_03_04.md

All scripts read the two runs

- Napa2026_03: `D:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT_LSPC\PEST\zzArchive\napa2026_03.zip` (read only)
- Napa2026_04: `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\testdata\example_pst.zip`

and import the repository's modules (`compare.py`, `utils.py`) from `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2`
(the `REPO` constant at the top of each script). `01_setup_changes.py` also needs makepst (`pip install makepst`).
They print their results; nothing is written to disk.

| Script | Answers | Note section |
|---|---|---|
| `01_setup_changes.py` | makepst diff summary, run-log lines, weight ratios by group and date, the ×0.3 wells, bound changes by parameter family, prior-information changes | 1 |
| `02_fit_same_weights.py` | φ by iteration with own weights, best iterations; φ of both best iterations under each weight choice; RMSE by group and by site | 2 |
| `03_parameter_changes.py` | parameters differing by more than ×2, largest changes, parameters on a bound at the best iteration and at the initial values | 3 |

Run from any directory with the project's Python environment (pandas, numpy, makepst):

```bat
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_run_compare_03_04_scripts\01_setup_changes.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_run_compare_03_04_scripts\02_fit_same_weights.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261009_run_compare_03_04_scripts\03_parameter_changes.py"
```
