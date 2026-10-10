# Scripts for 20261008_obs_conflict.md

All scripts read `D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\testdata\example_pst.zip`
and import the repository's `utils.py` and `pst.py` from
`D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2` (the `REPO` constant at the top of each script).
They print their results; nothing is written to disk.

| Script | Answers | Note section |
|---|---|---|
| `01_conflict_by_group.py` | For every Jacobian: how many parameters head and flow each pull up / down, and how many of the top-30 tension parameters have flow up and head down; tension and group shares of the top 10 at jco.6 | 1 |
| `02_head_pull_concentration.py` | At jco.6: share of the head pull by site, the pull direction of `nc-76` and of the other head targets, `nc-76` residuals and Jacobian signs | 2 |
| `03_net_pull_vs_step.py` | For every Jacobian: share of moved parameters whose step in that iteration has the sign of the net pull (needs `pst.iteration_step`, added in commit `d79d349`) | 3 |

Run from any directory with the project's Python environment (pandas, numpy):

```bat
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261008_obs_conflict_scripts\01_conflict_by_group.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261008_obs_conflict_scripts\02_head_pull_concentration.py"
python -I "D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2\notes\20261008_obs_conflict_scripts\03_net_pull_vs_step.py"
```
