# Observation conflict on the example data, and net pull vs. PEST's step

2026-10-08. Data: `testdata/example_pst.zip` (Napa2026_04, PEST_HP, 6 iterations; obs groups
`head` 4485 targets, `flow` 948). Scripts: [20261008_obs_conflict_scripts](20261008_obs_conflict_scripts/README.md).

## Questions

1. On the Observation Conflict page (last Jacobian, grouped by `obsgrp`), the heatmap shows head
   blue and flow red in every column. Does head uniformly want the parameters to decrease and
   flow want them to increase?
2. If a parameter's net pull is negative, will PEST decrease it in the next iteration?

## 1. Is the heatmap right: head down, flow up?

The reading is correct for the columns shown, but not for flow overall
(`01_conflict_by_group.py`).

| Jacobian | parameters | head up / down | flow up / down | top-30 columns with flow up, head down |
|---|---|---|---|---|
| jco.1 | 131 | 74 / 57 | 59 / 72 | 17 |
| jco.2 | 127 | 52 / 75 | 72 / 55 | 24 |
| jco.3 | 105 | 48 / 57 | 28 / 77 | 10 |
| jco.4 | 92 | 7 / 85 | 80 / 12 | 30 |
| jco.5 | 61 | 25 / 36 | 56 / 5 | 28 |
| jco.6 | 64 | 0 / 64 | 38 / 26 | 30 |

- **Flow does not want every parameter up.** At jco.6, flow pulls 38 parameters up and 26 down. The
  heatmap is one-sided because its columns are the top 30 parameters by tension, and with two groups a
  parameter only has tension if the groups disagree. Parameters on which flow pulls down mostly agree with
  head and are left out. With two groups each column is also just one red and one blue cell whose
  shares add up to 100%, so the heatmap repeats the bar chart.
- **Head does pull every parameter down at jco.6**, but the pattern only appears from jco.4 onward;
  in iterations 1–3 both groups pull both ways.

## 2. Why head pulls everything down: one well

(`02_head_pull_concentration.py`, jco.6.)

- Site `nc-76` (37 targets) carries 63% of the head group's absolute pull; the next site
  (`t143mw-3`) carries 5%. Its median share per parameter is 69%.
- `nc-76` pulls 62 of 64 parameters down. Without it, the remaining head targets pull 32 up and 32 down
  and oppose flow on 32 of 64 parameters, which is ordinary scatter.
- The model is far too low there: measured mean 15.6 ft, modelled mean −48.3 ft, residual mean
  63.9 ft (range 38.7–91.7 ft). Raising almost any parameter lowers the modelled head at that well
  (186 positive vs. 2182 negative Jacobian entries), so its pull points down on nearly everything.

So the conflict is `nc-76` against everything else, not heads against flows. A well that is 50–80 ft too
low and gets lower whichever parameter goes up suggests a local problem (layer/screen assignment, a
nearby pumping or drying cell, a datum error) that the calibration parameters cannot fix. To check in the app:
group by `grpSite`, or use Target Focus on `grpSite = nc-76`.

## 3. Does a negative net pull mean the parameter decreases next iteration?

No (`03_net_pull_vs_step.py`). The step of iteration N (`.par.(N-1)` → `.par.N`; log10 ratio,
relative change for the 3 untransformed parameters) against the sign of the net pull of `.jco.N`:

| Jacobian | parameters | moved | followed net pull | correlation |
|---|---|---|---|---|
| 1 | 131 | 122 | 87 (71%) | 0.29 |
| 2 | 127 | 118 | 68 (58%) | 0.02 |
| 3 | 105 | 101 | 57 (56%) | 0.10 |
| 4 | 92 | 90 | 52 (58%) | 0.04 |
| 5 | 61 | 58 | 32 (55%) | 0.15 |
| 6 | 64 | 60 | 33 (55%) | 0.07 |

After the first iteration the agreement is close to a coin flip. The net pull is the steepest-descent
direction, summed over targets `2 JᵀQr`. PEST takes the Gauss-Newton/Marquardt step
`(JᵀQJ + λI)⁻¹ JᵀQr`, and the two differ because of:

- **Parameter correlation:** `(JᵀQJ)⁻¹` lets PEST raise one parameter and lower a correlated one against its pull. Only a very large λ follows the pull.
- **Regularisation:** the `.pst` has `* prior information` and `* regularisation` sections (homogeneity regularisation on pilot points since Napa2026_03). It pulls too, and is not a target on the conflict page.
- **SVD:** PEST_HP's truncated SVD moves parameter combinations, not individual parameters.
- **Bounds and change limits:** `parchglim`, `facparmax`/`relparmax`, and tied parameters.
- **Linearisation:** the pull is only valid at the parameters where the Jacobian was computed.

Use the net pull to see which way the data want a parameter to go, not to predict the next value.

## Outcome

- Added the Next Step Check chart to the Observation Conflict page, and expanded its "About this
  analysis" text with the heatmap cautions and the explanation above (commit `d79d349`).

## Caveats

- Pull is a linear quantity at the parameters where each Jacobian was computed.
- An earlier in-chat version of the section 3 table used log10 for all parameters, so its jco.1
  correlation was 0.31 instead of 0.29; the agreement counts are the same.
