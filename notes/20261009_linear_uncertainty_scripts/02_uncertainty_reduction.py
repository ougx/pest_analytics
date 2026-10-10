"""
Linear (FOSM) uncertainty reduction on the example data, as on the Parameter Uncertainty page:
prior std from the bounds (span = 4 std), noise from the weights or scaled by phi/n, with and without
regularization rows, for .jco.1 and .jco.6. Prints phi/n, the median reduction over the parameters with
measurement sensitivity, and the mean reduction per parameter group (scaled noise, .jco.6).
Supports note section 2.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import pandas as pd
from utils import read_pest_zip
from pst import get_jco_rei
from fosm import linear_uncertainty, measurement_rows, prior_information_rows, noise_scale

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
measured = measurement_rows(obs)
for n in (1, 6):
    jco, rei = jcos[n], get_jco_rei(reis, n)
    scale = noise_scale(obs, rei)
    print(f".jco.{n}: residuals of iteration {n - 1}, phi/n = {scale:.1f} (noise std x {scale ** 0.5:.1f})")
    for regul in (False, True):
        rows = measurement_rows(obs, include_regul=regul)
        if regul:
            prior = prior_information_rows(obs, rei)
            rows = pd.concat([rows, prior[~prior.index.isin(rows.index)]])
        for label, s in (("weights", 1.0), ("scaled ", scale)):
            df = linear_uncertainty(jco, par, rows, 4.0, s, measured=measured.index)
            sens = df[df.sensitive]
            print(f"   regularization {'on ' if regul else 'off'}, noise {label}: median reduction "
                  f"{100 * sens.reduction.median():5.1f}% over {len(sens)} parameters "
                  f"({len(df) - len(sens)} without measurement sensitivity)")
    print()

df = linear_uncertainty(jcos[6], par, measured, 4.0, noise_scale(obs, get_jco_rei(reis, 6)), measured=measured.index)
g = df[df.sensitive].groupby("pargrp").reduction
print("jco.6, scaled noise, regularization off: mean reduction by group (parameters with sensitivity)")
print((100 * g.mean()).round(1).sort_values(ascending=False).to_string())
