"""
Distribution of the weighted residuals of the example run (weighted observations, regularization excluded) at
iterations 0 and 6: mean, standard deviation, quartile (Q-Q) line slope, and how much of phi the largest
weighted residuals carry. Supports note section 2.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import numpy as np
from utils import read_pest_zip
from plotting import normal_scores, qq_line

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, False, True, False)
for n in (0, 6):
    rei = reis[n]
    df = rei[(rei.weight > 0) & ~rei.obsgrp.str.startswith("regul")]
    wr = (df.weight * df.residual).values
    z, ordered, order = normal_scores(wr)
    a, b = qq_line(z, ordered)
    sq = np.sort(wr ** 2)[::-1]
    print(f"iteration {n}: n = {len(wr)}, mean {wr.mean():.3f}, std {wr.std():.2f}, quartile-line slope {b:.2f}")
    for frac in (0.01, 0.05):
        k = int(round(frac * len(wr)))
        print(f"   the largest {frac:.0%} ({k}) of the weighted residuals carry {sq[:k].sum() / sq.sum():.1%} of phi")
    top = df.assign(wr2=wr ** 2).sort_values("wr2", ascending=False).head(200)
    print("   sites among the 200 largest:", top.grpSite.value_counts().head(5).to_dict())
