"""
Composite scaled sensitivity (CSS) of the adjustable parameters on the example data, for .jco.1 and .jco.6:
the top parameters and the share of the flow and head groups in sum (w J)^2. Supports note section 1.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import pandas as pd
from utils import read_pest_zip
from pst import composite_sensitivity

pd.set_option("display.width", 200)
pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, False, False, True)
print("weighted observations by group:", obs[obs.weight > 0].obsgrp.value_counts().to_dict())
for n in (1, 6):
    css = composite_sensitivity(obs, jcos[n], "obsgrp")
    sens = css[css.css > 0]
    print(f"\n.jco.{n}: {len(sens)} of {len(css)} parameters with sensitivity; top 10:")
    print(sens.head(10).round(4).to_string())
    for k in (10, 30, len(sens)):
        print(f"  flow share of the squared sensitivity, median over the top {k}: {sens.flow.head(k).median():.3f}")
    print(f"  parameters where head carries more than half: {int((sens["head"] > 0.5).sum())} of {len(sens)}")
