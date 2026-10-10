"""
Parameters of Napa2026_03 vs Napa2026_04 as on the Parameter Changes page: best iterations, adjustable
parameters; how many differ by more than a factor of 2, which are on a bound, and the parameters whose
initial values sit on a bound. Supports note section 3.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
RUN_03 = r"D:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT_LSPC\PEST\zzArchive\napa2026_03.zip"
RUN_04 = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import numpy as np
import pandas as pd
import compare
from utils import parse_zip

pd.set_option("display.width", 200)
values = {}
for label, path in (("03", RUN_03), ("04", RUN_04)):
    pst, par, obs, pars, reis, jcos = parse_zip({"pestpath": path}, True, True)
    best = compare.pick_iteration(reis, compare.BEST)
    values[label] = (best, compare.parameter_values(par, pars, best), compare.parameter_values(par, pars, 0))

(i03, v03, init03), (i04, v04, init04) = values["03"], values["04"]
m = compare.paired_parameters(v03, v04)
m = m[m.partrans_from.isin(["log", "none"]) & m.partrans_to.isin(["log", "none"])]
big = m.change.abs() > np.where(m.log, np.log10(2), 1)
print(f"03 iteration {i03} -> 04 iteration {i04}: {len(m)} adjustable parameters, {int(big.sum())} differ by more than x2")
print(m[big].groupby("pargrp").size().to_string(), "\n")
print("largest changes (log10 ratio):")
print(m.reindex(m.change.abs().sort_values(ascending=False).index)[["pargrp", "value_from", "value_to", "change"]]
      .head(8).round(3).to_string(), "\n")
print("on a bound at the best iteration: 03:", list(m.index[m.at_bound_from]), " 04:", list(m.index[m.at_bound_to]), "\n")
for label, init in (("03", init03), ("04", init04)):
    a = init[init.partrans.isin(["log", "none"]) & init.at_bound]
    print(f"initial values on a bound in {label}: {list(a.index)}")
