"""
Which head targets carry the head group's pull at the last Jacobian (example data):
share of the head |pull| by site, the pull direction of site nc-76, the head pull
without nc-76, and nc-76's residuals and Jacobian signs. Supports note section 2.
"""
import sys
import numpy as np
import pandas as pd

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
SITE = "nc-76"
sys.path.insert(0, REPO)
from utils import read_pest_zip
from pst import get_obs_pull, get_jco_rei

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
o = obs.set_index("obsname")
n = max(jcos)
rei = get_jco_rei(reis, n)
pull = get_obs_pull(obs, jcos[n], rei)
grp, site = o.obsgrp.reindex(pull.index), o.grpSite.reindex(pull.index)
H, hs = pull[grp == "head"], site[grp == "head"]
F = pull[grp == "flow"].sum()

print(f"jco.{n}: {len(H)} head targets with pull, {H.shape[1]} parameters")
by_site = H.abs().groupby(hs.values).sum().sum(axis=1)
print("share of head |pull| by site (top 8):")
print((by_site / by_site.sum()).sort_values(ascending=False).head(8).round(3).to_string())

nc = (hs == SITE).values
print(f"\n{SITE}: {nc.sum()} targets; median share of head |pull| per parameter:",
      round((H[nc].abs().sum() / H.abs().sum()).median(), 2))
print(f"{SITE} pull up/down:", (H[nc].sum() > 0).sum(), (H[nc].sum() < 0).sum())
rest = H[~nc].sum()
print(f"head without {SITE} pull up/down:", (rest > 0).sum(), (rest < 0).sum(),
      "| opposing flow on", (np.sign(rest) * np.sign(F) < 0).sum(), "of", len(F))
J = jcos[n].loc[H.index[nc], H.columns]
print(f"{SITE} Jacobian entries +/-:", int((J > 0).sum().sum()), int((J < 0).sum().sum()))
print(rei.set_index("obsname").loc[H.index[nc], ["measured", "modelled", "residual"]].describe().round(1).to_string())
