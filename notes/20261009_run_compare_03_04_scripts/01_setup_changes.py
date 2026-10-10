"""
What changed in the control file from Napa2026_03 to Napa2026_04 (makepst diff, as on the Setup Changes
page): summary, run-log lines, weight ratios by group and by date, the observations with other ratios,
bound changes (old -> new values by parameter family) and prior-information changes.
Supports note section 1.
"""
import re
import sys
from collections import Counter

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
RUN_03 = r"D:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT_LSPC\PEST\zzArchive\napa2026_03.zip"
RUN_04 = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import numpy as np
import pandas as pd
import compare
from utils import parse_zip

pd.set_option("display.width", 200)
a, b = {"pestpath": RUN_03}, {"pestpath": RUN_04}
d = compare.setup_diff(compare.run_pst(a), compare.run_pst(b))
print("makepst diff:", d.summary(), "\n")
print("run-log lines added:")
print("\n".join("  " + c for c in d.comments.comment), "\n")

obs03, obs04 = parse_zip(a, False)[2], parse_zip(b, False)[2]
ch = compare.observation_changes(obs03, obs04)
w = ch[ch.status == "changed"].copy()
w["date"] = pd.to_datetime(w.time, errors="coerce")
print("weight ratio (04 / 03) by group:")
print(w.groupby(["obsgrp", w.weight_ratio.round(3)]).size().to_string(), "\n")
site = obs04.set_index("obsname").get("grpSite")
unchanged = ch[ch.status == "same"].assign(date=lambda x: pd.to_datetime(x.time, errors="coerce"))
fifth = w[np.isclose(w.weight_ratio, 0.2, atol=1e-3)]
print("dates of weights x0.2:      ", fifth.date.min().date(), "to", fifth.date.max().date())
print("dates of unchanged weights: ", unchanged.date.min().date(), "to", unchanged.date.max().date())
cut = pd.Timestamp("2007-01-01")
early = unchanged[unchanged.date < cut]
print(f"unchanged weights dated before 2007: {len(early)} "
      f"(weight 0 in both runs: {int((early.weight_to == 0).sum())}); by site:")
print(early.assign(site=early.index.map(site)).groupby("site").agg(
    n=("date", "size"), first=("date", "min"), last=("date", "max"), weight=("weight_to", "max")).to_string())
odd = w[~np.isclose(w.weight_ratio, 0.2, atol=1e-3)]
print(f"\nchanged weights with a ratio other than 0.2: {len(odd)}, by site:")
print(odd.assign(site=odd.index.map(site)).groupby(["site", "weight_ratio"]).agg(
    n=("date", "size"), first=("date", "min"), last=("date", "max")).to_string(), "\n")

par = d.par.assign(family=d.par.PARNME.str.replace(r"\d+$", "", regex=True))
print("parameters with changed bounds:", par.PARNME.nunique(), dict(Counter(par.drop_duplicates("PARNME").family)))
print(par.groupby(["column", "family", "old", "new"]).size().to_string(), "\n")
print("prior information:")
print(d.prior.assign(group=d.prior.PINME.str.replace(r"\d+$", "", regex=True))
      .groupby(["change", "column", "old", "new", "group"]).size().to_string())
