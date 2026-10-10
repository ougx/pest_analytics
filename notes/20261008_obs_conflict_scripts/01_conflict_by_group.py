"""
Observation conflict between obs groups (head vs flow) on the example data, for every
Jacobian iteration: tension and group shares as on the Observation Conflict page
(grouped by obsgrp, .pst weights), and how many parameters each group pulls up / down.
Supports note sections 1 and 2.
"""
import sys
import numpy as np
import pandas as pd

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
from utils import read_pest_zip
from pst import get_obs_pull, get_par_tension, get_jco_rei

pd.set_option("display.width", 200)
pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
o = obs.set_index("obsname")
print(obs.obsgrp.value_counts().to_string(), "\n")

for n in sorted(jcos):
    pull = get_obs_pull(obs, jcos[n], get_jco_rei(reis, n))
    G = pull.groupby(o.obsgrp.reindex(pull.index).values).sum()
    T = get_par_tension(G).sort_values("tension", ascending=False)
    top = T.index[:30]
    s = np.sign(G)
    print(f"jco.{n}: {G.shape[1]} parameters | head up/down {(s.loc['head'] > 0).sum()}/{(s.loc['head'] < 0).sum()}"
          f" | flow up/down {(s.loc['flow'] > 0).sum()}/{(s.loc['flow'] < 0).sum()}"
          f" | top-30 by tension with flow up & head down: {((s.loc['flow', top] > 0) & (s.loc['head', top] < 0)).sum()}")

n = max(jcos)
pull = get_obs_pull(obs, jcos[n], get_jco_rei(reis, n))
G = pull.groupby(o.obsgrp.reindex(pull.index).values).sum()
T = get_par_tension(G).sort_values("tension", ascending=False)
share = G / G.abs().sum()
print(f"\njco.{n}: top 10 parameters by tension, with group shares")
print(T.head(10).join(share.T.add_suffix("_share")).round(3).to_string())
