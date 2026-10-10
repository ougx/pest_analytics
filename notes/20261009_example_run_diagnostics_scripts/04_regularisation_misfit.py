"""
Regularisation misfit of the example run: (w r)^2 of the prior-information equations in each .rei file with
the .pst weights, by group per iteration, and the equations that carry most of it in .rei.6 (residuals in log10
units for the log() equations). Needs makepst. Supports note section 4.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
import pandas as pd
from utils import parse_zip
from module.regul import prior_weights, regularisation_rows

pd.set_option("display.width", 200)
data = {"pestpath": ZIP}
pst, par, obs, pars, reis, jcos = parse_zip(data, False, True)
prior = prior_weights(data)
rows = {i: regularisation_rows(rei, obs, prior) for i, rei in sorted(reis.items())}
by_group = pd.DataFrame({i: r.groupby("group").phi.sum() for i, r in rows.items() if len(r)}).T
print("regularisation misfit by group (.pst weights):")
print(by_group.round(1).to_string(), "\n")
r6 = rows[6].sort_values("phi", ascending=False)
r6["share"] = r6.phi / r6.phi.sum()
print(f".rei.6: total {r6.phi.sum():.1f}; largest equations:")
print(r6.head(6)[["group", "equation", "residual", "weight_pst", "phi", "share"]].round(4).to_string())
print(f"\nkdcr01 + kdfn01 share: {r6.loc[['kdcr01', 'kdfn01'], 'share'].sum():.1%}; "
      f"kdcr01 is a factor {10 ** abs(r6.loc['kdcr01', 'residual']):.2f} from its preferred value")
