"""
Addendum: what the run record (.rec) says about the all-zero Jacobian columns. For every iteration N:
the parameters PEST_HP reported as having "no effect on observations" (after computing .jco.N), the
columns of .jco.N that are zero for the weighted observations, for all observations, and for all rows
(observations and prior information), and the lambda tests whose model run failed.
Supports the note's addendum.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
from utils import read_pest_zip
from rec import read_rec

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
r = read_rec({"pestpath": ZIP})
events, it = r["events"], r["iterations"]
weighted = set(obs[obs.weight > 0].obsname)
observations = set(obs.obsname)

print("it  no_effect  zero_weighted  zero_all_obs  zero_all_rows  no_effect=zero_all_rows  failed_lambda/tests")
for n in sorted(jcos):
    J = jcos[n]
    zero = lambda rows: set(J.columns[J.loc[J.index.isin(rows)].abs().sum() == 0])
    z_w, z_o, z_all = zero(weighted), zero(observations), set(J.columns[J.abs().sum() == 0])
    no_effect = set(events[(events.iteration == n) & (events.event == "no effect")].parameter)
    print(f"{n:>2}  {len(no_effect):>9}  {len(z_w):>13}  {len(z_o):>12}  {len(z_all):>13}  "
          f"{str(no_effect == z_all):>23}  {int(it.loc[n, 'failed_lambda_runs']):>3}/{int(it.loc[n, 'lambda_runs'])}")

# which zero columns are not reported: are they the parameters in prior-information equations?
n = 5
J = jcos[n]
z_o = set(J.columns[J.loc[J.index.isin(observations)].abs().sum() == 0])
z_all = set(J.columns[J.abs().sum() == 0])
prior_rows = J.loc[~J.index.isin(observations)]
in_prior = set(prior_rows.columns[prior_rows.abs().sum() > 0])
print(f"\njco.{n}: {len(z_o - z_all)} parameters zero for every observation but not reported; "
      f"all of them in prior-information equations: {(z_o - z_all) <= in_prior}")
