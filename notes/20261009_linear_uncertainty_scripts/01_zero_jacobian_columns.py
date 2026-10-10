"""
All-zero Jacobian columns on the example data: for every .jco.N, how many adjustable parameters have
no sensitivity to any weighted observation, how many of them sit on a bound at the parameters where
the Jacobian was computed (.par.(N-1), the initial values for N = 1), and the PEST_HP run-failure
setting in the .rec file. Supports note section 1.
"""
import sys
import zipfile
import numpy as np

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
from utils import read_pest_zip

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
weighted = obs[obs.weight > 0].obsname
print(f"{len(weighted)} weighted observations, {jcos[1].shape[1]} adjustable parameters in the Jacobians\n")
print("jco  zero_columns  at_bound  zero_and_at_bound")
for n in sorted(jcos):
    J = jcos[n]
    Jo = J.loc[J.index.isin(weighted)]
    zero = set(Jo.columns[Jo.abs().sum() == 0])
    p = par.set_index("parname").loc[J.columns]
    value = p.parval0 if n == 1 else pars[n - 1].set_index("parname").parval.loc[J.columns]
    at_bound = set(J.columns[np.isclose(value, p.parlbnd, rtol=1e-6) | np.isclose(value, p.parubnd, rtol=1e-6)])
    print(f"{n:>3}  {len(zero):>12}  {len(at_bound):>8}  {len(zero & at_bound):>17}")

with zipfile.ZipFile(ZIP) as z:
    rec = [n for n in z.namelist() if n.lower().endswith(".rec")][0]
    for line in z.read(rec).decode(errors="replace").splitlines():
        if "Forgive model run failure" in line:
            print("\n.rec:", line.strip())
