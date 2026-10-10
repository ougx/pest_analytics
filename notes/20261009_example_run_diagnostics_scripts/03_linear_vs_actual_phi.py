"""
For every iteration of the example run: the first-order phi reduction from PEST's step (sum of pull x step),
the change of measurement phi the linearized model predicts for that step (residuals r - J dp), and the
actual change between the residual files, with the .pst weights. Supports note section 3.
"""
import sys

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
from utils import read_pest_zip
from pst import get_obs_pull, get_jco_rei, jacobian_step, step_contribution, linear_phi_change


def measurement_phi(rei):
    df = rei[(rei.weight > 0) & ~rei.obsgrp.str.startswith("regul")]
    return float(((df.weight * df.residual) ** 2).sum())


pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
print("it  first_order_reduction  linear_change  actual_change  actual/linear")
for n in sorted(jcos):
    rei, step = get_jco_rei(reis, n), jacobian_step(pars, n)
    C = step_contribution(get_obs_pull(obs, jcos[n], rei), step)
    linear = linear_phi_change(obs, jcos[n], rei, step)
    actual = measurement_phi(reis[n]) - measurement_phi(reis[n - 1])
    print(f"{n:>2}  {C.values.sum():>21.4g}  {linear:>+13.4g}  {actual:>+13.4g}  {actual / linear:>13.2f}")
