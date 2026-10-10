"""
Does PEST's step follow the net pull? For every .jco.N of the example data, the sign of the
net pull (all targets, .pst weights) against the step of iteration N, .par.(N-1) -> .par.N
(log10 ratio; relative change for untransformed parameters). Supports note section 3.
The same numbers are shown by the Next Step Check on the Observation Conflict page.
"""
import sys
import numpy as np

REPO = r"D:\Cloud\sspa Dropbox\Michael Ou\0025_PEST_HP_Postproc2"
ZIP = REPO + r"\testdata\example_pst.zip"
sys.path.insert(0, REPO)
from utils import read_pest_zip
from pst import get_obs_pull, get_jco_rei, iteration_step, step_agreement

pst, par, obs, pars, reis, jcos = read_pest_zip(ZIP, True, True, True)
print("jco | parameters | moved | followed net pull | correlation(net pull, step)")
for n in sorted(jcos):
    step, at_bound = iteration_step(pars, n)
    if step is None:
        continue
    net = get_obs_pull(obs, jcos[n], get_jco_rei(reis, n)).sum()
    agree, moved = step_agreement(net, step)
    d = step.reindex(net.index)
    m = d.abs() > 1e-6
    r = np.corrcoef(net[m], d[m])[0, 1]
    print(f"{n:3d} | {len(net):10d} | {moved:5d} | {agree:3d} ({agree / moved:.0%}) | {r:.2f}")
