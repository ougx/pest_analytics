"""
Reading the PEST / PEST_HP run record (.rec) in a run's zip: the settings that matter for reading the
run, and for every optimisation iteration the model calls, objective functions (measurement,
regularization, by group), the regularization weight factor and target, the Marquardt lambda tests,
parameters with no effect or frozen, and the largest parameter changes.
"""
import functools
import os
import re
import zipfile

import numpy as np
import pandas as pd

from utils import _data_source, check_data

NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
ITERATION = re.compile(r"OPTIMISATION ITERATION NO\.\s*:\s*(\d+)")
FIELDS = {
    "model_calls": r"Model calls so far\s*:\s*" + NUMBER,
    "weight_factor": r"Current regularisation weight factor\s*:\s*" + NUMBER,
    "phi_measurement": r"Current value of measurement objective function\s*:\s*" + NUMBER,
    "phi_regularization": r"Current value of regularisation objective function\s*:\s*" + NUMBER,
    "phi_start": r"Starting phi for this iteration\s*:\s*" + NUMBER,
    "phi_target": r"target measurement objective function\s*:\s*" + NUMBER,
    "new_weight_factor": r"Re-calculated regularisation weight factor\s*:\s*" + NUMBER,
}
LAMBDA = re.compile(r"Lambda\s*=\s*" + NUMBER + r"(?:\s+upgrade fraction\s*=\s*" + NUMBER + r")?\s*-+>")
PHI = re.compile(r"^\s*Phi\s*=\s*" + NUMBER)
MEAS = re.compile(r"Meas\. fn\.\s*=\s*" + NUMBER)
REGUL = re.compile(r"Regul\. fn\.\s*=\s*" + NUMBER)
GROUP = re.compile(r'Contribution to phi from observation group "([^"]+)"\s*:\s*' + NUMBER)
NO_EFFECT = re.compile(r'Parameter "([^"]+)" has no effect on observations', re.IGNORECASE)
FROZEN = re.compile(r'param "([^"]+)" frozen', re.IGNORECASE)
MAX_CHANGE = re.compile(r"Maximum\s+(factor|relative)\s+change:\s*" + NUMBER + r'\s+\["([^"]+)"\]')
SETTINGS = {
    "Forgive model run failure during lamda testing": "forgive_lambda_failures",
    "Forgive model run failure during Jacobian runs": "forgive_jacobian_failures",
    "Maximum number of optimisation iterations": "noptmax",
    "Relative phi reduction indicating convergence": "phiredstp",
    "Target/current measurement objective function ratio": "fracphim",
    "Minimum target measurement objective function value": "phimlim",
    "Number of adjustable parameters": "adjustable_parameters",
    "Number of observations": "observations",
    "Number of prior estimates": "prior_estimates",
}
END = re.compile(r"(Optimisation complete|OPTIMISATION RESULTS|Total model calls)", re.IGNORECASE)


def read_rec(data):
    """the run record of a loaded run (the browser's run reference) as a dict (see parse_rec), or None"""
    if not check_data(data):
        return None
    path = _data_source(data)
    if not path:
        return None
    st = os.stat(path)
    return _read_rec(os.path.abspath(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=8)
def _read_rec(path, mtime_ns, size):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".rec")]
        if not names:
            return None
        return parse_rec(z.read(names[0]).decode(errors="replace"), name=os.path.basename(names[0]))


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def parse_rec(text, name=""):
    """
    A run record as a dict:
      settings    {name: value} of the settings in SETTINGS that are reported
      iterations  DataFrame by iteration number (fields of FIELDS, lambda counts, accepted lambda, ...)
      lambdas     DataFrame of the lambda tests (iteration, lambda, fraction, phi, phi_ratio, meas, regul, failed)
      groups      DataFrame of phi by observation group at the start of each iteration (iteration x group)
      events      DataFrame of parameters reported with no effect or frozen (iteration, parameter, event)
      complete    True if the record reports the end of the optimisation
    """
    settings = {}
    for line in text.splitlines():
        for label, key in SETTINGS.items():
            if label in line and ":" in line and key not in settings:
                value = line.split(":", 1)[1].strip()
                settings[key] = value
    starts = [(m.start(), int(m.group(1))) for m in ITERATION.finditer(text)]
    rows, lambdas, groups, events = [], [], [], []
    for k, (start, it) in enumerate(starts):
        block = text[start: starts[k + 1][0] if k + 1 < len(starts) else len(text)]
        row = {"iteration": it}
        for key, pattern in FIELDS.items():
            m = re.search(pattern, block, re.IGNORECASE)
            row[key] = _number(m.group(1)) if m else np.nan
        seen = set()
        for g, value in GROUP.findall(block):          # the first block of contributions: before reweighting
            if g in seen:
                break
            seen.add(g)
            groups.append(dict(iteration=it, group=g, phi=_number(value)))
        events += [dict(iteration=it, parameter=p.lower(), event="no effect") for p in NO_EFFECT.findall(block)]
        events += [dict(iteration=it, parameter=p.lower(), event="frozen") for p in dict.fromkeys(FROZEN.findall(block))]
        for kind, value, par in MAX_CHANGE.findall(block):
            row[f"max_{kind}_change"] = _number(value)
            row[f"max_{kind}_parameter"] = par.lower()
        lines = block.splitlines()
        for i, line in enumerate(lines):
            m = LAMBDA.search(line)
            if not m:
                continue
            test = dict(iteration=it, test=len([t for t in lambdas if t["iteration"] == it]) + 1,
                        **{"lambda": _number(m.group(1))}, fraction=_number(m.group(2)) if m.group(2) else np.nan,
                        phi=np.nan, meas=np.nan, regul=np.nan, failed=False)
            for follow in lines[i + 1: i + 5]:
                if LAMBDA.search(follow):
                    break
                if "cannot be calculated" in follow or "run failure" in follow.lower():
                    test["failed"] = True
                for regex, key in ((PHI, "phi"), (MEAS, "meas"), (REGUL, "regul")):
                    mm = regex.search(follow)
                    if mm and np.isnan(test[key]):
                        test[key] = _number(mm.group(1))
            lambdas.append(test)
        rows.append(row)

    iterations = pd.DataFrame(rows).drop_duplicates("iteration", keep="first").set_index("iteration")
    lambdas = pd.DataFrame(lambdas, columns=["iteration", "test", "lambda", "fraction", "phi", "meas", "regul", "failed"])
    if len(iterations):
        lambdas["phi_ratio"] = lambdas.phi / lambdas.iteration.map(iterations.phi_start)
        iterations = _accepted(iterations, lambdas)
        calls = iterations.model_calls
        iterations["calls"] = calls.shift(-1) - calls            # model runs during the iteration
        iterations["lambda_runs"] = lambdas.groupby("iteration").size().reindex(iterations.index).fillna(0).astype(int)
        iterations["failed_lambda_runs"] = lambdas.groupby("iteration").failed.sum().reindex(iterations.index).fillna(0).astype(int)
        iterations["jacobian_runs"] = iterations.calls - iterations.lambda_runs
        ev = pd.DataFrame(events, columns=["iteration", "parameter", "event"])
        for e in ("no effect", "frozen"):
            iterations[e.replace(" ", "_")] = ev[ev.event == e].groupby("iteration").size().reindex(iterations.index).fillna(0).astype(int)
    return dict(
        name=name,
        settings=settings,
        iterations=iterations,
        lambdas=lambdas,
        groups=pd.DataFrame(groups, columns=["iteration", "group", "phi"]),
        events=pd.DataFrame(events, columns=["iteration", "parameter", "event"]),
        complete=bool(END.search(text)),
    )


def _accepted(iterations, lambdas):
    """
    The accepted lambda test of each iteration: the test whose measurement phi (or phi) is the next
    iteration's starting measurement phi. The record does not say which test was accepted.
    """
    iterations = iterations.copy()
    iterations["accepted_lambda"] = np.nan
    iterations["accepted_fraction"] = np.nan
    iterations["accepted_test"] = np.nan
    its = list(iterations.index)
    for a, b in zip(its, its[1:]):
        target = iterations.loc[b, "phi_measurement"]
        if np.isnan(target):
            target = iterations.loc[b, "phi_start"]
        tests = lambdas[(lambdas.iteration == a) & ~lambdas.failed]
        if np.isnan(target) or not len(tests):
            continue
        value = tests.meas.where(tests.meas.notna(), tests.phi)
        diff = (value - target).abs() / abs(target)
        if diff.min() < 1e-4:
            best = tests.loc[diff.idxmin()]
            iterations.loc[a, ["accepted_lambda", "accepted_fraction", "accepted_test"]] = \
                [best["lambda"], best.fraction, best.test]
    return iterations
