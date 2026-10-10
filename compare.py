"""
Comparing the setup of two PEST runs: what changed in the control file between a reference
run and another run. The control files are read and compared with makepst (makepst.diff:
parameters, observations, prior information, parameter groups, control values, ++ options,
template / instruction pairs, command lines and the comment lines at the top). Observation
weights are also compared numerically here, with the times from obs.csv, for plotting.
"""
import functools
import os
import zipfile

import numpy as np
import pandas as pd

from utils import _data_source, check_data

# direction of a comparison between the reference (loaded) run and another run
FORWARD, BACKWARD = "Reference → other run", "Other run → reference"

try:
    import makepst
    from makepst.diff import compare as _makepst_compare
except ImportError:          # the Compare pages say so
    makepst = None


def run_pst(data):
    """the control file of a run (the browser's run reference) as a makepst.Pst, or None"""
    if makepst is None:
        raise ImportError("the Compare pages need makepst (pip install makepst)")
    if not check_data(data):
        return None
    path = _data_source(data)
    if not path:
        return None
    st = os.stat(path)
    return _read_pst(os.path.abspath(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=8)
def _read_pst(path, mtime_ns, size):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".pst")]
        if not names:
            return None
        text = z.read(names[0]).decode(errors="replace")
    return makepst.from_text(text.replace("\r\n", "\n"))


def setup_diff(pst_from, pst_to):
    """makepst.diff.Diff from one control file to the other"""
    return _makepst_compare(pst_from, pst_to)


def summary_counts(pst_from, pst_to):
    """headline counts of two control files: DataFrame by item with columns from, to and change"""
    def counts(p):
        trans = p.par.PARTRANS.astype(str).str.lower()
        return {
            "Observations": len(p.obs),
            "Weighted observations": int((p.obs.WEIGHT.astype(float) > 0).sum()),
            "Observation groups": p.obs.OBGNME.nunique(),
            "Parameters": len(p.par),
            "Adjustable (log / none)": int(trans.isin(["log", "none"]).sum()),
            "Tied": int((trans == "tied").sum()),
            "Fixed": int((trans == "fixed").sum()),
            "Parameter groups": p.par.PARGP.nunique(),
            "Prior-information equations": len(p.prior),
        }
    df = pd.DataFrame({"from": counts(pst_from), "to": counts(pst_to)})
    df["change"] = df["to"] - df["from"]
    return df


def change_counts(df, key, groups):
    """
    A makepst diff table summarized: rows per (group, change, column) with the count and the first
    few names. groups: Series mapping names (lower case) to their group.
    """
    if not len(df):
        return pd.DataFrame(columns=["group", "change", "column", "count", "examples"])
    d = df.assign(group=df[key].astype(str).str.lower().map(groups).fillna(""))
    out = d.groupby(["group", "change", "column"], sort=False)[key].agg(
        count="size", examples=lambda s: ", ".join(s.astype(str).head(5)) + (", ..." if len(s) > 5 else ""))
    return out.reset_index()


def observation_changes(obs_from, obs_to):
    """
    Observations matched by name, with group, value and weight in both runs (_from / _to), the weight
    ratio and a status (added, removed, changed, same). obs_from / obs_to are the app's parsed
    observation data; time (from obs.csv) is taken from whichever run has it.
    """
    cols = ["obsname", "obsgrp", "obsval", "weight"]
    m = obs_from[cols].merge(obs_to[cols], on="obsname", how="outer", suffixes=("_from", "_to"), indicator=True)
    if "time" in obs_to.columns or "time" in obs_from.columns:
        times = pd.concat([o.set_index("obsname")["time"] for o in (obs_to, obs_from) if "time" in o.columns])
        m["time"] = m.obsname.map(times[~times.index.duplicated()])
    m["status"] = m.pop("_merge").map({"left_only": "removed", "right_only": "added", "both": "same"}).astype(str)
    both = m.status == "same"
    changed = both & ~(np.isclose(m.weight_from, m.weight_to, rtol=1e-6)
                       & np.isclose(m.obsval_from, m.obsval_to, rtol=1e-6)
                       & (m.obsgrp_from == m.obsgrp_to))
    m.loc[changed, "status"] = "changed"
    with np.errstate(divide="ignore", invalid="ignore"):
        m["weight_ratio"] = m.weight_to / m.weight_from
    m["obsgrp"] = m.obsgrp_to.fillna(m.obsgrp_from)
    return m.set_index("obsname")


def observation_group_summary(changes):
    """per observation group: observations added, removed, with changed weights, and the weight ratios"""
    def summarize(g):
        both = g[g.status.isin(["same", "changed"])]
        w = both[~np.isclose(both.weight_from, both.weight_to, rtol=1e-6)]
        ratio = w.weight_ratio[np.isfinite(w.weight_ratio) & (w.weight_ratio > 0)]
        return pd.Series(dict(
            observations=len(both) + int((g.status == "added").sum()),
            added=int((g.status == "added").sum()),
            removed=int((g.status == "removed").sum()),
            weight_changed=len(w),
            set_to_zero=int(((w.weight_to == 0) & (w.weight_from > 0)).sum()),
            from_zero=int(((w.weight_from == 0) & (w.weight_to > 0)).sum()),
            ratio_min=ratio.min() if len(ratio) else np.nan,
            ratio_median=ratio.median() if len(ratio) else np.nan,
            ratio_max=ratio.max() if len(ratio) else np.nan,
            value_changed=int((~np.isclose(both.obsval_from, both.obsval_to, rtol=1e-6)).sum()),
        ))
    return changes.groupby("obsgrp").apply(summarize, include_groups=False)


# ---------------------------------------------------------------------------
# fit: residuals of the two runs scored with the same weights
# ---------------------------------------------------------------------------

BEST, LAST = "Best (lowest φ)", "Last"
WEIGHTS = ["Reference run's weights", "Other run's weights", "Each run's own weights", "weight_orig (obs.csv)"]


def is_regul(groups):
    return groups.astype(str).str.startswith("regul")


def measurement_phi(rei, weight="weight"):
    """sum of (w r)^2 of the weighted observations, regularization excluded"""
    df = rei[(rei[weight] > 0) & ~is_regul(rei.obsgrp)]
    return float(((df[weight] * df.residual) ** 2).sum())


def pick_iteration(reis, choice):
    """iteration number for a choice: BEST (lowest measurement phi, the run's own weights), LAST or a number"""
    if not reis:
        return None
    if choice == LAST:
        return max(reis)
    if choice in (None, BEST):
        return min(reis, key=lambda i: (measurement_phi(reis[i]), -i))
    return int(choice) if int(choice) in reis else None


def run_weights(choice, obs_ref, obs_other, to_is_ref):
    """
    (weights for the 'from' run, weights for the 'to' run), Series by obsname, for a WEIGHTS choice.
    to_is_ref: True when the reference run is the 'to' side of the comparison.
    """
    ref = obs_ref.set_index("obsname")
    other = obs_other.set_index("obsname")
    if choice == WEIGHTS[1]:
        return other.weight, other.weight
    if choice == WEIGHTS[2]:
        return (other.weight, ref.weight) if to_is_ref else (ref.weight, other.weight)
    if choice == WEIGHTS[3]:
        src = ref if "weight_orig" in ref.columns else other if "weight_orig" in other.columns else None
        if src is None:
            raise ValueError("neither run has a weight_orig column (from obs.csv)")
        return src.weight_orig, src.weight_orig
    return ref.weight, ref.weight


def paired_residuals(rei_from, rei_to, w_from, w_to, groups):
    """
    Observations in both runs (regularization excluded): residual and weight of each side and the group.
    groups: Series by obsname (e.g. a grouping column of the reference run's observation data).
    """
    a = rei_from[~is_regul(rei_from.obsgrp)].set_index("obsname").residual
    b = rei_to[~is_regul(rei_to.obsgrp)].set_index("obsname").residual
    names = a.index.intersection(b.index).intersection(groups.dropna().index)
    return pd.DataFrame(dict(
        residual_from=a.loc[names], residual_to=b.loc[names],
        weight_from=w_from.reindex(names).fillna(0).astype(float),
        weight_to=w_to.reindex(names).fillna(0).astype(float),
        group=groups.loc[names].astype(str).str.lower(),
    ))


def fit_by_group(paired):
    """
    Per group: phi of each side (sum of (w r)^2 with its weights), and n, RMSE and mean residual
    (measured units) over the observations weighted on both sides; changes are to - from.
    """
    p = paired.assign(phi_from=(paired.weight_from * paired.residual_from) ** 2,
                      phi_to=(paired.weight_to * paired.residual_to) ** 2,
                      r2_from=paired.residual_from ** 2, r2_to=paired.residual_to ** 2)
    phi = p.groupby("group")[["phi_from", "phi_to"]].sum()
    both = p[(p.weight_from > 0) & (p.weight_to > 0)]
    g = both.groupby("group")
    stats = pd.DataFrame(dict(
        n=g.size(),
        rmse_from=np.sqrt(g.r2_from.mean()),
        rmse_to=np.sqrt(g.r2_to.mean()),
        mean_from=g.residual_from.mean(),
        mean_to=g.residual_to.mean(),
    ))
    out = phi.join(stats, how="left")
    out["n"] = out.n.fillna(0).astype(int)
    out["dphi"] = out.phi_to - out.phi_from
    out["drmse"] = out.rmse_to - out.rmse_from
    return out.sort_values("phi_from", ascending=False)


# ---------------------------------------------------------------------------
# parameters: values of the two runs matched by name
# ---------------------------------------------------------------------------

def parameter_values(par, pars, iteration):
    """
    Parameter values of a run at an iteration (0: the initial values of the control file), with
    transform, group and bounds; None if that iteration's .par file is missing.
    """
    p = par.set_index("parname")[["partrans", "pargrp", "parval0", "parlbnd", "parubnd"]].astype(
        {"parval0": float, "parlbnd": float, "parubnd": float})
    if iteration == 0:
        value = p.parval0
    elif iteration in pars:
        value = pars[iteration].set_index("parname").parval.astype(float).reindex(p.index)
    else:
        return None
    tol = 1e-6 * np.maximum(value.abs(), 1e-30)
    return p.assign(value=value, at_bound=(value - p.parlbnd <= tol) | (p.parubnd - value <= tol))


def paired_parameters(values_from, values_to):
    """
    Parameters in both runs: values, bounds flags and the change, log10(to/from) when both runs
    log-transform the parameter (and the values are positive), (to - from)/|from| otherwise.
    """
    m = values_from.join(values_to, how="inner", lsuffix="_from", rsuffix="_to")
    log = (m.partrans_from == "log") & (m.partrans_to == "log") & (m.value_from > 0) & (m.value_to > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m["change"] = np.where(log, np.log10(m.value_to / m.value_from),
                               (m.value_to - m.value_from) / m.value_from.abs())
    m["log"] = log
    m["pargrp"] = m.pargrp_to
    return m
