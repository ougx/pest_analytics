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
