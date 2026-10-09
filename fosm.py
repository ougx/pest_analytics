"""
Linear (first-order, second-moment) parameter uncertainty from a PEST Jacobian.

The prior parameter covariance is diagonal, with standard deviations from the parameter
bounds; measurement noise is taken from the observation weights. The posterior covariance
is the Schur complement written in parameter space:

    C_post = (J^T Q J + C_prior^-1)^-1,   Q = diag(w^2) / noise scale

Jacobian columns of log-transformed parameters are in log10 space, so their standard
deviations are in log10 units as well.
"""
import numpy as np
import pandas as pd


def prior_std(par, nstd=4.0):
    """
    Prior standard deviation of each adjustable parameter: its bounds span nstd standard
    deviations, in log10 units for log-transformed parameters. Series by parameter name.
    """
    p = par[par.partrans.isin(["log", "none"])].set_index("parname")
    lo, hi = p.parlbnd.astype(float), p.parubnd.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        span = np.where(p.partrans == "log", np.log10(hi / lo), hi - lo)
    return pd.Series(span / float(nstd), index=p.index)


def measurement_rows(obs, weight_col="weight", include_regul=False):
    """weighted observations (Series of weights by name); regularization groups only if asked"""
    o = obs[obs[weight_col] > 0]
    if not include_regul:
        o = o[~o.obsgrp.str.startswith("regul")]
    return o.set_index("obsname")[weight_col].astype(float)


def prior_information_rows(obs, rei):
    """
    Weighted prior-information equations (Series of weights by name): the rows of a residuals
    file that are not observations. Their weights there include the regularization weight factor.
    """
    if rei is None or "weight" not in rei.columns:
        return pd.Series(dtype=float)
    r = rei[~rei.obsname.isin(obs.obsname) & (rei.weight > 0)]
    return r.set_index("obsname")["weight"].astype(float)


def noise_scale(obs, rei, weight_col="weight"):
    """
    phi / n of the weighted measurements (regularization excluded) at the given residuals:
    the factor by which the noise variance implied by the weights must grow to explain the fit.
    """
    w = measurement_rows(obs, weight_col)
    r = rei.set_index("obsname")["residual"].reindex(w.index)
    ok = r.notna()
    return float(((w[ok] * r[ok]) ** 2).sum() / max(ok.sum(), 1))


def linear_uncertainty(jco, par, rows, nstd=4.0, scale=1.0, measured=None):
    """
    Prior and posterior standard deviations of the adjustable parameters in jco.
    rows: weights by row name (measurement_rows, optionally with prior_information_rows);
    scale: noise variance multiplier (noise_scale, or 1 to take the weights as they are).
    Returns a DataFrame by parameter: pargrp, partrans, prior_std, post_std, reduction
    (1 - post/prior) and sensitive (False where the Jacobian column is all zero in the
    measured rows, e.g. after a failed derivative run, so the measurements cannot reduce that
    parameter's uncertainty). measured: names of the measurement rows (default: all rows).
    """
    sd = prior_std(par, nstd)
    sd = sd[sd.index.isin(jco.columns) & np.isfinite(sd) & (sd > 0)]
    rows = rows[rows.index.isin(jco.index)]
    J = jco.loc[rows.index, sd.index].values
    WJ = J * (rows.values / np.sqrt(scale))[:, None]
    post = np.linalg.inv(WJ.T @ WJ + np.diag(1 / sd.values**2))
    post_std = np.sqrt(np.clip(np.diag(post), 0, None))
    info = par.set_index("parname").loc[sd.index]
    return pd.DataFrame(dict(
        pargrp=info.pargrp.values,
        partrans=info.partrans.values,
        prior_std=sd.values,
        post_std=post_std,
        reduction=1 - post_std / sd.values,
        sensitive=(np.abs(J[rows.index.isin(rows.index if measured is None else measured)]).sum(axis=0) > 0),
    ), index=sd.index)
