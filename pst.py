import numpy as np
import pandas as pd

def get_weighted_jco(obs, jco0, filter_insensitivetive=True, remove_regul=True, weight_col="weight"):
    """
    product of observation weights and Jacobian matrix
    jco = pd.DataFrame(j, index=observations, columns=parameters)
    """
    mask = jco0.index.isin(obs.obsname)
    jco = jco0.loc[mask, ]
    WJ = jco * obs.set_index("obsname").loc[jco.index, weight_col].values[:, None]
    if filter_insensitivetive:
        csplimit = 1e-10
        nobs_useful = (obs[weight_col] > 0).sum()
        csp = (WJ**2).sum() ** 0.5 / nobs_useful
        mask = WJ.columns.isin(csp[csp>csplimit].index) # filter non-sensitive parameters
        WJ = WJ.loc[:,mask]
    if remove_regul:
        mask = obs.set_index("obsname").loc[WJ.index].obsgrp.str.startswith("regul")
        WJ = WJ.loc[~mask]
    return WJ.copy()

def get_cov(obs, jco0, filter_insensitivetive=True, weight_col="weight"):
    """
    Compute parameter covariance matrix: cov = (J^T J)^-1
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive, weight_col=weight_col)
    J2 = np.matmul(WJ.T, WJ)
    try:
        cov = np.linalg.inv(J2)
    except:
        cov = np.linalg.pinv(J2)
    return pd.DataFrame(cov, index=WJ.columns, columns=WJ.columns)


def get_cor(obs, jco0, filter_insensitivetive=True, weight_col="weight"):
    """
    Compute parameter correlation matrix: C = cov_{ij} / sqrt(cov_{ii} * cov_{jj})
    """
    cov = get_cov(obs, jco0, filter_insensitivetive, weight_col)
    std = np.sqrt(np.diag(cov))
    cor = cov / np.outer(std, std)
    return pd.DataFrame(cor, index=cov.index, columns=cov.columns)


def get_obssim(obs, jco0, filter_insensitivetive=True, weight_col="weight"):
    """
    Compute observation similarity matrix: S = J J^T
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive, weight_col=weight_col)
    return WJ @ WJ.T



def get_obscor(obs, jco0, filter_insensitivetive=True, weight_col="weight"):
    """
    Compute observation correlation (normalized observation similarity)
    """
    So = get_obssim(obs, jco0, filter_insensitivetive, weight_col)
    d = np.sqrt(np.diag(So))
    return So / np.outer(d, d)


def get_obsinflu(obs, jco0, filter_insensitivetive=True, weight_col="weight"):
    """
    Compute observation leverage using the hat matrix: H = J (J^T J)^-1 J^T
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive, weight_col=weight_col)
    cov = get_cov(obs, jco0, filter_insensitivetive, weight_col)
    H =  WJ @ cov @ WJ.T
    leverage = np.diag(H)
    return pd.Series(leverage, index=WJ.index).sort_values(ascending=False)

def get_dataworth(obs, jco0, filter_insensitivetive=True):
    """
    Compute observation data worth by leave-one-out analysis
    """
    Jm = J.values
    base = np.trace(np.linalg.pinv(Jm @ Jm.T))

    worth = {}
    for j, obs in enumerate(J.columns):
        J_sub = np.delete(Jm, j, axis=1)
        val = np.trace(np.linalg.pinv(J_sub @ J_sub.T))
        worth[obs] = val - base

    return pd.Series(worth).sort_values(ascending=False)


def get_obs_pull(obs, jco0, rei, weight_col="weight", filter_insensitivetive=True):
    """
    Pull of each observation on each parameter: P_ij = 2 w_i^2 r_i J_ij = -dphi_i/dp_j
    Positive: observation i wants parameter j to increase; negative: to decrease.
    rei must hold residuals (measured - modelled) at the parameters where jco0 was computed.
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive, weight_col=weight_col)
    w = obs.set_index("obsname").loc[WJ.index, weight_col]
    r = rei.set_index("obsname")["residual"].reindex(WJ.index).fillna(0)
    wr = (w * r).values
    pull = WJ * (2 * wr[:, None])
    return pull.loc[wr != 0]


def get_par_tension(pull):
    """
    Summarize observation (or group) pulls per parameter.
    pull: DataFrame (observations/groups x parameters) from get_obs_pull.
    tension = sum|P| - |sum P| = 2 min(up, |down|): pull that cancels out.
    """
    up = pull.clip(lower=0).sum()
    down = pull.clip(upper=0).sum()
    tension = 2 * np.minimum(up, -down)
    ratio = np.minimum(up, -down) / np.maximum(up, -down).replace(0, np.nan)
    return pd.DataFrame(dict(up=up, down=down, net=up + down, tension=tension, ratio=ratio.fillna(0)))


def get_jco_rei(reis, iiter):
    """
    Residuals at the parameters where .jco.N was computed. PEST computes .jco.N at the
    parameters of iteration N-1, so pair it with .rei.(N-1) (the .hp file for N = 1).
    Falls back to .rei.N if .rei.(N-1) is not available; returns None if neither is.
    """
    return reis.get(iiter - 1, reis.get(iiter))


def parameter_change(pars):
    """
    Change of each adjustable parameter from its initial value, by name: log10(p/p0) for
    log-transformed parameters and (p - p0)/|p0| for untransformed ones (fixed and tied excluded).
    pars: a .par iteration merged with the .pst parameter data (parval, parval0, partrans).
    """
    df = pars.set_index("parname")
    df = df[df.partrans.isin(["log", "none"])]
    p, p0 = df.parval.astype(float), df.parval0.astype(float)
    relative = (p - p0) / p0.abs().where(p0 != 0)
    return pd.Series(np.where(df.partrans == "log", np.log10(p / p0), relative), index=df.index)
