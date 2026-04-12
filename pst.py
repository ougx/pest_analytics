import numpy as np
import pandas as pd

def get_weighted_jco(obs, jco0, filter_insensitivetive=True):
    """
    product of observation weights and Jacobian matrix
    jco = pd.DataFrame(j, index=observations, columns=parameters)
    """
    mask = jco0.index.isin(obs.obsname)
    jco = jco0.loc[mask, ]
    WJ = jco * obs.set_index("obsname").loc[jco.index, "weight"].values[:, None]
    if filter_insensitivetive:
        csplimit = 1e-10
        nobs_useful = (obs.weight > 0).sum()
        csp = (WJ**2).sum() ** 0.5 / nobs_useful
        mask = WJ.columns.isin(csp[csp>csplimit].index) # filter non-sensitive parameters
        WJ = WJ.loc[:,mask]
    return WJ.copy()

def get_cov(obs, jco0, filter_insensitivetive=True):
    """
    Compute parameter covariance matrix: cov = (J^T J)^-1
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive)
    J2 = np.matmul(WJ.T, WJ)
    try:
        cov = np.linalg.inv(J2)
    except:
        cov = np.linalg.pinv(J2)
    return pd.DataFrame(cov, index=WJ.columns, columns=WJ.columns)


def get_cor(obs, jco0, filter_insensitivetive=True):
    """
    Compute parameter correlation matrix: C = cov_{ij} / sqrt(cov_{ii} * cov_{jj})
    """
    cov = get_cov(obs, jco0, filter_insensitivetive)
    std = np.sqrt(np.diag(cov))
    cor = cov / np.outer(std, std)
    return pd.DataFrame(cor, index=cov.index, columns=cov.columns)


def get_obssim(obs, jco0, filter_insensitivetive=True):
    """
    Compute observation similarity matrix: S = J J^T
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive)
    return WJ @ WJ.T



def get_obscor(obs, jco0, filter_insensitivetive=True):
    """
    Compute observation correlation (normalized observation similarity)
    """
    So = get_obssim(obs, jco0, filter_insensitivetive)
    d = np.sqrt(np.diag(So))
    return So / np.outer(d, d)


def get_obsinflu(obs, jco0, filter_insensitivetive=True):
    """
    Compute observation leverage using the hat matrix: H = J (J^T J)^-1 J^T
    """
    WJ = get_weighted_jco(obs, jco0, filter_insensitivetive)
    cov = get_cov(obs, jco0, filter_insensitivetive)
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
