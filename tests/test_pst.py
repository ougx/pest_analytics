"""Unit tests for pst.py — analytical / statistical helpers.

All fixtures come from conftest.py which loads testdata/example_pst.zip.
Tests verify mathematical properties (symmetry, bounds, etc.) that must hold
for any valid PEST dataset.
"""
import numpy as np
import pandas as pd
import pytest

from pst import (
    get_weighted_jco,
    get_cov,
    get_cor,
    get_obssim,
    get_obscor,
    get_obsinflu,
    get_obs_pull,
    get_par_tension,
    get_jco_rei,
)


# ---------------------------------------------------------------------------
# get_weighted_jco
# ---------------------------------------------------------------------------

def test_wj_has_nonzero_shape(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df, filter_insensitivetive=False, remove_regul=False)
    assert WJ.shape[0] > 0
    assert WJ.shape[1] > 0

def test_wj_rows_are_subset_of_jco_index(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df, filter_insensitivetive=False, remove_regul=False)
    assert set(WJ.index).issubset(set(jco_df.index))

def test_wj_removes_regul_rows(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df, filter_insensitivetive=False, remove_regul=True)
    regul_obs = obs_df[obs_df.obsgrp.str.startswith("regul")]["obsname"].tolist()
    assert not any(o in WJ.index for o in regul_obs)

def test_wj_zero_weight_rows_are_all_zero(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df, filter_insensitivetive=False, remove_regul=False)
    zero_w = obs_df[obs_df.weight == 0]["obsname"]
    in_both = zero_w[zero_w.isin(WJ.index)]
    for obsname in in_both:
        assert (WJ.loc[obsname] == 0).all()

def test_wj_weight_applied_correctly(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df, filter_insensitivetive=False, remove_regul=False)
    # Pick one obs present in both with positive weight
    sample = obs_df[obs_df.obsname.isin(jco_df.index) & (obs_df.weight > 0)].iloc[0]
    obsname, w = sample["obsname"], sample["weight"]
    if obsname in WJ.index:
        np.testing.assert_allclose(
            WJ.loc[obsname].values,
            w * jco_df.loc[obsname].values,
            atol=1e-10,
        )

def test_wj_filters_all_zero_columns(obs_df):
    zero_jco = pd.DataFrame(
        np.zeros((len(obs_df), 3)),
        index=obs_df["obsname"],
        columns=["p1", "p2", "p3"],
    )
    WJ = get_weighted_jco(obs_df, zero_jco, filter_insensitivetive=True, remove_regul=False)
    assert WJ.shape[1] == 0


# ---------------------------------------------------------------------------
# get_cov
# ---------------------------------------------------------------------------

def test_cov_is_square(obs_df, jco_df):
    cov = get_cov(obs_df, jco_df)
    assert cov.shape[0] == cov.shape[1]

def test_cov_is_symmetric(obs_df, jco_df):
    cov = get_cov(obs_df, jco_df)
    np.testing.assert_allclose(cov.values, cov.values.T, atol=1e-8)

def test_cov_diagonal_positive(obs_df, jco_df):
    cov = get_cov(obs_df, jco_df)
    assert (np.diag(cov.values) > 0).all()

def test_cov_index_matches_columns(obs_df, jco_df):
    cov = get_cov(obs_df, jco_df)
    assert list(cov.index) == list(cov.columns)


# ---------------------------------------------------------------------------
# get_cor
# ---------------------------------------------------------------------------

def test_cor_diagonal_ones(obs_df, jco_df):
    cor = get_cor(obs_df, jco_df)
    np.testing.assert_allclose(np.diag(cor.values), 1.0, atol=1e-8)

def test_cor_is_symmetric(obs_df, jco_df):
    cor = get_cor(obs_df, jco_df)
    np.testing.assert_allclose(cor.values, cor.values.T, atol=1e-8)

def test_cor_bounded(obs_df, jco_df):
    cor = get_cor(obs_df, jco_df)
    assert (cor.values >= -1.0 - 1e-8).all()
    assert (cor.values <=  1.0 + 1e-8).all()

def test_cor_is_square(obs_df, jco_df):
    cor = get_cor(obs_df, jco_df)
    assert cor.shape[0] == cor.shape[1]


# ---------------------------------------------------------------------------
# get_obssim
# ---------------------------------------------------------------------------

def test_obssim_is_symmetric(obs_df, jco_df):
    So = get_obssim(obs_df, jco_df)
    np.testing.assert_allclose(So.values, So.values.T, atol=1e-8)

def test_obssim_shape_matches_active_obs(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df)
    So = get_obssim(obs_df, jco_df)
    assert So.shape == (len(WJ), len(WJ))

def test_obssim_diagonal_nonneg(obs_df, jco_df):
    So = get_obssim(obs_df, jco_df)
    assert (np.diag(So.values) >= 0).all()


# ---------------------------------------------------------------------------
# get_obscor
# ---------------------------------------------------------------------------

def test_obscor_diagonal_ones(obs_df, jco_df):
    Sc = get_obscor(obs_df, jco_df)
    diag = np.diag(Sc.values)
    # Skip NaN entries — they arise from zero-sensitivity observations (0/0)
    valid = diag[~np.isnan(diag)]
    assert len(valid) > 0
    np.testing.assert_allclose(valid, 1.0, atol=1e-8)

def test_obscor_is_symmetric(obs_df, jco_df):
    Sc = get_obscor(obs_df, jco_df)
    # NaN != NaN, so compare only where both sides are finite
    v = Sc.values
    finite = np.isfinite(v) & np.isfinite(v.T)
    np.testing.assert_allclose(v[finite], v.T[finite], atol=1e-8)

def test_obscor_bounded(obs_df, jco_df):
    Sc = get_obscor(obs_df, jco_df)
    valid = Sc.values[~np.isnan(Sc.values)]
    assert (valid >= -1.0 - 1e-8).all()
    assert (valid <=  1.0 + 1e-8).all()


# ---------------------------------------------------------------------------
# get_obsinflu (leverage)
# ---------------------------------------------------------------------------

def test_obsinflu_is_series(obs_df, jco_df):
    lev = get_obsinflu(obs_df, jco_df)
    assert isinstance(lev, pd.Series)

def test_obsinflu_nonneg(obs_df, jco_df):
    lev = get_obsinflu(obs_df, jco_df)
    assert (lev.values >= -1e-8).all()

def test_obsinflu_sorted_descending(obs_df, jco_df):
    lev = get_obsinflu(obs_df, jco_df)
    assert lev.is_monotonic_decreasing

def test_obsinflu_sum_leq_nparams(obs_df, jco_df):
    WJ = get_weighted_jco(obs_df, jco_df)
    lev = get_obsinflu(obs_df, jco_df)
    # trace(H) = rank(J) ≤ nparams
    assert lev.sum() <= WJ.shape[1] + 1e-6


# ---------------------------------------------------------------------------
# get_obs_pull / get_par_tension
# ---------------------------------------------------------------------------

def test_pull_matches_definition(obs_df, jcos_dict, reis_dict):
    pull = get_obs_pull(obs_df, jcos_dict[1], reis_dict[0])
    o = pull.index[0]; p = pull.columns[0]
    w = obs_df.set_index("obsname").loc[o, "weight"]
    r = reis_dict[0].set_index("obsname").loc[o, "residual"]
    assert np.isclose(pull.loc[o, p], 2 * w**2 * r * jcos_dict[1].loc[o, p])

def test_pull_sum_is_negative_phi_gradient(obs_df, jcos_dict, reis_dict):
    """sum_i P_ij = -dphi/dp_j = 2 J^T Q r"""
    pull = get_obs_pull(obs_df, jcos_dict[1], reis_dict[0], filter_insensitivetive=False)
    WJ = get_weighted_jco(obs_df, jcos_dict[1], filter_insensitivetive=False)
    w = obs_df.set_index("obsname").loc[WJ.index, "weight"].values
    r = reis_dict[0].set_index("obsname").loc[WJ.index, "residual"].values
    np.testing.assert_allclose(pull.sum().values, 2 * WJ.values.T @ (w * r), rtol=1e-8)

def test_pull_drops_zero_weight_rows(obs_df, jcos_dict, reis_dict):
    pull = get_obs_pull(obs_df, jcos_dict[1], reis_dict[0])
    zero = obs_df[obs_df.weight == 0].obsname
    assert not pull.index.isin(zero).any()

def test_tension_identity():
    pull = pd.DataFrame({"k1": [3.0, -1.0, -0.5], "k2": [1.0, 2.0, 0.0], "k3": [-2.0, 2.0, 0.0]})
    T = get_par_tension(pull)
    np.testing.assert_allclose(T.tension, pull.abs().sum() - pull.sum().abs())
    assert T.loc["k2", "tension"] == 0            # all pulls agree
    assert T.loc["k3", "ratio"] == 1              # perfectly balanced tug-of-war
    assert T.loc["k1", "net"] == 1.5


def test_jco_rei_pairs_with_previous_iteration():
    reis = {0: "hp", 1: "rei1", 2: "rei2"}
    assert get_jco_rei(reis, 1) == "hp"
    assert get_jco_rei(reis, 2) == "rei1"
    assert get_jco_rei(reis, 3) == "rei2"      # no rei.3 needed for jco.3
    assert get_jco_rei({2: "rei2"}, 2) == "rei2"  # fallback when rei.(N-1) missing
    assert get_jco_rei({}, 1) is None


# ---------------------------------------------------------------------------
# parameter_change (Phi Evolution's parameter variance)
# ---------------------------------------------------------------------------

def test_parameter_change_by_name_and_transform():
    from pst import parameter_change
    pars = pd.DataFrame(dict(
        parname=["b", "a", "c", "d", "e"],
        parval=[10.0, 2.0, 3.0, 5.0, 7.0],
        parval0=[1.0, 4.0, 3.0, 1.0, 1.0],
        partrans=["log", "none", "log", "fixed", "tied"],
    ))
    change = parameter_change(pars)
    assert sorted(change.index) == ["a", "b", "c"]          # fixed and tied excluded
    assert np.isclose(change["b"], 1.0)                      # log10(10/1)
    assert np.isclose(change["a"], -0.5)                     # (2 - 4)/4
    assert change["c"] == 0

def test_parameter_change_real_iteration(pars_dict, par_df):
    from pst import parameter_change
    change = parameter_change(pars_dict[6])
    assert len(change) == int(par_df.partrans.isin(["log", "none"]).sum())
    assert np.isfinite(change).all()
