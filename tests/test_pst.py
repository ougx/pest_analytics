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
    get_identifiability,
    eigenvalue_ratios,
    jacobian_sensitivity,
    rank_sensitivity,
    composite_sensitivity,
    get_obs_pull,
    get_par_tension,
    get_jco_rei,
    jacobian_step,
    step_contribution,
    linear_phi_change,
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
# get_identifiability
# ---------------------------------------------------------------------------

def test_identifiability_bounds_and_count(obs_df, jco_df):
    ident, nkept, nvec = get_identifiability(obs_df, jco_df, 0.0)
    assert nkept == nvec == len(ident)        # every direction kept: all parameters fully identifiable
    np.testing.assert_allclose(ident.values, 1.0, atol=1e-8)

def test_composite_sensitivity():
    obs = pd.DataFrame(dict(obsname=["h1", "h2", "f1", "z", "r"], weight=[1.0, 1.0, 2.0, 0.0, 9.0],
                            obsgrp=["head", "head", "flow", "head", "regul_a"]))
    jco = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [5.0, 5.0], [7.0, 7.0]],
                       index=["h1", "h2", "f1", "z", "r"], columns=["a", "b"])
    css = composite_sensitivity(obs, jco)
    # a: (1 + 1 + 4) / 3 weighted, non-regularization observations; b: no sensitivity
    assert css.css.to_dict() == pytest.approx({"a": np.sqrt(2.0), "b": 0.0})
    assert css.loc["a", ["head", "flow"]].tolist() == pytest.approx([2 / 6, 4 / 6])
    assert css.loc["b", ["head", "flow"]].tolist() == [0, 0]


def test_eigenvalue_ratios_match_identifiability(obs_df, jco_df):
    ratios = eigenvalue_ratios(obs_df, jco_df)
    assert ratios.iloc[0] == pytest.approx(1.0) and ratios.is_monotonic_decreasing
    for thresh in (1e-3, 1e-6):
        assert get_identifiability(obs_df, jco_df, thresh)[1] == int((ratios >= thresh).sum())


def test_identifiability_cutoff_on_eigenvalue_ratio():
    # weighted Jacobian with singular values 1 and 0.01 -> eigenvalue ratio 1e-4
    obs = pd.DataFrame(dict(obsname=["o1", "o2"], weight=[1.0, 1.0], obsgrp=["g", "g"]))
    jco = pd.DataFrame([[1.0, 0.0], [0.0, 0.01]], index=["o1", "o2"], columns=["a", "b"])
    ident, nkept, nvec = get_identifiability(obs, jco, 1e-3)
    assert (nkept, nvec) == (1, 2)
    assert ident.round(8).to_dict() == {"a": 1.0, "b": 0.0}
    assert get_identifiability(obs, jco, 1e-5)[1] == 2


# ---------------------------------------------------------------------------
# jacobian_sensitivity / rank_sensitivity
# ---------------------------------------------------------------------------

def _sen_data():
    obs = pd.DataFrame(dict(obsname=["o1", "o2", "o3"], weight=[1.0, 2.0, 0.0], obsgrp=["g"] * 3))
    jco = pd.DataFrame([[1.0, -3.0], [-1.0, 0.5], [5.0, 5.0], [9.0, 9.0]],
                       index=["o1", "o2", "o3", "prior1"], columns=["a", "b"])
    return obs, {2: jco}

def test_jacobian_sensitivity_by_observation():
    obs, jcos = _sen_data()
    mag, signed = jacobian_sensitivity(jcos, obs, "L1 Norm", parameters=["a", "b"])
    assert list(mag.columns) == ["Jacobian 2"]
    assert list(mag.index) == ["o1", "o2", "o3"]             # rows not in the observation data are left out
    assert mag["Jacobian 2"].tolist() == [4.0, 3.0, 0.0]     # sum |w J| over a and b
    assert signed["Jacobian 2"].tolist() == [-2.0, -1.0, 0.0]

def test_jacobian_sensitivity_by_parameter():
    obs, jcos = _sen_data()
    mag, signed = jacobian_sensitivity(jcos, obs, "L2 Norm", observations=["o1", "o2"])
    assert mag["Jacobian 2"].to_dict() == {"a": 5.0, "b": 10.0}   # sum (w J)^2, not yet rooted
    assert signed["Jacobian 2"].to_dict() == {"a": -1.0, "b": -2.0}

def test_rank_sensitivity_top_n_and_symbols():
    mag = pd.DataFrame({"Jacobian 1": [1.0, 4.0, 9.0], "Jacobian 2": [16.0, 4.0, 0.0]}, index=["x", "y", "z"])
    signed = pd.DataFrame({"Jacobian 1": [1.0, -1.0, 1.0], "Jacobian 2": [-1.0, 1.0, 1.0]}, index=["x", "y", "z"])
    top, symbols = rank_sensitivity(mag, signed, "L2 Norm", 2)
    # rooted: x (1, 4), y (2, 2), z (3, 0); scaled by column max: x 1/3+1, y 2/3+1/2, z 1+0
    assert list(top.index) == ["x", "y"]
    assert top.loc["x"].tolist() == [1.0, 4.0]
    assert symbols.values.tolist() == [["triangle-up", "triangle-down"], ["triangle-down", "triangle-up"]]


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


# ---------------------------------------------------------------------------
# iteration_step / step_agreement (Observation Conflict's next step check)
# ---------------------------------------------------------------------------

def _par_iteration(values):
    return pd.DataFrame(dict(
        parname=["a", "b", "c"], parval=values, parval0=[1.0, 1.0, 1.0],
        partrans=["log", "log", "none"], parlbnd=[0.1, 0.1, -10.0], parubnd=[100.0, 100.0, 10.0],
    ))

def test_iteration_step_from_previous_iteration():
    from pst import iteration_step
    pars = {1: _par_iteration([10.0, 1.0, 2.0]), 2: _par_iteration([100.0, 0.1, 3.0])}
    step, at_bound = iteration_step(pars, 2)
    assert np.isclose(step["a"], 1.0) and np.isclose(step["b"], -1.0)   # from .par.1, not parval0
    assert np.isclose(step["c"], 0.5)                                     # (3 - 2)/2
    assert at_bound.to_dict() == {"a": True, "b": True, "c": False}
    step, _ = iteration_step(pars, 1)
    assert np.isclose(step["a"], 1.0)                                     # first iteration: from parval0
    assert iteration_step(pars, 3) == (None, None)
    assert iteration_step({2: pars[2]}, 2) == (None, None)

def test_step_agreement_counts_moved_parameters():
    from pst import step_agreement
    net = pd.Series(dict(a=1.0, b=-2.0, c=3.0, d=1.0, e=0.0))
    step = pd.Series(dict(a=0.5, b=0.1, c=0.0, d=-1e-9, e=1.0))
    assert step_agreement(net, step) == (1, 2)   # a agrees, b disagrees; c, d did not move; e has no pull

def test_iteration_step_real_data(pars_dict, par_df):
    from pst import iteration_step
    step, at_bound = iteration_step(pars_dict, 6)
    assert len(step) == int(par_df.partrans.isin(["log", "none"]).sum())
    assert np.isfinite(step).all() and at_bound.dtype == bool


def test_step_contribution_on_a_linear_model():
    """for an exactly linear model, the contributions add up to the first-order change and
    linear_phi_change is the true change of phi"""
    rng = np.random.default_rng(1)
    names, pnames = [f"o{i}" for i in range(8)], ["a", "b", "c"]
    J = pd.DataFrame(rng.normal(size=(8, 3)), index=names, columns=pnames)
    obs = pd.DataFrame(dict(obsname=names, weight=rng.uniform(0.5, 2, 8), obsgrp="head"))
    measured = rng.normal(size=8)
    p0, p1 = np.zeros(3), np.array([0.3, -0.2, 0.1])
    model = lambda p: J.values @ p
    rei = pd.DataFrame(dict(obsname=names, residual=measured - model(p0)))
    phi = lambda p: float(((obs.weight.values * (measured - model(p))) ** 2).sum())
    step = pd.Series(p1 - p0, index=pnames)
    C = step_contribution(get_obs_pull(obs, J, rei, filter_insensitivetive=False), step)
    first_order = -2 * (obs.weight.values ** 2 * rei.residual.values) @ (-J.values) @ step.values
    assert C.values.sum() == pytest.approx(first_order)
    assert linear_phi_change(obs, J, rei, step) == pytest.approx(phi(p1) - phi(p0))


def test_jacobian_step_units():
    base = pd.DataFrame(dict(parname=["a", "b", "c"], partrans=["log", "none", "fixed"],
                             parval0=[1.0, 2.0, 5.0]))
    pars = {1: base.assign(parval=[10.0, 2.5, 5.0]), 2: base.assign(parval=[100.0, 1.5, 5.0])}
    assert jacobian_step(pars, 1).to_dict() == pytest.approx({"a": 1.0, "b": 0.5})     # from the initial values
    assert jacobian_step(pars, 2).to_dict() == pytest.approx({"a": 1.0, "b": -1.0})    # absolute change for none
    assert jacobian_step(pars, 3) is None
