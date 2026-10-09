"""Linear (FOSM) parameter uncertainty (fosm.py) and the Parameter Uncertainty page."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fosm import linear_uncertainty, measurement_rows, noise_scale, prior_information_rows, prior_std


def _par(**bounds):
    """parameter data: name -> (partrans, lower, upper)"""
    return pd.DataFrame([dict(parname=n, partrans=t, parlbnd=lo, parubnd=hi, pargrp="g")
                         for n, (t, lo, hi) in bounds.items()])


def test_prior_std_from_bounds():
    par = _par(a=("log", 0.01, 100.0), b=("none", 0.0, 8.0), c=("fixed", 1.0, 2.0), d=("tied", 1.0, 2.0))
    sd = prior_std(par, nstd=4)
    assert sd.to_dict() == pytest.approx({"a": 1.0, "b": 2.0})      # log10 span 4 / 4; span 8 / 4
    assert prior_std(par, nstd=2)["a"] == pytest.approx(2.0)


def test_single_observation_posterior():
    # one observation (weight 3) with sensitivity 2 to one parameter of prior std 1:
    # posterior variance = 1 / (w^2 J^2 + 1/sd^2) = 1/37
    par = _par(a=("log", 0.01, 100.0))
    jco = pd.DataFrame([[2.0]], index=["o1"], columns=["a"])
    rows = pd.Series({"o1": 3.0})
    df = linear_uncertainty(jco, par, rows)
    assert df.loc["a", "post_std"] == pytest.approx(37 ** -0.5)
    assert df.loc["a", "reduction"] == pytest.approx(1 - 37 ** -0.5)
    # noise variance x4: Q = 9/4 -> variance 1/(9 + 1)
    assert linear_uncertainty(jco, par, rows, scale=4.0).loc["a", "post_std"] == pytest.approx(0.1 ** 0.5)


def test_matches_observation_space_schur_complement():
    rng = np.random.default_rng(0)
    par = _par(**{f"p{i}": ("log", 10.0 ** -i, 10.0 ** (i + 1)) for i in range(4)})
    J = rng.normal(size=(6, 4))
    jco = pd.DataFrame(J, index=[f"o{i}" for i in range(6)], columns=par.parname)
    w = pd.Series(rng.uniform(0.5, 2.0, 6), index=jco.index)
    df = linear_uncertainty(jco, par, w, scale=2.0)
    C = np.diag(prior_std(par) ** 2)
    R = np.diag(2.0 / w.values ** 2)
    post = C - C @ J.T @ np.linalg.inv(J @ C @ J.T + R) @ J @ C
    np.testing.assert_allclose(df.post_std.values, np.sqrt(np.diag(post)))


def test_zero_column_is_not_sensitive():
    par = _par(a=("log", 0.01, 100.0), b=("log", 0.01, 100.0))
    jco = pd.DataFrame([[1.0, 0.0], [0.0, 0.5]], index=["o1", "prior1"], columns=["a", "b"])
    rows = pd.Series({"o1": 1.0, "prior1": 1.0})
    df = linear_uncertainty(jco, par, rows, measured=["o1"])
    assert df.sensitive.to_dict() == {"a": True, "b": False}     # b only has a prior-information row
    assert df.loc["b", "reduction"] > 0                           # ... which still constrains it
    df = linear_uncertainty(jco, par, rows[["o1"]], measured=["o1"])
    assert df.loc["b", "reduction"] == pytest.approx(0.0)


def test_rows_and_noise_scale():
    obs = pd.DataFrame(dict(obsname=["o1", "o2", "o3", "r1"], weight=[2.0, 1.0, 0.0, 5.0],
                            obsgrp=["head", "head", "head", "regul_x"]))
    assert measurement_rows(obs).to_dict() == {"o1": 2.0, "o2": 1.0}
    assert set(measurement_rows(obs, include_regul=True).index) == {"o1", "o2", "r1"}
    rei = pd.DataFrame(dict(obsname=["o1", "o2", "o3", "r1", "pi1", "pi2"],
                            residual=[1.0, 3.0, 9.0, 1.0, 0.1, 0.1], weight=[2.0, 1.0, 0.0, 5.0, 0.5, 0.0]))
    assert noise_scale(obs, rei) == pytest.approx(((2 * 1) ** 2 + (1 * 3) ** 2) / 2)
    assert prior_information_rows(obs, rei).to_dict() == {"pi1": 0.5}
    assert prior_information_rows(obs, None).empty


# ---------------------------------------------------------------------------
# the page, on the example run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def page(pest_data):
    with patch("module.parunc.parse_zip", return_value=pest_data):
        from module import parunc
        yield parunc


def _run(page, *args, **kwargs):
    return page.plot(1, {"pestpath": "x"}, *args, **kwargs)


def test_page_placeholders(page):
    assert page.plot(None, {"pestpath": "x"}, "6").layout.annotations[0].text.startswith("Load data")
    assert "No data loaded" in page.plot(1, None, "6").layout.annotations[0].text


def test_page_flags_missing_derivatives(page, pest_data):
    jco = pest_data[5][6]
    fig = _run(page, "6")
    zero = next(t for t in fig.data if t.name.startswith("No measurement sensitivity"))
    obs = pest_data[2]
    measured = jco.loc[jco.index.isin(obs[obs.weight > 0].obsname)]
    assert set(zero.y) == set(measured.columns[(measured.abs().sum() == 0)])
    assert f"{len(zero.y)} of {jco.shape[1]} parameters have no measurement sensitivity" in fig.layout.title.text


def test_page_noise_scaling_lowers_reduction(page):
    def reductions(fig):
        return np.concatenate([np.asarray(t.x, dtype=float) for t in fig.data if t.type == "bar"])
    scaled = reductions(_run(page, "1"))                                    # default: scaled to the fit
    plain = reductions(_run(page, "1", noise="From weights (1/w)"))
    assert scaled.mean() < plain.mean()
    assert ((scaled >= 0) & (scaled <= 100)).all()


def test_page_regularization_rows(page):
    assert "+ 70 regularization rows" in _run(page, "6", regul=True).layout.title.text
    # .jco.1 pairs with the residuals from the .hp file, which hold no prior information
    assert "no regularization rows" in _run(page, "1", regul=True).layout.title.text


def test_page_groups_and_weight_column(page):
    fig = _run(page, "6", show="Parameter Groups")
    assert len(fig.data[0].x) == 6 and all("of" in x for x in fig.data[0].x)
    assert _run(page, "6", "weight_orig").to_json() != _run(page, "6", "weight").to_json()


def test_pages_registered():
    import module
    assert {"parident", "parunc"} <= set(module.sections["Parameters"])
