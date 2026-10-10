"""Comparing the setup of two runs (compare.py, the Setup Changes page, runs to compare on the Upload page)."""
import pathlib
import zipfile

import numpy as np
import pytest

makepst = pytest.importorskip("makepst")

import compare  # noqa: E402

EXAMPLE = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"
PST = "Napa2026_04.pst"


@pytest.fixture(scope="module")
def variant(tmp_path_factory):
    """
    The example run with a changed setup: weights x0.5 for the first 10 observations, weight 0 for the
    11th, a new upper bound for kdcr01, one prior-information equation removed and a run-log line.
    """
    path = tmp_path_factory.mktemp("runs") / "variant.zip"
    with zipfile.ZipFile(EXAMPLE) as src:
        p = makepst.from_text(src.read(PST).decode().replace("\r\n", "\n"))
        p.obs.loc[p.obs.index[:10], "WEIGHT"] = p.obs.WEIGHT.iloc[:10] * 0.5
        p.obs.loc[p.obs.index[10], "WEIGHT"] = 0.0
        p.par.loc[p.par.PARNME == "kdcr01", "PARUBND"] = 0.5
        p.prior = p.prior[p.prior.PINME != "kdfn01"].reset_index(drop=True)
        p.comments = p.comments + ["| 2026-10-09 | variant | test changes |"]
        with zipfile.ZipFile(path, "w") as dst:
            for item in src.infolist():
                if item.filename == PST:
                    dst.writestr(PST, makepst.to_text(p, validate=False))
                elif not item.filename.endswith((".jco.1", ".jco.2", ".jco.3", ".jco.4", ".jco.5")):
                    dst.writestr(item, src.read(item.filename))
    return {"pestpath": str(path), "name": "Napa2026_04"}


REF = {"pestpath": str(EXAMPLE)}


def test_setup_diff(variant):
    a, b = compare.run_pst(REF), compare.run_pst(variant)
    d = compare.setup_diff(a, b)
    assert len(d.obs) == 11 and set(d.obs.column) == {"WEIGHT"}
    assert d.par[["PARNME", "column", "new"]].values.tolist() == [["kdcr01", "PARUBND", "0.5"]]
    assert d.prior[["PINME", "change"]].values.tolist() == [["kdfn01", "removed"]]
    assert d.comments.change.tolist() == ["added"]
    assert compare.setup_diff(a, a).empty


def test_summary_counts(variant):
    df = compare.summary_counts(compare.run_pst(REF), compare.run_pst(variant))
    assert df.loc["Prior-information equations"].tolist() == [70, 69, -1]
    assert df.loc["Weighted observations", "change"] == -1
    assert (df.drop(["Prior-information equations", "Weighted observations"]).change == 0).all()


def test_observation_changes(pest_data):
    obs = pest_data[2]
    other = obs.copy()
    other.loc[other.index[:3], "weight"] *= 0.2
    other.loc[other.index[3], "weight"] = 0.0
    other = other.drop(other.index[4]).drop(columns=["time"])
    ch = compare.observation_changes(obs, other)
    assert ch.status.value_counts().to_dict() == {"same": len(obs) - 5, "changed": 4, "removed": 1}
    assert np.allclose(ch.loc[obs.obsname.iloc[:3], "weight_ratio"], 0.2)    # matched by name
    assert ch.time.notna().sum() == obs.time.notna().sum()   # times from the run that has obs.csv
    g = compare.observation_group_summary(ch).loc[obs.obsgrp.iloc[0]]
    assert (g.weight_changed, g.set_to_zero, g.removed, g.ratio_median) == (4, 1, 1, pytest.approx(0.2))


def test_change_counts():
    import pandas as pd
    df = pd.DataFrame(dict(PARNME=["a", "b", "c"], change="changed", column=["PARLBND", "PARLBND", "PARUBND"]))
    out = compare.change_counts(df, "PARNME", pd.Series({"a": "hk", "b": "hk", "c": "vk"}))
    assert out[["group", "column", "count", "examples"]].values.tolist() == [
        ["hk", "PARLBND", 2, "a, b"], ["vk", "PARUBND", 1, "c"]]


# ---------------------------------------------------------------------------
# the page and the Upload page's run list
# ---------------------------------------------------------------------------

def test_page_compares_in_either_direction(variant):
    from module.cmpsetup import compare_setup, FORWARD, BACKWARD
    out = compare_setup(1, REF, [variant], "0", FORWARD)
    summary, obs_summary, weights, obs_table, par_summary, par_table, prior_table, other = out
    assert "obs: 11 changed" in summary[0].children
    assert weights.layout.title.text == "11 weights changed: ratio Napa2026_04 / Napa2026_04"
    assert any(np.isclose(np.asarray(t.y, dtype=float), 0.5).sum() == 10 for t in weights.data)
    assert list(prior_table.data[0].cells.values[2]) == ["removed"]
    assert "(weight 50" in prior_table.data[0].cells.values[4][0]        # the removed equation is shown
    back = compare_setup(1, REF, [variant], "0", BACKWARD)
    assert list(back[6].data[0].cells.values[2]) == ["added"]


def test_page_messages():
    from module.cmpsetup import compare_setup
    assert "Add another run" in compare_setup(1, REF, [], None)[0].children
    assert "No data loaded" in compare_setup(1, None, [], None)[0].children


def test_add_compare_run(variant, monkeypatch):
    import module.uploadpest as up
    monkeypatch.setattr(up, "ctx", type("ctx", (), {"triggered_id": "btn-add-compare"}))
    runs, msg, _ = up.add_compare_run(1, None, None, variant["pestpath"], [])
    assert runs == [{"filename": None, "upload": None, "pestpath": variant["pestpath"], "name": "Napa2026_04"}]
    assert up.add_compare_run(1, None, None, variant["pestpath"], runs)[1].children == "That run is already in the list."
    runs2, msg, _ = up.add_compare_run(1, None, None, "Z:/no/such/run.zip", runs)
    assert runs2 is up.no_update and msg.color == "red"


def test_compare_section_registered():
    import module
    assert list(module.sections["Compare"]) == ["cmpsetup", "cmpfit", "cmppar"]


# ---------------------------------------------------------------------------
# fit: both runs scored with the same weights (Fit Changes page)
# ---------------------------------------------------------------------------

def _rei(names, residuals, groups, weight=1.0):
    import pandas as pd
    return pd.DataFrame(dict(obsname=names, residual=residuals, obsgrp=groups, weight=weight))


def test_paired_residuals_and_fit_by_group():
    import pandas as pd
    a = _rei(["a", "b", "c", "p"], [1.0, 2.0, 3.0, 9.0], ["h", "h", "f", "regul1"])
    b = _rei(["a", "b", "d"], [0.5, 2.0, 1.0], ["h", "h", "f"])
    w = pd.Series({"a": 1.0, "b": 2.0, "c": 1.0, "d": 1.0})
    p = compare.paired_residuals(a, b, w, w, pd.Series({"a": "G1", "b": "G1", "c": "g2", "d": "g2"}))
    assert sorted(p.index) == ["a", "b"] and set(p.group) == {"g1"}     # common, no regularization, lower case
    f = compare.fit_by_group(p).loc["g1"]
    assert (f.phi_from, f.phi_to, f.n) == (1 + 16, 0.25 + 16, 2)
    assert f.rmse_from == pytest.approx(np.sqrt(2.5)) and f.drmse == pytest.approx(np.sqrt(2.125) - np.sqrt(2.5))


def test_pick_iteration():
    reis = {0: _rei(["a"], [3.0], ["h"]), 1: _rei(["a"], [1.0], ["h"]), 2: _rei(["a"], [2.0], ["h"])}
    assert compare.pick_iteration(reis, compare.BEST) == 1
    assert compare.pick_iteration(reis, compare.LAST) == 2
    assert compare.pick_iteration(reis, "0") == 0 and compare.pick_iteration(reis, "7") is None


def test_fit_page_scores_both_runs_with_the_same_weights(variant):
    from module.cmpfit import compare_fit
    args = (1, REF, [variant], "0", compare.FORWARD, compare.LAST, compare.LAST)
    # same residuals in both runs: with shared weights phi does not change ...
    summary, phi, rmse, _ = compare_fit(*args, compare.WEIGHTS[0], "obsgrp", "None", [])
    assert "(+0.0%)" in summary[0].children
    assert np.allclose(phi.data[0].y, phi.data[1].y)
    # ... with each run's own weights it changes by the reweighting alone
    summary, phi, _, _ = compare_fit(*args, compare.WEIGHTS[2], "obsgrp", "None", [])
    assert "(+0.0%)" not in summary[0].children
    assert not np.allclose(phi.data[0].y, phi.data[1].y)


def test_fit_page_map_needs_a_site_grouping(variant):
    from module.cmpfit import compare_fit
    args = (1, REF, [variant], "0", compare.FORWARD, compare.LAST, compare.LAST, compare.WEIGHTS[0])
    by_site = compare_fit(*args, "grpSite", "None", [])[3]
    assert by_site.data and len(by_site.data[0].lat) > 100
    by_type = compare_fit(*args, "grpType", "None", [])[3]
    assert "group by a site column" in by_type.layout.annotations[0].text


# ---------------------------------------------------------------------------
# parameters: values matched by name (Parameter Changes page)
# ---------------------------------------------------------------------------

def test_parameter_values_and_pairing():
    import pandas as pd
    par = pd.DataFrame(dict(parname=["a", "b", "c"], partrans=["log", "none", "fixed"], pargrp="g",
                            parval0=[1.0, 2.0, 5.0], parlbnd=[0.1, -10.0, 1.0], parubnd=[10.0, 10.0, 10.0]))
    pars = {1: pd.DataFrame(dict(parname=["a", "b", "c"], parval=[10.0, 3.0, 5.0]))}
    v0, v1 = compare.parameter_values(par, pars, 0), compare.parameter_values(par, pars, 1)
    assert v1.at_bound.to_dict() == {"a": True, "b": False, "c": False}
    assert compare.parameter_values(par, pars, 2) is None
    m = compare.paired_parameters(v0, v1)
    assert m.change.to_dict() == pytest.approx({"a": 1.0, "b": 0.5, "c": 0.0})
    assert m.log.to_dict() == {"a": True, "b": False, "c": False}


def test_parameter_page_matches_parameter_change(pest_data):
    """initial values -> last iteration of the same run is what pst.parameter_change computes"""
    from pst import parameter_change
    from module.cmppar import compare_parameters, ADJUSTABLE
    pars = pest_data[3]
    summary, change, values = compare_parameters(1, REF, [dict(REF, name="same")], "0", compare.FORWARD,
                                                 "0", compare.LAST, ADJUSTABLE)
    shown = {y: x for t in change.data if t.type == "bar" for x, y in zip(t.x, t.y)}
    expected = parameter_change(pars[max(pars)])
    assert len(shown) == len(expected)
    assert np.allclose([shown[p] for p in expected.index], expected.values)


def test_parameter_page_same_run_has_no_change():
    from module.cmppar import compare_parameters, ALL
    out = compare_parameters(1, REF, [dict(REF, name="same")], "0", compare.FORWARD, compare.LAST, compare.LAST, ALL)
    assert out[0].children.startswith("434 parameters compared") and "0 changed by more than" in out[0].children
