"""
Tests for Dash module callback functions.

GOOD_RESULT is loaded once at module import from testdata/example_pst.zip so
all @patch mocks return real PEST data (Napa2026_04, 6 iterations).
"""
import pathlib
from unittest.mock import patch
import pytest
import plotly.graph_objects as go

from utils import read_pest_zip

_ZIP = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"
_pst, _par, _obs, _pars, _reis, _jcos = read_pest_zip(str(_ZIP), True, True, True)

GOOD_RESULT  = (_pst, _par, _obs, _pars, _reis, _jcos)
NO_PST_RESULT = ("", None, None, {}, {}, {})

# Stable references derived from the real data — used as test arguments.
_PST_NAME     = _pst                                         # "Napa2026_04"
_FIRST_OBS    = _obs["obsname"].iloc[0]                      # "64221_1"
_FIRST_OBSGRP = str(_obs["obsgrp"].iloc[0])                  # "head"
_FIRST_PAR    = _par[_par.partrans.isin(["log", "none"])]["parname"].iloc[0]   # "kdcr01"
_FIRST_PARGRP = str(_par["pargrp"].iloc[0])                  # "power"
_N_ITER       = len(_pars)                                   # 6


# ---------------------------------------------------------------------------
# uploadpest
# ---------------------------------------------------------------------------

@patch("module.uploadpest.parse_zip", return_value=GOOD_RESULT)
def test_uploadpest_callback_success(mock_pz):
    from module.uploadpest import upload_pest
    status, tab = upload_pest({"pestpath": "dummy"})
    assert _PST_NAME in status
    assert tab == "phiupdate"

@patch("module.uploadpest.parse_zip", return_value=NO_PST_RESULT)
def test_uploadpest_callback_no_pst(mock_pz):
    from module.uploadpest import upload_pest
    status, tab = upload_pest({"pestpath": "dummy"})
    assert tab == "uploadpest"

def test_uploadpest_callback_no_data():
    from module.uploadpest import upload_pest
    status, tab = upload_pest(None)
    assert tab == "uploadpest"

def test_update_pest_data_stores_dict():
    from module.uploadpest import update_pest_data
    result = update_pest_data(1, "file.zip", "base64data==", None)
    assert result["filename"] == "file.zip"
    assert result["contents"] == "base64data=="

def test_update_pest_data_no_click_returns_none():
    from module.uploadpest import update_pest_data
    result = update_pest_data(0, "file.zip", "base64==", None)
    assert result is None


# ---------------------------------------------------------------------------
# phiupdate
# ---------------------------------------------------------------------------

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_returns_figure(mock_pz):
    from module.phiupdate import upload_pest
    fig = upload_pest({"pestpath": "dummy"}, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

@patch("module.phiupdate.parse_zip", return_value=NO_PST_RESULT)
def test_phiupdate_no_pst_returns_empty_figure(mock_pz):
    from module.phiupdate import upload_pest
    fig = upload_pest({"pestpath": "dummy"}, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

def test_phiupdate_no_data_returns_figure():
    from module.phiupdate import upload_pest
    fig = upload_pest(None, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_figure_has_traces(mock_pz):
    from module.phiupdate import upload_pest
    fig = upload_pest({"pestpath": "dummy"}, "obsgrp", "weight")
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# parcor
# ---------------------------------------------------------------------------

def test_parcor_no_clicks_returns_scatter():
    from module.parcor import plot
    fig = plot(None, {}, "40", "1", "0.98", "RdBu", True)
    assert isinstance(fig, go.Figure)

def test_parcor_no_data_returns_scatter():
    from module.parcor import plot
    fig = plot(1, None, "40", "1", "0.98", "RdBu", True)
    assert isinstance(fig, go.Figure)

@patch("module.parcor.parse_zip", return_value=GOOD_RESULT)
def test_parcor_with_data_returns_figure(mock_pz):
    from module.parcor import plot
    fig = plot(1, {"pestpath": "dummy"}, "40", "1", "0.98", "RdBu", True)
    assert isinstance(fig, go.Figure)

@patch("module.parcor.parse_zip", return_value=NO_PST_RESULT)
def test_parcor_no_pst_returns_scatter(mock_pz):
    from module.parcor import plot
    fig = plot(1, {"pestpath": "dummy"}, "40", "1", "0.98", "RdBu", True)
    assert isinstance(fig, go.Figure)

@patch("module.parcor.parse_zip", return_value=GOOD_RESULT)
def test_parcor_inverted_colorscale(mock_pz):
    from module.parcor import plot
    fig_inv  = plot(1, {"pestpath": "dummy"}, "40", "1", "0.98", "RdBu", True)
    fig_norm = plot(1, {"pestpath": "dummy"}, "40", "1", "0.98", "RdBu", False)
    assert isinstance(fig_inv, go.Figure)
    assert isinstance(fig_norm, go.Figure)


# ---------------------------------------------------------------------------
# parsen (parameter sensitivity)
# ---------------------------------------------------------------------------

def test_parsen_no_clicks_returns_scatter():
    from module.parsen import plot
    fig = plot(None, {}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    assert isinstance(fig, go.Figure)

def test_parsen_no_data_returns_scatter():
    from module.parsen import plot
    fig = plot(1, None, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    assert isinstance(fig, go.Figure)

def test_parsen_no_param_selected_returns_scatter():
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [])
    assert isinstance(fig, go.Figure)

@patch("module.parsen.parse_zip", return_value=GOOD_RESULT)
def test_parsen_l1_norm(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L1 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    assert isinstance(fig, go.Figure)

@patch("module.parsen.parse_zip", return_value=GOOD_RESULT)
def test_parsen_l2_norm(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    assert isinstance(fig, go.Figure)

@patch("module.parsen.parse_zip", return_value=GOOD_RESULT)
def test_parsen_multiple_params(mock_pz):
    from module.parsen import plot
    two_pars = _par[_par.partrans.isin(["log", "none"])]["parname"].iloc[:2].tolist()
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", two_pars)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

@patch("module.parsen.parse_zip", return_value=GOOD_RESULT)
def test_parsen_single_string_param(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", _FIRST_PAR)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# obssen (observation sensitivity)
# ---------------------------------------------------------------------------

def test_obssen_no_clicks_returns_scatter():
    from module.obssen import plot
    fig = plot(None, {}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_OBS])
    assert isinstance(fig, go.Figure)

def test_obssen_no_obs_selected_returns_scatter():
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [])
    assert isinstance(fig, go.Figure)

@patch("module.obssen.parse_zip", return_value=GOOD_RESULT)
def test_obssen_l2_norm(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_OBS])
    assert isinstance(fig, go.Figure)

@patch("module.obssen.parse_zip", return_value=GOOD_RESULT)
def test_obssen_l1_norm(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L1 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_OBS])
    assert isinstance(fig, go.Figure)

@patch("module.obssen.parse_zip", return_value=GOOD_RESULT)
def test_obssen_obs_groups_mode(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Obs Groups", "obsgrp", "10", "Jet", [_FIRST_OBSGRP])
    assert isinstance(fig, go.Figure)

@patch("module.obssen.parse_zip", return_value=GOOD_RESULT)
def test_obssen_figure_has_traces(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_OBS])
    assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# scatterplot
# ---------------------------------------------------------------------------

def test_scatterplot_no_clicks_returns_scatter():
    from module.scatterplot import plot
    fig = plot(None, {}, "obsgrp", _FIRST_OBSGRP, False, "Greyout", "Jet", False)
    assert isinstance(fig, go.Figure)

def test_scatterplot_no_data_returns_scatter():
    from module.scatterplot import plot
    fig = plot(1, None, "obsgrp", _FIRST_OBSGRP, False, "Greyout", "Jet", False)
    assert isinstance(fig, go.Figure)

@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_scatterplot_with_data(mock_pz):
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "obsgrp", _FIRST_OBSGRP, False, "Greyout", "Jet", False)
    assert isinstance(fig, go.Figure)

@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_scatterplot_as_residual(mock_pz):
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "obsgrp", _FIRST_OBSGRP, True, "Greyout", "Jet", False)
    assert isinstance(fig, go.Figure)

@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_scatterplot_hide_zeroweight(mock_pz):
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "obsgrp", _FIRST_OBSGRP, False, "Hide", "Jet", False)
    assert isinstance(fig, go.Figure)

@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_scatterplot_inverted_palette(mock_pz):
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "obsgrp", _FIRST_OBSGRP, False, "Greyout", "Jet", True)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# resinfl (residual analysis)
# ---------------------------------------------------------------------------

def test_resinfl_no_clicks_returns_scatter():
    from module.resinfl import plot
    fig = plot(None, {}, "1", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert isinstance(fig, go.Figure)

def test_resinfl_no_data_returns_scatter():
    from module.resinfl import plot
    fig = plot(1, None, "1", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert isinstance(fig, go.Figure)

@patch("module.resinfl.parse_zip", return_value=GOOD_RESULT)
def test_resinfl_with_data(mock_pz):
    from module.resinfl import plot
    fig = plot(1, {"pestpath": "dummy"}, "1", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert isinstance(fig, go.Figure)

@patch("module.resinfl.parse_zip", return_value=GOOD_RESULT)
def test_resinfl_figure_has_traces(mock_pz):
    from module.resinfl import plot
    fig = plot(1, {"pestpath": "dummy"}, "1", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert len(fig.data) > 0

@patch("module.resinfl.parse_zip", return_value=GOOD_RESULT)
def test_resinfl_last_iteration(mock_pz):
    from module.resinfl import plot
    fig = plot(1, {"pestpath": "dummy"}, str(_N_ITER), "obsgrp", "weight", "Jet", False, "Linear", "Linear")
    assert isinstance(fig, go.Figure)


@patch("module.resinfl.parse_zip", return_value=GOOD_RESULT)
def test_resinfl_uses_residuals_of_previous_iteration(mock_pz):
    """jco.N is computed at the parameters of iteration N-1, so residuals come from rei.(N-1)."""
    import numpy as np
    from module.resinfl import plot
    fig = plot(1, {"pestpath": "dummy"}, "2", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    rei1 = _reis[1].set_index("obsname").residual
    for tr in fig.data:
        names = [t.split("<br>")[0] for t in tr.text]
        np.testing.assert_allclose(tr.marker.color, rei1.loc[names].values)

def test_resinfl_missing_files_return_message():
    from module.resinfl import plot
    no_jco = (_pst, _par, _obs, _pars, _reis, {})
    with patch("module.resinfl.parse_zip", return_value=no_jco):
        fig = plot(1, {"pestpath": "dummy"}, "1", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert "No .jco" in fig.layout.title.text
    no_rei = (_pst, _par, _obs, _pars, {5: _reis[5]}, _jcos)
    with patch("module.resinfl.parse_zip", return_value=no_rei):
        fig = plot(1, {"pestpath": "dummy"}, "2", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert "No .rei" in fig.layout.title.text


# ---------------------------------------------------------------------------
# parupdate
# ---------------------------------------------------------------------------

def test_parupdate_no_clicks_returns_scatter():
    from module.parupdate import update_plot
    fig = update_plot(None, {}, "Relative", [_FIRST_PARGRP], ["log", "none"], True, "98%", "Jet", "linear", "Horizontal")
    assert isinstance(fig, go.Figure)

def test_parupdate_no_data_returns_scatter():
    from module.parupdate import update_plot
    fig = update_plot(1, None, "Relative", [_FIRST_PARGRP], ["log", "none"], True, "98%", "Jet", "linear", "Horizontal")
    assert isinstance(fig, go.Figure)

@patch("module.parupdate.parse_zip", return_value=GOOD_RESULT)
def test_parupdate_with_data(mock_pz):
    from module.parupdate import update_plot
    pargroups = _par["pargrp"].unique().tolist()
    fig = update_plot(1, {"pestpath": "dummy"}, "Relative", pargroups, ["log", "none", "fixed", "tied"], True, "98%", "Jet", "linear", "Horizontal")
    assert isinstance(fig, go.Figure)

@patch("module.parupdate.parse_zip", return_value=GOOD_RESULT)
def test_parupdate_vertical_layout(mock_pz):
    from module.parupdate import update_plot
    pargroups = _par["pargrp"].unique().tolist()
    fig = update_plot(1, {"pestpath": "dummy"}, "Raw", pargroups, ["log", "none", "fixed", "tied"], False, "95%", "Jet", "log", "Vertical")
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# obsconf (observation conflict)
# ---------------------------------------------------------------------------

def test_obsconf_no_clicks_returns_scatter():
    from module.obsconf import plot
    bar, heat, opts, sel = plot(None, {}, "1", "weight", "obsgrp", "Parameters", "30", "30")
    assert isinstance(bar, go.Figure)

def test_obsconf_no_data_returns_scatter():
    from module.obsconf import plot
    bar, heat, opts, sel = plot(1, None, "1", "weight", "obsgrp", "Parameters", "30", "30")
    assert isinstance(bar, go.Figure)

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_with_data(mock_pz):
    from module.obsconf import plot
    bar, heat, opts, sel = plot(1, {"pestpath": "dummy"}, str(_N_ITER), "weight", "grpSite", "Parameters", "20", "10")
    assert len(bar.data) == 3
    assert len(bar.data[0].x) == 20
    assert len(heat.data[0].y) == 10
    assert 0 < len(sel) <= 5 and set(sel) <= set(opts)

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_par_groups(mock_pz):
    from module.obsconf import plot
    bar, heat, opts, sel = plot(1, {"pestpath": "dummy"}, "1", "weight", "obsgrp", "Par Groups", "30", "30")
    assert set(opts) <= set(_par["pargrp"])

@patch("module.obsconf.parse_zip", return_value=NO_PST_RESULT)
def test_obsconf_no_pst(mock_pz):
    from module.obsconf import plot
    bar, heat, opts, sel = plot(1, {"pestpath": "dummy"}, "1", "weight", "obsgrp", "Parameters", "30", "30")
    assert isinstance(bar, go.Figure)

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_sankey(mock_pz):
    from module.obsconf import plot, plot_sankey
    _, _, opts, sel = plot(1, {"pestpath": "dummy"}, "1", "weight", "obsgrp", "Parameters", "30", "30")
    fig = plot_sankey(sel[:2], "obsname", "5", True, {"pestpath": "dummy"}, "1", "weight", "Parameters")
    sk = fig.data[0]
    assert isinstance(sk, go.Sankey)
    assert len(sk.link.value) <= 2 * (5 + 2)

def test_obsconf_sankey_nothing_selected():
    from module.obsconf import plot_sankey
    fig = plot_sankey([], "obsname", "5", True, {"pestpath": "dummy"}, "1", "weight", "Parameters")
    assert isinstance(fig, go.Figure)

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_iterations_pair_with_previous_rei(mock_pz):
    from module.obsconf import upload_pest
    iters, value = upload_pest({"pestpath": "dummy"})
    assert iters == [str(i) for i in range(1, _N_ITER + 1)]
    assert value == str(_N_ITER)

def test_obsconf_opponents_scores():
    import pandas as pd
    from module.obsconf import get_opponents, get_focus_stats
    F = pd.Series({"k1": 2.0, "k2": -1.0, "k3": 1.0})
    O = pd.DataFrame({"k1": [-5.0, 1.0], "k2": [-0.5, 0.0], "k3": [0.0, 1.0]}, index=["a", "b"])
    opp = get_opponents(F, O)
    # a opposes F on k1 (min(2, 5) = 2) and agrees on k2 (min(1, 0.5) = 0.5); total |F| = 4
    assert opp.loc["a", "oppose"] == 0.5 and opp.loc["a", "agree"] == 0.125
    assert opp.loc["a", "main_par"] == "k1"
    assert opp.loc["b", "oppose"] == 0 and opp.loc["b", "agree"] == 0.5 and opp.loc["b", "main_par"] == ""
    assert opp.index[0] == "a"
    stats = get_focus_stats(F, O.sum())
    assert stats.index[0] == "k1" and stats.loc["k1", "opposed"] == 2
    assert stats.loc["k3", "opposed"] == 0

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_focus_observation(mock_pz):
    from module.obsconf import plot_focus
    fbar, fopp, fsankey = plot_focus("obsname", [_FIRST_OBS], "grpSite", "10", "10",
                                     {"pestpath": "dummy"}, "1", "weight", "Parameters")
    assert len(fbar.data) == 2 and len(fbar.data[0].x) <= 10
    assert len(fopp.data) == 2 and len(fopp.data[0].y) <= 10
    sk = fsankey.data[0]
    assert isinstance(sk, go.Sankey)
    assert sk.node.label[0].startswith("FOCUS")

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_focus_group_excludes_itself(mock_pz):
    from module.obsconf import plot_focus
    site = str(_obs[_obs.weight > 0]["grpSite"].iloc[0])
    fbar, fopp, fsankey = plot_focus("grpSite", [site], "grpSite", "20", "50",
                                     {"pestpath": "dummy"}, "6", "weight", "Par Groups")
    assert site not in list(fopp.data[0].y)
    assert set(fbar.data[0].x) <= set(_par["pargrp"])

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_focus_zero_weight_target(mock_pz):
    from module.obsconf import plot_focus
    zero = _obs[_obs.weight == 0]["obsname"].iloc[0]
    fbar, _, _ = plot_focus("obsname", [zero], "obsname", "10", "10",
                            {"pestpath": "dummy"}, "1", "weight", "Parameters")
    assert "no pull" in fbar.layout.title.text

def test_obsconf_focus_nothing_selected():
    from module.obsconf import plot_focus
    fbar, _, _ = plot_focus("obsname", [], "obsname", "10", "10", {"pestpath": "dummy"}, "1", "weight", "Parameters")
    assert isinstance(fbar, go.Figure)

def test_obsconf_focus_from_heatmap_click():
    from module.obsconf import focus_from_heatmap
    ftype, focus = focus_from_heatmap({"points": [{"x": "kdcr01", "y": "nc-76"}]}, "grpSite")
    assert ftype == "grpSite" and focus == ["nc-76"]

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_focus_options(mock_pz):
    from module.obsconf import update_focus_options
    groups = update_focus_options("obsname", {"pestpath": "dummy"}, "weight")
    names = {o for g in groups for o in g["items"]}
    assert names == set(_obs[_obs.weight > 0]["obsname"])
    sites = update_focus_options("grpSite", {"pestpath": "dummy"}, "weight")
    assert sites == sorted(sites) and len(sites) > 1


# ---------------------------------------------------------------------------
# runs without a .hp file (no iteration-0 residuals)
# ---------------------------------------------------------------------------

NO_HP_RESULT = (_pst, _par, _obs, _pars, {k: v for k, v in _reis.items() if k != 0}, _jcos)
_FIRST_SITE = str(_obs["grpSite"].iloc[0])

@patch("module.scatterplot.parse_zip", return_value=NO_HP_RESULT)
def test_scatterplot_without_hp(mock_pz):
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, False, "Greyout", "Jet", False)
    assert [t.name for t in fig.data if t.type == "scatter"][1:] == [f"Iter {i}" for i in range(1, _N_ITER + 1)]

@patch("module.tsplot.parse_zip", return_value=NO_HP_RESULT)
def test_tsplot_without_hp(mock_pz):
    from module.tsplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, "Greyout", "Lines+Markers", "Jet", False)
    assert len([t for t in fig.data if t.type == "scatter"]) == _N_ITER + 1

def test_tsplot_without_time_column():
    from module.tsplot import plot
    no_time = (_pst, _par, _obs.drop(columns="time"), _pars, _reis, _jcos)
    with patch("module.tsplot.parse_zip", return_value=no_time):
        fig = plot(1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, "Greyout", "Lines+Markers", "Jet", False)
    assert "time" in fig.layout.title.text

@patch("module.parsen.parse_zip", return_value=NO_HP_RESULT)
def test_parsen_without_hp(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    names = [t.name for t in fig.data if t.mode == "lines+markers"]
    assert names == [f"Iter {i}" for i in range(0, _N_ITER + 1)]

@patch("module.obssen.parse_zip", return_value=NO_HP_RESULT)
def test_obssen_without_hp_and_wide_group(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Obs Groups", "obsgrp", "10", "Jet", [_FIRST_OBSGRP])
    assert fig.layout.xaxis2.domain[1] - fig.layout.xaxis2.domain[0] < 0.5   # side panel capped
    names = [t.name for t in fig.data if t.mode == "lines+markers"]
    assert names == sorted(names, key=lambda c: int(c.split()[1]))

def test_phiupdate_flags_unreasonable_residuals():
    from module.phiupdate import upload_pest
    bad = {k: v.copy() for k, v in _reis.items()}
    bad[3]["residual"] = 1e30
    with patch("module.phiupdate.parse_zip", return_value=(_pst, _par, _obs, _pars, bad, _jcos)):
        fig = upload_pest({"pestpath": "dummy"}, "obsgrp", "weight")
    assert "rei.3" in fig.layout.title.text

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_includes_regularization_groups(mock_pz):
    from module.phiupdate import upload_pest
    fig = upload_pest({"pestpath": "dummy"}, "obsgrp", "weight")
    assert any(str(t.name).startswith("regul") for t in fig.data)
