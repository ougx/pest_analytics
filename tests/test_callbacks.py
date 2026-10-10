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

def _message(fig):
    """text of a placeholder figure (plotting.message_figure)"""
    return " ".join(a.text for a in fig.layout.annotations)

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
    status, tab, msg = upload_pest({"pestpath": "dummy"})
    assert _PST_NAME in status
    assert tab == "phiupdate"
    assert msg[0].color == "green" and f"{len(_obs)} observations" in msg[0].children

@patch("module.uploadpest.parse_zip", return_value=NO_PST_RESULT)
def test_uploadpest_callback_no_pst(mock_pz):
    from module.uploadpest import upload_pest
    status, tab, msg = upload_pest({"pestpath": "dummy"})
    assert tab == "uploadpest"
    assert msg.color == "red" and "No .pst" in msg.children

def test_uploadpest_callback_no_data():
    from module.uploadpest import upload_pest
    status, tab, msg = upload_pest(None)
    assert tab == "uploadpest" and msg is None

def test_uploadpest_callback_missing_path():
    from module.uploadpest import upload_pest
    status, tab, msg = upload_pest({"pestpath": "Z:/no/such/run.zip", "upload": None})
    assert tab == "uploadpest"
    assert msg.color == "red" and "File not found" in msg.children

def test_uploadpest_callback_bad_zip(tmp_path):
    from module.uploadpest import upload_pest
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"this is not a zip file")
    status, tab, msg = upload_pest({"pestpath": str(bad)})
    assert tab == "uploadpest"
    assert msg.color == "red" and "Could not read" in msg.children

def test_upload_shows_selected_file_name():
    from module.uploadpest import show_upload_name, UPLOAD_PROMPT
    assert "run.zip" in show_upload_name("run.zip")
    assert show_upload_name(None) == UPLOAD_PROMPT

def test_update_pest_data_saves_upload_on_server(tmp_path, monkeypatch):
    import base64
    import utils
    from module.uploadpest import update_pest_data
    monkeypatch.setattr(utils, "UPLOAD_DIR", str(tmp_path))
    contents = "data:application/zip;base64," + base64.b64encode(_ZIP.read_bytes()).decode()
    result = update_pest_data(1, "file.zip", contents, None)
    assert result["filename"] == "file.zip"
    assert "contents" not in result                       # the file is not kept in the browser
    assert utils.upload_path(result["upload"]) is not None
    assert utils.parse_zip(result)[0] == _PST_NAME

def test_update_pest_data_no_click_keeps_loaded_data():
    from dash import no_update
    from module.uploadpest import update_pest_data
    result = update_pest_data(0, "file.zip", "base64==", None)
    assert result is no_update


# ---------------------------------------------------------------------------
# phiupdate
# ---------------------------------------------------------------------------

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_returns_figure(mock_pz):
    from module.phiupdate import phi_figure
    fig = phi_figure({"pestpath": "dummy"}, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

@patch("module.phiupdate.parse_zip", return_value=NO_PST_RESULT)
def test_phiupdate_no_pst_returns_empty_figure(mock_pz):
    from module.phiupdate import phi_figure
    fig = phi_figure({"pestpath": "dummy"}, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

def test_phiupdate_no_data_returns_figure():
    from module.phiupdate import phi_figure
    fig = phi_figure(None, "obsgrp", "weight")
    assert isinstance(fig, go.Figure)

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_figure_has_traces(mock_pz):
    from module.phiupdate import phi_figure
    fig = phi_figure({"pestpath": "dummy"}, "obsgrp", "weight")
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
    assert "No .jco" in _message(fig)
    no_rei = (_pst, _par, _obs, _pars, {5: _reis[5]}, _jcos)
    with patch("module.resinfl.parse_zip", return_value=no_rei):
        fig = plot(1, {"pestpath": "dummy"}, "2", "obsgrp", "weight", "Jet", True, "Log", "Linear")
    assert "No .rei" in _message(fig)


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

@patch("layout.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_iterations_pair_with_previous_rei(mock_pz):
    from module.obsconf import fill_iterations
    iters, value = fill_iterations({"pestpath": "dummy"})
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
    assert "no pull" in _message(fbar)

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
    assert [t.name for t in fig.data if t.type == "scattergl"] == [f"Iter {i}" for i in range(1, _N_ITER + 1)]

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
    assert "time" in _message(fig)

@patch("module.parsen.parse_zip", return_value=NO_HP_RESULT)
def test_parsen_without_hp(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    names = [t.name for t in fig.data if t.mode == "lines+markers"]
    assert names == [f"Jacobian {i}" for i in range(1, _N_ITER + 1)]

@patch("module.obssen.parse_zip", return_value=NO_HP_RESULT)
def test_obssen_without_hp_and_wide_group(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Obs Groups", "obsgrp", "10", "Jet", [_FIRST_OBSGRP])
    assert fig.layout.xaxis2.domain[1] - fig.layout.xaxis2.domain[0] < 0.5   # side panel capped
    names = [t.name for t in fig.data if t.mode == "lines+markers"]
    assert names == sorted(names, key=lambda c: int(c.split()[1]))

def test_phiupdate_flags_unreasonable_residuals():
    from module.phiupdate import phi_figure
    bad = {k: v.copy() for k, v in _reis.items()}
    bad[3]["residual"] = 1e30
    with patch("module.phiupdate.parse_zip", return_value=(_pst, _par, _obs, _pars, bad, _jcos)):
        fig = phi_figure({"pestpath": "dummy"}, "obsgrp", "weight")
    assert "rei.3" in _message(fig)

@patch("module.phiupdate.parse_zip", return_value=GOOD_RESULT)
def test_phiupdate_includes_regularization_groups(mock_pz):
    from module.phiupdate import phi_figure
    fig = phi_figure({"pestpath": "dummy"}, "obsgrp", "weight")
    assert any(str(t.name).startswith("regul") for t in fig.data)


# ---------------------------------------------------------------------------
# scatter / time series statistics use weighted targets only
# ---------------------------------------------------------------------------

_w = _obs.groupby("grpSite").weight
_MIXED_SITE = str(((_w.min() == 0) & (_w.max() > 0)).idxmax())   # site with zero- and non-zero-weight targets

def _stats_table(fig):
    table = next(t for t in fig.data if t.type == "table")
    return [list(col) for col in table.cells.values]

def test_residual_stats_excludes_zero_weight():
    import numpy as np
    import pandas as pd
    from plotting import residual_stats
    ts = pd.DataFrame(dict(measured=[1.0, 2.0, 3.0, 9.0], modelled=[1.0, 2.5, 2.0, 0.0],
                           residual=[0.0, -0.5, 1.0, 9.0], weight=[1.0, 1.0, 1.0, 0.0]))
    rmse, r2, mae, mean, rmin, rmax = residual_stats(ts)
    assert np.isclose(rmse, np.sqrt((0.25 + 1.0) / 3))
    assert np.isclose(mae, 0.5) and np.isclose(mean, 0.5 / 3)
    assert rmin == -0.5 and rmax == 1.0
    assert all(np.isnan(residual_stats(ts[ts.weight == 0])))

def test_stats_table_layout():
    from plotting import stats_table, stat_names
    table = stats_table({"Iter 1": [1.0, 0.5, 0.8, 0.1, -2.0, 3.0], "Iter 2": [0.5, 0.9, 0.4, 0.0, -1.0, 1.0]})
    assert list(table.header.values) == ["Statistics<br>(weight > 0)", "Iter 1", "Iter 2"]
    names, it1, it2 = [list(c) for c in table.cells.values]
    assert names == stat_names
    assert [float(v) for v in it1] == [1.0, 0.5, 0.8, 0.1, -2.0, 3.0]
    assert [float(v) for v in it2] == [0.5, 0.9, 0.4, 0.0, -1.0, 1.0]

@pytest.mark.parametrize("module", ["scatterplot", "tsplot"])
def test_stats_ignore_zero_weight_display_option(module):
    import importlib
    plot = importlib.import_module(f"module.{module}").plot
    with patch(f"module.{module}.parse_zip", return_value=GOOD_RESULT):
        tables = []
        for zeroweight in ["Hide", "Greyout", "DoNothing"]:
            if module == "scatterplot":
                fig = plot(1, {"pestpath": "dummy"}, "grpSite", _MIXED_SITE, False, zeroweight, "Jet", False)
            else:
                fig = plot(1, {"pestpath": "dummy"}, "grpSite", _MIXED_SITE, zeroweight, "Lines", "Jet", False)
            tables.append(_stats_table(fig))
    assert tables[0] == tables[1] == tables[2]


# ---------------------------------------------------------------------------
# Jacobian N is shown in the color of iteration N-1 (where it was computed)
# ---------------------------------------------------------------------------

@patch("module.parsen.parse_zip", return_value=GOOD_RESULT)
def test_parsen_jacobian_colored_by_previous_iteration(mock_pz):
    from module.parsen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_PAR])
    values = {t.name: t.marker.color for t in fig.data if t.mode == "markers"}
    for t in fig.data:
        if t.mode == "lines+markers":
            k = int(t.name.split()[1])
            assert t.line.color == values[f"Iter {k - 1}"]

@patch("module.obssen.parse_zip", return_value=GOOD_RESULT)
def test_obssen_jacobian_colored_by_previous_iteration(mock_pz):
    from module.obssen import plot
    fig = plot(1, {"pestpath": "dummy"}, "L2 Norm", "Observations", "obsgrp", "10", "Jet", [_FIRST_OBS])
    values = {t.name: t.marker.color for t in fig.data if t.mode == "markers" and t.name != "Observed"}
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == [f"Jacobian {i}" for i in range(1, _N_ITER + 1)]
    for t in lines:
        assert t.line.color == values[f"Iter {int(t.name.split()[1]) - 1}"]


# ---------------------------------------------------------------------------
# every graph starts as a plain placeholder, not an empty chart with axes
# ---------------------------------------------------------------------------

def _graphs(component):
    from dash import dcc
    if isinstance(component, dcc.Graph):
        yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and hasattr(child, "to_plotly_json"):
            yield from _graphs(child)

def test_initial_figures_are_placeholders():
    from module import apps
    graphs = [g for page in apps.values() for g in _graphs(page.layout)]
    assert len(graphs) >= 15
    for g in graphs:
        fig = go.Figure(g.figure)
        assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False, g.id


# ---------------------------------------------------------------------------
# CSV download of the data shown in a plot
# ---------------------------------------------------------------------------

def _as_browser_json(fig):
    """a figure as dcc.Graph sends it back (typed arrays base64-encoded)"""
    import json
    import plotly.io as pio
    return json.loads(pio.to_json(fig))

@patch("module.parcor.parse_zip", return_value=GOOD_RESULT)
def test_csv_heatmap_is_matrix(mock_pz):
    import numpy as np
    from plotting import figure_to_frame
    from module.parcor import plot
    fig = plot(1, {"pestpath": "dummy"}, "20", "6", "0.98", "RdBu", True)
    df = figure_to_frame(_as_browser_json(fig))
    assert df.shape == (20, 20)
    assert list(df.index) == list(df.columns)
    np.testing.assert_allclose(np.diag(df.values), 1.0)

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_csv_bars_and_sankey(mock_pz):
    from plotting import figure_to_frame
    from module.obsconf import plot, plot_sankey
    bar, heat, opts, sel = plot(1, {"pestpath": "dummy"}, "6", "weight", "obsgrp", "Parameters", "10", "10")
    df = figure_to_frame(_as_browser_json(bar))
    assert set(df.trace) == {"Pull up (wants increase)", "Pull down (wants decrease)", "Net pull"}
    assert (df[df.trace == "Pull up (wants increase)"].y >= 0).all()
    assert len(df) == 30
    sk = figure_to_frame(_as_browser_json(plot_sankey(sel[:2], "obsname", "5", True, {"pestpath": "dummy"}, "6", "weight", "Parameters")))
    assert {"source", "target", "value", "details"} <= set(sk.columns)
    assert {t.split()[0] for t in sk.target} <= set(sel[:2])   # labels also carry the up/down split

@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_csv_includes_table(mock_pz):
    from plotting import figure_to_frame
    from module.scatterplot import plot
    fig = plot(1, {"pestpath": "dummy"}, "grpSite", _MIXED_SITE, False, "Hide", "Jet", False)
    df = figure_to_frame(_as_browser_json(fig))
    assert "RMSE" in set(df.iloc[:, -1].astype(str)) or "Statistics<br>(weight > 0)" in df.columns

def test_csv_message_figure_has_no_data():
    from plotting import figure_to_frame, message_figure
    assert figure_to_frame(_as_browser_json(message_figure("Load data"))) is None
    assert figure_to_frame(None) is None

def test_every_graph_has_csv_download():
    from module import apps
    import dash
    graphs = [g.id for page in apps.values() for g in _graphs(page.layout)]
    outputs = list(dash._callback.GLOBAL_CALLBACK_MAP)
    for gid in graphs:
        assert any(o.startswith(gid + "-csv.data") for o in outputs), gid


@patch("module.scatterplot.parse_zip", return_value=GOOD_RESULT)
def test_scatterplot_hovermode_option(mock_pz):
    from module.scatterplot import plot
    args = (1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, False, "Greyout", "Jet", False)
    assert plot(*args).layout.hovermode == "closest"              # default
    assert plot(*args, "x unified").layout.hovermode == "x unified"


# ---------------------------------------------------------------------------
# Scatter / Time Series: iteration selection
# ---------------------------------------------------------------------------

def test_update_rei_iter_defaults_to_last():
    from layout import update_rei_iter
    with patch("layout.parse_zip", return_value=GOOD_RESULT):
        options, value = update_rei_iter({"pestpath": "dummy"})
    assert options == ["All"] + [str(i) for i in range(0, _N_ITER + 1)]
    assert value == [str(_N_ITER)]

@pytest.mark.parametrize("module", ["scatterplot", "tsplot"])
def test_plot_selected_iterations(module):
    import importlib
    plot = importlib.import_module(f"module.{module}").plot
    def run(iters):
        with patch(f"module.{module}.parse_zip", return_value=GOOD_RESULT):
            if module == "scatterplot":
                fig = plot(1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, False, "Greyout", "Jet", False, "closest", iters)
            else:
                fig = plot(1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, "Greyout", "Lines", "Jet", False, iters)
        return fig, {t.name: t for t in fig.data if str(t.name).startswith("Iter")}
    fig, last = run(["6"])
    assert list(last) == ["Iter 6"]
    table = next(t for t in fig.data if t.type == "table")
    assert list(table.header.values)[1:] == ["Iter 6"]            # statistics for the selection only
    _, two = run(["2", "6"])
    assert list(two) == ["Iter 2", "Iter 6"]
    _, every = run(["All"])
    assert len(every) == _N_ITER + 1
    # an iteration keeps its color whatever else is selected
    color = lambda t: t.line.color if module == "tsplot" else t.marker.color
    assert color(last["Iter 6"]) == color(every["Iter 6"]) == color(two["Iter 6"])


@patch("module.tsplot.parse_zip", return_value=GOOD_RESULT)
def test_tsplot_hovermode_option(mock_pz):
    from module.tsplot import plot
    args = (1, {"pestpath": "dummy"}, "grpSite", _FIRST_SITE, "Greyout", "Lines", "Jet", False, ["6"])
    assert plot(*args).layout.hovermode == "closest"              # default
    assert plot(*args, "x unified").layout.hovermode == "x unified"


# ---------------------------------------------------------------------------
# Identifiability: EIGTHRESH is an eigenvalue ratio (squared singular values)
# ---------------------------------------------------------------------------

@patch("module.parident.parse_zip", return_value=GOOD_RESULT)
def test_parident_cutoff_uses_eigenvalue_ratio(mock_pz):
    import re
    import numpy as np
    from pst import get_weighted_jco
    from module.parident import plot
    fig = plot(1, {"pestpath": "dummy"}, "6", "1e-5", "Vertical")
    kept, total = map(int, re.search(r"(\d+) of (\d+)", fig.layout.title.text).groups())
    s = np.linalg.svd(get_weighted_jco(_obs, _jcos[6]).values, compute_uv=False)
    assert kept == int(((s / s[0])**2 >= 1e-5).sum()) and total == len(s)
    assert kept < int((s / s[0] >= 1e-5).sum())            # a singular-value cutoff would keep more
    f = np.asarray(fig.data[0].x, dtype=float)
    assert ((f > -1e-9) & (f < 1 + 1e-9)).all()


# ---------------------------------------------------------------------------
# every page uses the selected weight column (the example has weight and weight_orig)
# ---------------------------------------------------------------------------

_WEIGHT_PAGES = {
    "parsen":      lambda plot, w: plot(1, {"pestpath": "x"}, "L2 Norm", "Obs Groups", "obsgrp", "10", "Jet", [_FIRST_PAR], w),
    "obssen":      lambda plot, w: plot(1, {"pestpath": "x"}, "L2 Norm", "Obs Groups", "obsgrp", "10", "Jet", ["head"], w),
    "parcor":      lambda plot, w: plot(1, {"pestpath": "x"}, "20", "6", "0.98", "RdBu", True, w),
    "obssim":      lambda plot, w: plot(1, {"pestpath": "x"}, "20", "6", "0.98", "RdBu", True, True, w),
    "parident":    lambda plot, w: plot(1, {"pestpath": "x"}, "6", "1e-5", "Vertical", w),
    "scatterplot": lambda plot, w: plot(1, {"pestpath": "x"}, "grpType", "head", False, "Hide", "Jet", False, "closest", ["6"], w),
    "tsplot":      lambda plot, w: plot(1, {"pestpath": "x"}, "grpSite", _MIXED_SITE, "Hide", "Lines", "Jet", False, ["6"], "closest", w),
}

def _zero_some_original_weights():
    """the example run with weight_orig = 0 for every other observation, so 'zero weight' differs by column"""
    def zero(df):
        df = df.copy()
        df.loc[df.index[::2], "weight_orig"] = 0.0
        return df
    obs = _obs.copy()
    obs["weight_orig"] = obs.obsname.map(zero(_obs).set_index("obsname").weight_orig)
    reis = {k: v.assign(weight_orig=v.obsname.map(obs.set_index("obsname").weight_orig)) for k, v in _reis.items()}
    return (_pst, _par, obs, _pars, reis, _jcos)

@pytest.mark.parametrize("page", list(_WEIGHT_PAGES))
def test_page_uses_selected_weight_column(page):
    import importlib
    module = importlib.import_module(f"module.{page}")
    # scatter / time series only use the weight to tell zero-weight targets apart
    result = _zero_some_original_weights() if page in ("scatterplot", "tsplot") else GOOD_RESULT
    with patch(f"module.{page}.parse_zip", return_value=result):
        default = _WEIGHT_PAGES[page](module.plot, "weight")
        original = _WEIGHT_PAGES[page](module.plot, "weight_orig")
    assert len(default.data) > 0
    assert default.to_json() != original.to_json()      # weight_orig changes the result

def test_weight_selector_options():
    from layout import update_weights
    with patch("layout.parse_zip", return_value=GOOD_RESULT):
        assert update_weights({"pestpath": "x"}, "weight") == [["weight", "weight_orig"], "weight"]


# ---------------------------------------------------------------------------
# obsconf next step check
# ---------------------------------------------------------------------------

@patch("module.obsconf.parse_zip", return_value=GOOD_RESULT)
def test_obsconf_next_step(mock_pz):
    from module.obsconf import plot_next_step
    fig = plot_next_step(1, {"pestpath": "dummy"}, str(_N_ITER))
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 1 and list(bars[0].x) == [str(i) for i in range(1, _N_ITER + 1)]
    assert all(0 <= y <= 1 for y in bars[0].y)
    points = sum(len(t.x) for t in fig.data if t.type == "scatter" and t.x[0] is not None)
    assert points > 0

@patch("module.obsconf.parse_zip", return_value=(_pst, _par, _obs, {}, _reis, _jcos))
def test_obsconf_next_step_without_par_files(mock_pz):
    from module.obsconf import plot_next_step
    fig = plot_next_step(1, {"pestpath": "dummy"}, "2")
    assert ".par.2" in _message(fig)

def test_obsconf_next_step_no_clicks():
    from module.obsconf import plot_next_step
    assert isinstance(plot_next_step(None, None, "1"), go.Figure)
