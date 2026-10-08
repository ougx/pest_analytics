"""Tests for layout.py helper functions with parse_zip mocked out.

FULL_RESULT is loaded once at module import from the real example zip so the
mock always returns data that reflects the actual PEST run structure.
"""
import pathlib
from unittest.mock import patch
import pytest
from dash import no_update

from utils import read_pest_zip
from layout import update_groups, update_weights, update_obs, update_niter

_ZIP = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"

# Load with par + par-iterations so update_niter tests are meaningful.
FULL_RESULT  = read_pest_zip(str(_ZIP), loadpar=True, loadrei=False, loadjco=False)
EMPTY_RESULT = ("",  None, None, {}, {}, {})

_PST, _PAR, _OBS, _PARS, _REIS, _JCOS = FULL_RESULT
_N_ITER = len(_PARS)          # 6 for the example zip
_OBS_GROUPS = sorted(_OBS.obsgrp.unique().tolist())   # ['flow', 'head']
_N_ACTIVE_OBS = int((_OBS.weight > 0).sum())          # 5161


# ---------------------------------------------------------------------------
# update_groups
# ---------------------------------------------------------------------------

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_groups_returns_list(mock_pz):
    cols, val = update_groups({"pestpath": "dummy"})
    assert isinstance(cols, list)

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_groups_includes_obsgrp(mock_pz):
    cols, val = update_groups({"pestpath": "dummy"})
    assert "obsgrp" in cols

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_groups_no_extra_grp_cols_in_real_data(mock_pz):
    # The example zip has no obs.csv, so only the 'obsgrp' column from PST is present
    cols, val = update_groups({"pestpath": "dummy"})
    assert cols == ["obsgrp"]

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_groups_default_value_obsgrp(mock_pz):
    cols, val = update_groups({"pestpath": "dummy"})
    assert val == "obsgrp"

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_groups_missing_default_falls_back(mock_pz):
    cols, val = update_groups({"pestpath": "dummy"}, default_value="nonexistent")
    assert val == "obsgrp"

def test_update_groups_none_data_returns_no_update():
    result = update_groups(None)
    assert result == [no_update, no_update]


# ---------------------------------------------------------------------------
# update_weights
# ---------------------------------------------------------------------------

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_weights_returns_weight_cols(mock_pz):
    cols, val = update_weights({"pestpath": "dummy"})
    assert "weight" in cols

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_weights_default_value(mock_pz):
    cols, val = update_weights({"pestpath": "dummy"}, default_value="weight")
    assert val == "weight"

def test_update_weights_none_data_returns_no_update():
    result = update_weights(None)
    assert result == [no_update, no_update]


# ---------------------------------------------------------------------------
# update_obs
# ---------------------------------------------------------------------------

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_obs_observations_mode_returns_grouped(mock_pz):
    result = update_obs({"pestpath": "dummy"}, "Observations", "obsgrp")
    assert isinstance(result, list)
    assert all("group" in item and "items" in item for item in result)

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_obs_observations_only_active(mock_pz):
    # filter_zeroweight=True by default → only weight>0 obs returned
    result = update_obs({"pestpath": "dummy"}, "Observations", "obsgrp")
    total = sum(len(grp["items"]) for grp in result)
    assert total == _N_ACTIVE_OBS

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_obs_groups_mode_returns_known_groups(mock_pz):
    result = update_obs({"pestpath": "dummy"}, "Obs Groups", "obsgrp")
    assert isinstance(result, list)
    assert all(isinstance(g, str) for g in result)
    assert set(result) == set(_OBS_GROUPS)

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_obs_groups_excludes_regul_prefix(mock_pz):
    result = update_obs({"pestpath": "dummy"}, "Obs Groups", "obsgrp")
    assert not any(g.startswith("regul") for g in result)

def test_update_obs_none_data_returns_no_update():
    result = update_obs(None, "Observations", "obsgrp")
    assert result is no_update


# ---------------------------------------------------------------------------
# update_niter
# ---------------------------------------------------------------------------

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_niter_returns_string_list(mock_pz):
    result = update_niter({"pestpath": "dummy"}, loadpar=True)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_niter_count_matches_par_iterations(mock_pz):
    result = update_niter({"pestpath": "dummy"}, loadpar=True)
    assert len(result) == _N_ITER
    assert result == [str(i + 1) for i in range(_N_ITER)]

@patch("layout.parse_zip", return_value=FULL_RESULT)
def test_update_niter_no_flag_returns_100_capped(mock_pz):
    # No loadpar/loadrei/loadjco → n = min(100,100,100) = 100
    result = update_niter({"pestpath": "dummy"})
    assert len(result) == 100

def test_update_niter_none_data_returns_none():
    # update_niter returns None (not no_update) when data is absent
    result = update_niter(None, loadpar=True)
    assert result is None
