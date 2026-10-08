"""
Shared fixtures for the PEST_HP_Postproc2 test suite.

All data is loaded once per session from testdata/example_pst.zip so every
test file works against the same real PEST run (Napa2026_04, 6 iterations).
"""
import pathlib
import pytest
from utils import read_pest_zip

TESTDATA_ZIP = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"


@pytest.fixture(scope="session")
def pest_data():
    """Full parse result loaded once for the entire test session."""
    pst, par, obs, pars, reis, jcos = read_pest_zip(str(TESTDATA_ZIP), True, True, True)
    return pst, par, obs, pars, reis, jcos


@pytest.fixture(scope="session")
def pst_name(pest_data):
    return pest_data[0]


@pytest.fixture(scope="session")
def par_df(pest_data):
    return pest_data[1]


@pytest.fixture(scope="session")
def obs_df(pest_data):
    return pest_data[2]


@pytest.fixture(scope="session")
def pars_dict(pest_data):
    return pest_data[3]


@pytest.fixture(scope="session")
def reis_dict(pest_data):
    return pest_data[4]


@pytest.fixture(scope="session")
def jcos_dict(pest_data):
    return pest_data[5]


@pytest.fixture(scope="session")
def jco_df(pest_data):
    """Jacobian for iteration 1."""
    return pest_data[5][1]


@pytest.fixture(scope="session")
def rei_df(pest_data):
    """Residuals for iteration 1."""
    return pest_data[4][1]


@pytest.fixture(scope="session")
def parse_result(pest_data):
    """Full (pst, par, obs, pars, reis, jcos) tuple — use as mock return value."""
    return pest_data
