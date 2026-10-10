"""The run-record parser (rec.py) and the Run Record page."""
import pathlib
import zipfile

import numpy as np
import pytest

from rec import parse_rec, read_rec

EXAMPLE = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"
REF = {"pestpath": str(EXAMPLE)}

SMALL = """
    Number of adjustable parameters                :     3
    Forgive model run failure during Jacobian runs               : yes

 OPTIMISATION ITERATION NO.        : 1
    Model calls so far             : 1
    Current regularisation weight factor                     :   1.0000
    Current value of measurement objective function          :  100.00
    Current value of regularisation objective function       :   2.0
    Starting phi for this iteration                          :  100.00
    Contribution to phi from observation group "head"        :  60.000
    Contribution to phi from observation group "flow"        :  40.000
    FRACPHIM-adjusted target measurement objective function  :  30.000
    Re-calculated regularisation weight factor               :  0.5000
    Contribution to phi from observation group "head"        :  60.000
    Parameter "p3" has no effect on observations.
        param "p2" frozen: - update vector out of bounds

        Lambda =   10.000     upgrade fraction =   1.0000     ----->
           Phi =  80.000      (  0.800 of starting phi)
     Meas. fn. =  80.000
    Regul. fn. =   3.0

        Lambda =   10.000     upgrade fraction =   2.0000     ----->
           Phi = cannot be calculated: model run failure

        Lambda =   1.0000     ----->
           Phi =  70.000      (  0.700 of starting phi)

    Maximum   factor change:  2.000     ["P1"]
    Maximum relative change:  1.000     ["P1"]

 OPTIMISATION ITERATION NO.        : 2
    Model calls so far             : 7
    Current value of measurement objective function          :  80.000
    Starting phi for this iteration                          :  80.000
"""


def test_parse_small_record():
    r = parse_rec(SMALL, "small.rec")
    it, lam = r["iterations"], r["lambdas"]
    assert list(it.index) == [1, 2] and not r["complete"]
    assert r["settings"] == {"adjustable_parameters": "3", "forgive_jacobian_failures": "yes"}
    one = it.loc[1]
    assert (one.model_calls, one.phi_measurement, one.phi_target, one.new_weight_factor) == (1, 100, 30, 0.5)
    assert (one.max_factor_change, one.max_factor_parameter) == (2.0, "p1")
    # three tests: one with a fraction, one failed, one classic PEST line without a fraction
    assert lam["lambda"].tolist() == [10, 10, 1] and lam.failed.tolist() == [False, True, False]
    assert np.isnan(lam.fraction.iloc[2]) and lam.phi_ratio.tolist()[0] == pytest.approx(0.8)
    # accepted: the test whose measurement phi is iteration 2's (80), not the lowest phi (70)
    assert (one.accepted_lambda, one.accepted_fraction, one.accepted_test) == (10, 1, 1)
    # 7 - 1 = 6 runs in iteration 1, 3 of them lambda tests
    assert (one.calls, one.lambda_runs, one.failed_lambda_runs, one.jacobian_runs) == (6, 3, 1, 3)
    assert r["groups"][["group", "phi"]].values.tolist() == [["head", 60.0], ["flow", 40.0]]   # first block only
    assert r["events"][["parameter", "event"]].values.tolist() == [["p3", "no effect"], ["p2", "frozen"]]


def test_example_record():
    r = read_rec(REF)
    it = r["iterations"]
    assert list(it.index) == list(range(1, 8)) and not r["complete"]
    assert (it.jacobian_runs.dropna() == 132).all()               # one run per adjustable parameter
    assert it.failed_lambda_runs.tolist()[:6] == [0, 3, 35, 46, 72, 55]
    assert it.accepted_test.notna().sum() == 6                     # matched for every complete iteration
    # the record's measurement phi agrees with the residual files (iteration N starts at .rei.(N-1))
    from utils import parse_zip
    from compare import measurement_phi
    reis = parse_zip(REF, False, True)[4]
    for n in (1, 4, 7):
        assert it.loc[n, "phi_measurement"] == pytest.approx(measurement_phi(reis[n - 1]), rel=1e-4)


def test_page(tmp_path):
    from module.runrec import read_record
    out = read_record(1, REF)
    assert "the record stops during iteration 7" in out[0][0].children
    assert len(out[2].data) > 6                                    # one panel per iteration, with failures
    without = tmp_path / "no_rec.zip"
    with zipfile.ZipFile(EXAMPLE) as src, zipfile.ZipFile(without, "w") as dst:
        for item in src.infolist():
            if not item.filename.endswith((".rec", ".jco.1", ".jco.2", ".jco.3", ".jco.4", ".jco.5", ".jco.6")):
                dst.writestr(item, src.read(item.filename))
    assert "No .rec file" in read_record(1, {"pestpath": str(without)})[0].children
    assert "No data loaded" in read_record(1, None)[0].children


# ---------------------------------------------------------------------------
# Regularisation page (review 7.5)
# ---------------------------------------------------------------------------

def test_regularisation_rows_use_pst_weights():
    import pandas as pd
    from module.regul import regularisation_rows
    obs = pd.DataFrame(dict(obsname=["h1", "r1"], weight=[1.0, 2.0], obsgrp=["head", "regul_obs"]))
    rei = pd.DataFrame(dict(obsname=["h1", "r1", "pi1", "pi2"], obsgrp=["head", "regul_obs", "regulhk", "regulhk"],
                            residual=[5.0, 0.5, 0.2, -1.0], weight=[1.0, 2.0, 0.01, 0.01]))
    prior = pd.DataFrame(dict(equation=["log(a) = 0", "log(a) - log(b) = 0"], weight_pst=[50.0, 10.0],
                              group=["regulhk", "regulhk"]), index=["pi1", "pi2"])
    rows = regularisation_rows(rei, obs, prior)
    assert list(rows.index) == ["r1", "pi1", "pi2"]                       # measurements left out
    assert rows.phi.tolist() == pytest.approx([1.0, 100.0, 100.0])        # (2*0.5)^2, (50*0.2)^2, (10*1)^2
    assert rows.loc["pi1", "equation"] == "log(a) = 0"
    assert regularisation_rows(rei, obs, None).loc["pi1", "phi"] == pytest.approx((0.01 * 0.2) ** 2)


def test_regularisation_page():
    from module.regul import plot
    summary, tradeoff, groups, equations = plot(1, REF, "6", "20")
    assert "70 prior-information equations" in summary.children
    assert sum(len(t.x) for t in tradeoff.data if t.name.startswith("Iteration")) == 844 - 211
    assert {t.name for t in groups.data} == {"regulhk", "regulvk", "regulss", "regulsy"}
    assert sum(len(t.y) for t in equations.data) == 20
    top = max((x, y) for t in equations.data for x, y in zip(t.x, t.y))
    assert top[1] == "kdcr01"
