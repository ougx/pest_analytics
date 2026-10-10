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
    assert list(module.sections["Compare"]) == ["cmpsetup"]
