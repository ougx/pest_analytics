"""Unit tests for utils.py — pure parsing helpers, no Dash dependency."""
import io
import pathlib
import numpy as np
import pandas as pd
import pytest

from utils import (
    isiterable,
    read_df,
    read_par,
    read_rei,
    read_jco,
    check_data,
    read_par_in_pst,
    read_obs_in_pst,
    parse_zip,
)

TESTDATA_ZIP = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"


# ---------------------------------------------------------------------------
# isiterable
# ---------------------------------------------------------------------------

def test_isiterable_list():
    assert isiterable([1, 2, 3]) is True

def test_isiterable_tuple():
    assert isiterable((1, 2)) is True

def test_isiterable_dict():
    assert isiterable({"a": 1}) is True

def test_isiterable_string_false():
    assert isiterable("hello") is False

def test_isiterable_int_false():
    assert isiterable(42) is False

def test_isiterable_none_false():
    assert isiterable(None) is False


# ---------------------------------------------------------------------------
# read_df
# ---------------------------------------------------------------------------

def test_read_df_with_header():
    text = "a  b  c\n1  2  3\n4  5  6\n"
    df = read_df(text, header=0)
    assert list(df.columns) == ["a", "b", "c"]
    assert len(df) == 2
    assert df["a"].tolist() == [1, 4]

def test_read_df_with_names():
    text = "1  2  3\n4  5  6\n"
    df = read_df(text, names=["x", "y", "z"])
    assert list(df.columns) == ["x", "y", "z"]
    assert df["z"].tolist() == [3, 6]

def test_read_df_whitespace_separator():
    text = "p1  1.0  1.0  0.0\np2  0.5  1.0  0.0\n"
    df = read_df(text, names=["parname", "parval", "scale", "offset"])
    assert df["parname"].tolist() == ["p1", "p2"]
    assert df["parval"].tolist() == [1.0, 0.5]


# ---------------------------------------------------------------------------
# read_par
# ---------------------------------------------------------------------------

def _make_par_bytes(*rows):
    lines = b"2\n"
    for row in rows:
        lines += row.encode() + b"\n"
    lines += b"\n"
    return io.BytesIO(lines)

def test_read_par_columns():
    f = _make_par_bytes("p1  1.0  1.0  0.0", "p2  0.5  1.0  0.0")
    df = read_par(f)
    assert list(df.columns) == ["parname", "parval", "scale", "offset"]

def test_read_par_values():
    f = _make_par_bytes("p1  1.0  1.0  0.0", "p2  0.5  1.0  0.0")
    df = read_par(f)
    assert df["parname"].tolist() == ["p1", "p2"]
    assert df["parval"].tolist() == [1.0, 0.5]

def test_read_par_single_row():
    f = _make_par_bytes("p1  2.5  1.0  0.0")
    df = read_par(f)
    assert len(df) == 1
    assert df["parval"].iloc[0] == 2.5


# ---------------------------------------------------------------------------
# read_rei
# ---------------------------------------------------------------------------

def _make_rei_bytes(*data_rows):
    lines = b"some header line\n"
    lines += b"NAME GROUP MEASURED MODELLED WEIGHT RESIDUAL\n"
    for row in data_rows:
        lines += row.encode() + b"\n"
    return io.BytesIO(lines)

def test_read_rei_columns():
    f = _make_rei_bytes(
        "obs1  grp1  1.5  1.6  1.0  -0.1",
        "obs2  grp1  2.5  2.3  1.0   0.2",
    )
    df = read_rei(f)
    assert "obsname" in df.columns
    assert "obsgrp" in df.columns
    assert "measured" in df.columns
    assert "modelled" in df.columns

def test_read_rei_values():
    f = _make_rei_bytes(
        "obs1  grp1  1.5  1.6  1.0  -0.1",
        "obs2  grp1  2.5  2.3  1.0   0.2",
    )
    df = read_rei(f)
    assert df["obsname"].tolist() == ["obs1", "obs2"]
    assert df["obsgrp"].tolist() == ["grp1", "grp1"]
    assert df["measured"].tolist() == pytest.approx([1.5, 2.5])

def test_read_rei_single_row():
    f = _make_rei_bytes("obs1  grp1  3.0  3.1  1.0  -0.1")
    df = read_rei(f)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# read_jco  (sparse / NESPAR > 0 format)
# ---------------------------------------------------------------------------

def _make_jco_bytes(npar, nobs, dense_data, par_names, obs_names):
    vals = []
    for i in range(nobs):
        for j in range(npar):
            if dense_data[i, j] != 0:
                vals.append((i, j, dense_data[i, j]))
    NESPAR, NXROW, NVAL = npar, nobs, len(vals)
    buf = np.array([NESPAR, NXROW, NVAL], dtype="i4").tobytes()
    for vi, vj, vval in vals:
        buf += np.array([(vi, vj, vval)], dtype=[("i", "i4"), ("j", "i4"), ("val", "f8")]).tobytes()
    for p in par_names:
        buf += p.encode().ljust(200)[:200]
    for o in obs_names:
        buf += o.encode().ljust(200)[:200]
    return buf

def test_read_jco_shape():
    data = np.array([[0.5, 0.2], [0.3, 0.8], [0.0, 0.1]])
    buf = _make_jco_bytes(2, 3, data, ["p1", "p2"], ["obs1", "obs2", "obs3"])
    df = read_jco(buf)
    assert df.shape == (3, 2)

def test_read_jco_index_and_columns():
    # read_jco returns rows=observations, cols=parameters
    data = np.array([[0.5, 0.2], [0.3, 0.8]])
    buf = _make_jco_bytes(2, 2, data, ["p1", "p2"], ["obs1", "obs2"])
    df = read_jco(buf)
    assert list(df.index) == ["obs1", "obs2"]
    assert list(df.columns) == ["p1", "p2"]

def test_read_jco_values():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    buf = _make_jco_bytes(2, 2, data, ["p1", "p2"], ["obs1", "obs2"])
    df = read_jco(buf)
    np.testing.assert_allclose(df.values, data, atol=1e-5)


# ---------------------------------------------------------------------------
# check_data
# ---------------------------------------------------------------------------

def test_check_data_none():
    assert check_data(None) is False

def test_check_data_empty_dict():
    assert check_data({}) is False

def test_check_data_all_falsy():
    assert check_data({"filename": None, "contents": None, "pestpath": None}) is False

def test_check_data_pestpath_present():
    assert check_data({"pestpath": "some/path", "contents": None}) is True

def test_check_data_contents_present():
    assert check_data({"filename": "f.zip", "contents": "data==", "pestpath": None}) is True

def test_check_data_non_dict():
    assert check_data("not a dict") is False


# ---------------------------------------------------------------------------
# read_par_in_pst / read_obs_in_pst  (synthetic minimal PST)
# ---------------------------------------------------------------------------

def _make_pst_bytes():
    return io.BytesIO(
        b"* control data\n"
        b"restart estimation\n"
        b"2 4 0 0 0\n"
        b"* parameter data\n"
        b"p1  log  factor  1.0  0.1  10.0  grp1  1.0  0.0  1\n"
        b"p2  none factor  0.5  0.01  5.0  grp1  1.0  0.0  1\n"
        b"* observation data\n"
        b"obs1  1.5  1.0  grp1\n"
        b"obs2  2.5  1.0  grp1\n"
        b"obs3  3.5  1.0  grp2\n"
        b"obs4  0.0  0.0  regul\n"
        b"* end\n"
    )

def test_read_par_in_pst_columns():
    par = read_par_in_pst(_make_pst_bytes())
    expected = "parname partrans parchglim parval0 parlbnd parubnd pargrp scale offset dercom tiedto".split()
    assert list(par.columns) == expected

def test_read_par_in_pst_values():
    par = read_par_in_pst(_make_pst_bytes())
    assert par["parname"].tolist() == ["p1", "p2"]
    assert par["partrans"].tolist() == ["log", "none"]
    assert par["parval0"].tolist() == pytest.approx([1.0, 0.5])

def test_read_par_in_pst_lowercase():
    par = read_par_in_pst(_make_pst_bytes())
    assert par["parname"].str.islower().all()
    assert par["partrans"].str.islower().all()

def test_read_obs_in_pst_columns():
    obs = read_obs_in_pst(_make_pst_bytes())
    assert list(obs.columns) == ["obsname", "obsval", "weight", "obsgrp"]

def test_read_obs_in_pst_values():
    obs = read_obs_in_pst(_make_pst_bytes())
    assert obs["obsname"].tolist() == ["obs1", "obs2", "obs3", "obs4"]
    assert obs["obsval"].tolist() == pytest.approx([1.5, 2.5, 3.5, 0.0])
    assert obs["weight"].tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])

def test_read_obs_in_pst_lowercase():
    obs = read_obs_in_pst(_make_pst_bytes())
    assert obs["obsname"].str.islower().all()
    assert obs["obsgrp"].str.islower().all()


def _pst_with_blank_lines():
    # blank lines inside sections, CRLF line ends, a tied parameter
    text = _make_pst_bytes().getvalue().replace(b"p2  none", b"p2  tied").replace(b"\n", b"\r\n")
    text = text.replace(b"* observation data\r\n", b"p2 p1\r\n\r\n* observation data\r\n\r\n")
    return text.replace(b"obs2", b"\r\n   \r\nobs2")

def test_read_par_in_pst_skips_blank_lines():
    par = read_par_in_pst(io.BytesIO(_pst_with_blank_lines()))
    assert par["parname"].tolist() == ["p1", "p2"]
    assert par["tiedto"].tolist() == ["", "p1"]

def test_read_obs_in_pst_skips_blank_lines():
    obs = read_obs_in_pst(io.BytesIO(_pst_with_blank_lines()))
    assert obs["obsname"].tolist() == ["obs1", "obs2", "obs3", "obs4"]

def test_read_pst_missing_or_external_section():
    with pytest.raises(ValueError, match="observation data"):
        read_obs_in_pst(io.BytesIO(b"* control data\n* parameter data\np1 log factor 1 0.1 10 g 1 0 1\n"))
    with pytest.raises(ValueError, match="external"):
        read_par_in_pst(io.BytesIO(b"* parameter data external\npar.dat\n"))

def test_read_par_stops_at_blank_line_with_crlf():
    f = io.BytesIO(b"single point\r\np1  1.0  1.0  0.0\r\np2  0.5  1.0  0.0\r\n\r\nnot a parameter\r\n")
    assert read_par(f)["parname"].tolist() == ["p1", "p2"]

def test_read_rei_stops_at_blank_line():
    f = io.BytesIO(b"header\r\nName Group Measured Modelled Residual Weight\r\n"
                   b"obs1 g 1.5 1.6 -0.1 1.0\r\n\r\nsummary text\r\n")
    assert read_rei(f)["obsname"].tolist() == ["obs1"]

def test_read_rei_without_header():
    with pytest.raises(ValueError, match="header"):
        read_rei(io.BytesIO(b"no table here\n1 2 3\n"))


# ---------------------------------------------------------------------------
# parse_zip — uses the real example zip
# ---------------------------------------------------------------------------

def test_parse_zip_none_data():
    pst, par, obs, pars, reis, jcos = parse_zip(None)
    assert pst is None

def test_parse_zip_empty_dict():
    pst, par, obs, pars, reis, jcos = parse_zip({})
    assert pst is None

def test_parse_zip_all_none_values():
    pst, par, obs, pars, reis, jcos = parse_zip({"pestpath": None, "contents": None})
    assert pst is None

def test_parse_zip_real_pst_name():
    pst, par, obs, pars, reis, jcos = parse_zip({"pestpath": str(TESTDATA_ZIP)})
    assert pst == "Napa2026_04"

def test_parse_zip_real_obs_columns():
    pst, par, obs, pars, reis, jcos = parse_zip({"pestpath": str(TESTDATA_ZIP)})
    assert set(["obsname", "obsval", "weight", "obsgrp"]).issubset(obs.columns)

def test_parse_zip_real_par_columns():
    pst, par, obs, pars, reis, jcos = parse_zip({"pestpath": str(TESTDATA_ZIP)})
    expected = {"parname", "partrans", "parval0", "pargrp"}
    assert expected.issubset(par.columns)

def test_parse_zip_real_loads_jcos():
    pst, par, obs, pars, reis, jcos = parse_zip(
        {"pestpath": str(TESTDATA_ZIP)}, loadjco=True
    )
    assert len(jcos) == 6

def test_parse_zip_real_loads_reis():
    pst, par, obs, pars, reis, jcos = parse_zip(
        {"pestpath": str(TESTDATA_ZIP)}, loadrei=True
    )
    # Iteration 0 comes from .hp file; iterations 1–6 from .rei files
    assert 0 in reis
    assert len(reis) == 7


# ---------------------------------------------------------------------------
# read_pest_zip: file naming and obs.csv/par.csv handling
# ---------------------------------------------------------------------------

def _rewrite_zip(rename=None, extra=None, obs_csv=None):
    """
    Copy the example zip with renamed members, extra files and/or a modified obs.csv.
    Any site.csv in the example zip is left out, so tests choose their own sites.
    """
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(TESTDATA_ZIP) as src, zipfile.ZipFile(buf, "w") as dst:
        for name, data in (extra or {}).items():
            dst.writestr(name, data)
        for n in src.namelist():
            if n.lower().endswith("site.csv"):
                continue
            data = src.read(n)
            if n == "obs.csv" and obs_csv:
                data = obs_csv(pd.read_csv(io.BytesIO(data))).to_csv(index=False).encode()
            dst.writestr(rename(n) if rename else n, data)
    buf.seek(0)
    return buf

def _shift_iterations(n, by=9):
    for ext in ("jco", "rei", "par"):
        if f".{ext}." in n:
            base, k = n.rsplit(".", 1)
            return f"{base}.{int(k) + by}"
    return n

def test_read_pest_zip_multi_digit_iterations():
    from utils import read_pest_zip
    pst, par, obs, pars, reis, jcos = read_pest_zip(_rewrite_zip(rename=_shift_iterations), True, True, True)
    assert sorted(jcos) == list(range(10, 16))
    assert sorted(pars) == list(range(10, 16))
    assert sorted(reis) == [0] + list(range(10, 16))

def test_read_pest_zip_pst_name_keeps_trailing_letters():
    from utils import read_pest_zip
    rename = lambda n: "sub/model_test.pst" if n.endswith(".pst") else n
    assert read_pest_zip(_rewrite_zip(rename=rename))[0] == "model_test"

def test_read_pest_zip_numeric_obs_group_column():
    from utils import read_pest_zip
    def add_layer(ob):
        ob["grpLayer"] = 3
        return ob
    obs = read_pest_zip(_rewrite_zip(obs_csv=add_layer))[2]
    assert set(obs.grpLayer.dropna()) == {"3"}

def test_read_pest_zip_par_csv_before_pst():
    from utils import read_pest_zip
    extra = {"par.csv": "parname,grpZone\nKDCR01,North\n"}
    par = read_pest_zip(_rewrite_zip(extra=extra))[1]
    assert par.set_index("parname").loc["kdcr01", "grpZone"] == "north"

def test_read_pest_zip_keeps_prior_information_rows():
    from utils import read_pest_zip
    pst, par, obs, pars, reis, jcos = read_pest_zip(str(TESTDATA_ZIP), False, True, False)
    rei = reis[6]
    prior = rei[~rei.obsname.isin(obs.obsname)]
    assert len(prior) == 70
    assert prior.obsgrp.str.startswith("regul").all()
    assert (prior.grpType == prior.obsgrp).all()
    assert prior.weight.notna().all()


# ---------------------------------------------------------------------------
# parse_zip caching and server-side uploads
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    import utils
    monkeypatch.setattr(utils, "UPLOAD_DIR", str(tmp_path / "uploads"))
    utils.clear_cache()
    yield utils
    utils.clear_cache()

def _count_reads(utils, monkeypatch):
    calls = []
    real = utils.read_pest_zip
    def counting(*args, **kwargs):
        calls.append(args[1:])
        return real(*args, **kwargs)
    monkeypatch.setattr(utils, "read_pest_zip", counting)
    return calls

def test_parse_zip_reads_file_once(fresh_cache, monkeypatch):
    utils = fresh_cache
    calls = _count_reads(utils, monkeypatch)
    data = {"pestpath": str(TESTDATA_ZIP)}
    a = utils.parse_zip(data, False, True, True)
    b = utils.parse_zip(data, True, True, True)
    c = utils.parse_zip(data, False, False, False)   # served by the result with Jacobians
    assert len(calls) == 1
    assert a[5][1] is b[5][1]                         # same Jacobian object, not re-read
    assert c[3] == {} and c[4] == {} and c[5] == {}   # flags still honored
    assert len(b[3]) == 6 and len(b[4]) == 7

def test_parse_zip_without_jco_does_not_load_jacobians(fresh_cache, monkeypatch):
    utils = fresh_cache
    calls = _count_reads(utils, monkeypatch)
    data = {"pestpath": str(TESTDATA_ZIP)}
    utils.parse_zip(data, True, True, False)
    utils.parse_zip(data, False, False, False)
    assert calls == [(True, True, False)]
    utils.parse_zip(data, False, False, True)
    assert calls[-1] == (True, True, True)

def test_parse_zip_rereads_updated_file(fresh_cache, monkeypatch, tmp_path):
    import os, shutil
    utils = fresh_cache
    calls = _count_reads(utils, monkeypatch)
    path = tmp_path / "run.zip"
    shutil.copy(TESTDATA_ZIP, path)
    utils.parse_zip({"pestpath": str(path)})
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))   # PEST re-run updates the zip
    utils.parse_zip({"pestpath": str(path)})
    assert len(calls) == 2

def test_parse_zip_concurrent_callbacks_parse_once(fresh_cache, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    utils = fresh_cache
    calls = _count_reads(utils, monkeypatch)
    data = {"pestpath": str(TESTDATA_ZIP)}
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda _: utils.parse_zip(data, True, True, True), range(8)))
    assert len(calls) == 1
    assert all(r[0] == "Napa2026_04" for r in results)

def test_parse_zip_cache_is_bounded(fresh_cache, monkeypatch, tmp_path):
    import shutil
    utils = fresh_cache
    monkeypatch.setattr(utils, "CACHE_SIZE", 2)
    for i in range(3):
        path = tmp_path / f"run{i}.zip"
        shutil.copy(TESTDATA_ZIP, path)
        utils.parse_zip({"pestpath": str(path)})
    assert len(utils._cache) == 2

def test_save_upload_and_parse_by_key(fresh_cache):
    import base64
    utils = fresh_cache
    contents = "data:application/zip;base64," + base64.b64encode(TESTDATA_ZIP.read_bytes()).decode()
    key = utils.save_upload(contents)
    assert key == utils.save_upload(contents)         # same file, same key, stored once
    assert utils.parse_zip({"upload": key})[0] == "Napa2026_04"

def test_upload_key_is_validated(fresh_cache):
    utils = fresh_cache
    assert utils.upload_path("../../etc/passwd") is None
    assert utils.upload_path("0" * 32) is None        # well-formed but unknown
    assert utils.parse_zip({"upload": "../x"})[0] is None


# ---------------------------------------------------------------------------
# site.csv (site locations for the residual map); synthetic coordinates
# ---------------------------------------------------------------------------

def _zip_with_sites(tmp_path, text):
    path = tmp_path / "run.zip"
    path.write_bytes(_rewrite_zip(extra={"Data/site.csv": text}).getvalue())
    return {"pestpath": str(path)}

def test_read_sites(fresh_cache, tmp_path):
    utils = fresh_cache
    data = _zip_with_sites(tmp_path, "Site,Latitude,Longitude,Type\nNapaR_Napa,38.1,-121.9,gage\nW1,38.2,-122.0,well\n")
    sites = utils.read_sites(data)
    assert list(sites.site) == ["napar_napa", "w1"]          # lower case, like the grp* columns
    assert list(sites.columns) == ["site", "lat", "lon", "type"]
    assert sites.lat.tolist() == [38.1, 38.2]

def test_read_sites_absent(fresh_cache, tmp_path):
    path = tmp_path / "no_sites.zip"
    path.write_bytes(_rewrite_zip().getvalue())
    assert fresh_cache.read_sites({"pestpath": str(path)}) is None
    assert fresh_cache.read_sites(None) is None

def test_read_sites_missing_columns(fresh_cache, tmp_path):
    with pytest.raises(ValueError, match="missing: lon"):
        fresh_cache.read_sites(_zip_with_sites(tmp_path, "site,lat,x\nw1,38.2,6479526\n"))

def test_read_sites_projected_coordinates_rejected(fresh_cache, tmp_path):
    with pytest.raises(ValueError, match="latitude/longitude"):
        fresh_cache.read_sites(_zip_with_sites(tmp_path, "site,lat,lon\nw1,1887920.3,6479526.1\n"))


# ---------------------------------------------------------------------------
# warnings about files the app does not read
# ---------------------------------------------------------------------------

def _as_file(tmp_path, buf):
    path = tmp_path / "run.zip"
    path.write_bytes(buf.getvalue())
    return {"pestpath": str(path)}

def test_check_pest_files_ok(fresh_cache):
    assert fresh_cache.check_pest_files({"pestpath": str(TESTDATA_ZIP)}) == []
    assert fresh_cache.check_pest_files(None) == []

def test_check_pest_files_pestpp(fresh_cache, tmp_path):
    extra = {"case.jcb": b"x", "case.3.par": b"x", "case.0.par.csv": b"x"}
    (warning,) = fresh_cache.check_pest_files(_as_file(tmp_path, _rewrite_zip(extra=extra)))
    assert warning.startswith("3 file(s) look like PEST++ output")

def test_check_pest_files_missing_iterations(fresh_cache, tmp_path):
    rename = lambda n: n.replace(".jco.", ".jacobian.")
    (warning,) = fresh_cache.check_pest_files(_as_file(tmp_path, _rewrite_zip(rename=rename)))
    assert ".jco.N" in warning and "JCOSAVEITN" in warning and ".rei.N" not in warning
