"""Tests for the Residual Map page, using the example run with a synthetic site.csv."""
import pathlib
import shutil
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

ZIP = pathlib.Path(__file__).parent.parent / "testdata" / "example_pst.zip"


@pytest.fixture(scope="module")
def mapped_data(tmp_path_factory):
    """the example zip plus a site.csv with made-up coordinates for every grpSite value"""
    import utils
    from tests.test_utils import _rewrite_zip
    tmp = tmp_path_factory.mktemp("resmap")
    path = tmp / "run_with_sites.zip"
    path.write_bytes(_rewrite_zip().getvalue())     # the example run without any site.csv of its own
    obs = pd.read_csv(zipfile.ZipFile(ZIP).open("obs.csv"))
    sites = sorted(obs.grpSite.dropna().astype(str).unique())
    rng = np.random.default_rng(0)
    site_csv = pd.DataFrame(dict(site=sites, lat=40 + rng.random(len(sites)), lon=-100 + rng.random(len(sites)),
                                 type="test"))
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("site.csv", site_csv.to_csv(index=False))
    utils.clear_cache()
    parsed = utils.parse_zip({"pestpath": str(path)})[2]
    weighted = parsed[(parsed.weight > 0) & parsed.obsgrp.isin(["head", "flow"])]
    return {"pestpath": str(path)}, weighted.grpSite.dropna().nunique()


def _plot(data, **kw):
    from module.resmap import plot
    args = dict(clicks=1, basemap="None", opacity=70, layer_on=[], data=data, layers=[], iiter="6",
                site_col="grpSite", weight="weight", groups=["head", "flow"], years=None, no_years=True,
                stat="mean", size_by="Observations")
    args.update(kw)
    return plot(**args)


def test_site_stats():
    from module.resmap import site_stats
    rei = pd.DataFrame(dict(residual=[1.0, -3.0, 2.0, 5.0], weight=[1.0, 2.0, 1.0, 0.0],
                            obsgrp=["head"] * 4, grpSite=["A", "A", "B", "B"]))
    s = site_stats(rei, "grpSite", "weight")
    assert s.loc["a", "n"] == 2 and s.loc["b", "n"] == 1          # zero weight excluded; sites lower case
    assert s.loc["a", "mean"] == -1.0 and s.loc["a", "wmean"] == (1 - 6) / 2
    assert np.isclose(s.loc["a", "rmse"], np.sqrt(5)) and s.loc["a", "mae"] == 2.0


def test_map_shows_every_located_site(mapped_data):
    data, n_sites = mapped_data
    fig, note = _plot(data)
    (trace,) = fig.data
    assert isinstance(trace, go.Scattermap)
    assert len(trace.lat) == n_sites and note[0].startswith(f"{n_sites} sites mapped")
    assert note[2].startswith("Moran's I of mean residual =")
    assert trace.marker.cmid == 0                                  # diverging around zero for a signed statistic


def test_map_change_in_rmse_and_filters(mapped_data):
    data, n_sites = mapped_data
    fig, _ = _plot(data, stat="drmse", groups=["head"], years=[2000, 2010], no_years=False, size_by="RMSE")
    assert 0 < len(fig.data[0].lat) < n_sites                      # flow gauges and other years filtered out
    assert "Change in RMSE" in fig.layout.title.text


def test_basemap_and_layers(mapped_data, tmp_path, monkeypatch):
    import utils
    import geo
    from tests.test_geo import _shapefile_zip, _napa_like_layers, _contents
    monkeypatch.setattr(utils, "UPLOAD_DIR", str(tmp_path / "uploads"))
    geo._read_layers_cached.cache_clear()
    key = utils.save_upload(_contents(_shapefile_zip(tmp_path, _napa_like_layers())), "zip")
    layers = [dict(key=key, ext="zip", index=i, name=n, kind=k, count=1, category=c)
              for i, (n, k, c) in enumerate([("domain", "fill", "Model domain"), ("stream", "line", "Stream")])]
    data, _ = mapped_data
    fig, _ = _plot(data, basemap="Satellite (USGS)", opacity=40, layers=layers, layer_on=[True, False])
    raster, *shapes = fig.layout.map.layers
    assert "USGSImageryOnly" in raster.source[0] and raster.opacity == 0.4
    assert [s.type for s in shapes] == ["fill", "line"]            # only the switched-on polygon layer
    assert shapes[0].color == "#bdbdbd"                              # styled as a model domain
    # layers are drawn in category order (domain at the bottom), whatever their upload order
    fig, _ = _plot(data, basemap="None", layers=layers[::-1], layer_on=[True, True])
    assert [s.color for s in fig.layout.map.layers] == ["#bdbdbd", "#737373", "#1f78b4"]
    fig, _ = _plot(data, basemap="None", layers=layers, layer_on=[False, False])
    assert len(fig.layout.map.layers) == 0


def test_messages_without_sites_or_clicks(tmp_path):
    from tests.test_utils import _rewrite_zip
    path = tmp_path / "no_sites.zip"
    path.write_bytes(_rewrite_zip().getvalue())
    fig, _ = _plot({"pestpath": str(path)})
    assert "site.csv" in fig.layout.annotations[0].text
    fig, _ = _plot({"pestpath": str(path)}, clicks=None)
    assert "Plot Map" in fig.layout.annotations[0].text


def test_click_opens_time_series():
    from module.resmap import open_time_series
    click = {"points": [{"customdata": ["napar_napa", "120"]}]}
    assert open_time_series(click, "grpSite", 2) == ("tsplot", "grpSite", "napar_napa", 3)


def test_time_series_keeps_site_chosen_on_map():
    from module.tsplot import upload_pest
    assert upload_pest(["a", "napar_napa"], "napar_napa") == "napar_napa"
    assert upload_pest(["a", "b"], "napar_napa") == "a"


def test_map_csv_has_coordinates(mapped_data):
    import json
    import plotly.io as pio
    from plotting import figure_to_frame
    data, n_sites = mapped_data
    fig, _ = _plot(data)
    df = figure_to_frame(json.loads(pio.to_json(fig)))
    assert {"lat", "lon"} <= set(df.columns) and len(df) == n_sites


def test_site_share_of_total_phi():
    from module.resmap import site_stats, total_phi
    rei = pd.DataFrame(dict(residual=[1.0, -3.0, 2.0, 5.0, 4.0], weight=[1.0, 2.0, 1.0, 0.0, 1.0],
                            obsgrp=["head", "head", "flow", "head", "regul1"],
                            grpSite=["A", "A", "B", "B", "R"]))
    assert total_phi(rei, "weight") == 1 + 36 + 4                  # zero weight and regularization excluded
    s = site_stats(rei, "grpSite", "weight")
    assert s.loc["a", "phi"] == 37 and np.isclose(s.loc["a", "phi_share"], 37 / 41)
    # filtering groups changes which observations a site sums, not the total it is compared with
    assert np.isclose(site_stats(rei, "grpSite", "weight", groups=["flow"]).loc["b", "phi_share"], 4 / 41)


def test_phi_share_in_hover_and_note(mapped_data):
    import utils
    from module.resmap import total_phi
    data, n_sites = mapped_data
    fig, note = _plot(data)
    trace = fig.data[0]
    assert "share of total φ" in trace.hovertemplate
    shares = [float(row[2].split("%")[0]) for row in trace.customdata]
    rei = utils.parse_zip(data, False, True, False)[4][6]
    weighted = rei[(rei.weight > 0) & ~rei.obsgrp.str.startswith("regul")]
    unsited = ((weighted.weight * weighted.residual)**2)[weighted.grpSite.isna()].sum() / total_phi(rei, "weight")
    assert np.isclose(sum(shares) / 100 + unsited, 1, atol=0.01)   # every bit of phi is accounted for
    assert "of the total measurement φ of iteration 6" in note[0]
