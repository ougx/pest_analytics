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
    tmp = tmp_path_factory.mktemp("resmap")
    path = tmp / "run_with_sites.zip"
    shutil.copy(ZIP, path)
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
    assert len(trace.lat) == n_sites and note.startswith(f"{n_sites} sites mapped")
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
    layers = [dict(key=key, ext="zip", index=i, name=n, kind=k, count=1)
              for i, (n, k) in enumerate([("domain", "fill"), ("stream", "line")])]
    data, _ = mapped_data
    fig, _ = _plot(data, basemap="Satellite (USGS)", opacity=40, layers=layers, layer_on=[True, False])
    raster, *shapes = fig.layout.map.layers
    assert "USGSImageryOnly" in raster.source[0] and raster.opacity == 0.4
    assert [s.type for s in shapes] == ["fill", "line"]            # only the switched-on polygon layer
    fig, _ = _plot(data, basemap="None", layers=layers, layer_on=[False, False])
    assert len(fig.layout.map.layers) == 0


def test_messages_without_sites_or_clicks():
    from module.resmap import plot
    fig, _ = _plot({"pestpath": str(ZIP)})
    assert "site.csv" in fig.layout.annotations[0].text
    fig, _ = _plot({"pestpath": str(ZIP)}, clicks=None)
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
