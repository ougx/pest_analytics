"""Tests for geo.py: map layers from uploaded shapefiles / GeoJSON (synthetic data)."""
import base64
import io
import json
import zipfile
import pytest


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    import utils
    import geo
    monkeypatch.setattr(utils, "UPLOAD_DIR", str(tmp_path / "uploads"))
    geo._read_layers_cached.cache_clear()
    return tmp_path


def _contents(raw, mime="application/zip"):
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def _shapefile_zip(tmp_path, layers, with_prj=True):
    """zip of shapefiles: layers = {name: GeoDataFrame}"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, gdf in layers.items():
            folder = tmp_path / name
            folder.mkdir()
            gdf.to_file(folder / f"{name}.shp")
            for f in folder.iterdir():
                if with_prj or f.suffix != ".prj":
                    z.write(f, f"shapes/{f.name}")
    return buf.getvalue()


def _napa_like_layers():
    import geopandas as gpd
    from shapely.geometry import Polygon, LineString
    # EPSG:2226 (California State Plane zone II, US ft)
    x0, y0 = 6450000, 1930000
    domain = gpd.GeoDataFrame(geometry=[Polygon([(x0, y0), (x0 + 30000, y0), (x0 + 30000, y0 + 40000), (x0, y0 + 40000)])], crs=2226)
    stream = gpd.GeoDataFrame(geometry=[LineString([(x0, y0, 10), (x0 + 5000, y0 + 8000, 12)])], crs=2226)   # 3D
    return {"domain": domain, "stream": stream}


def test_read_layers_reprojects_shapefiles(uploads):
    from utils import save_upload
    from geo import read_layers
    key = save_upload(_contents(_shapefile_zip(uploads, _napa_like_layers())), "zip")
    layers = {l["name"]: l for l in read_layers(key, "zip")}
    assert layers["domain"]["kind"] == "fill" and layers["stream"]["kind"] == "line"
    lon, lat = layers["domain"]["geojson"]["features"][0]["geometry"]["coordinates"][0][0]
    assert 38 < lat < 39 and -123 < lon < -122            # converted to latitude/longitude
    assert len(layers["stream"]["geojson"]["features"][0]["geometry"]["coordinates"][0]) == 2   # 3D -> 2D


def test_read_layers_without_prj_must_be_lonlat(uploads):
    from utils import save_upload
    from geo import read_layers
    key = save_upload(_contents(_shapefile_zip(uploads, _napa_like_layers(), with_prj=False)), "zip")
    with pytest.raises(ValueError, match="no coordinate system"):
        read_layers(key, "zip")


def test_read_layers_geojson(uploads):
    from utils import save_upload
    from geo import read_layers
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [-122.3, 38.4]}}]}
    key = save_upload(_contents(json.dumps(gj).encode(), "application/geo+json"), "geojson")
    (layer,) = read_layers(key, "geojson")
    assert layer["kind"] == "circle" and layer["count"] == 1


def test_read_layers_missing_upload(uploads):
    from geo import read_layers
    with pytest.raises(ValueError, match="no longer available"):
        read_layers("0" * 32, "zip")


def test_add_layers_callback(uploads):
    from module.uploadpest import add_layers, clear_layers, show_layers
    contents = [_contents(_shapefile_zip(uploads, _napa_like_layers())), _contents(b"not a zip")]
    layers, msg, cleared = add_layers(contents, ["napa.zip", "notes.txt"], None)
    assert sorted(l["name"] for l in layers) == ["domain", "stream"]
    assert "notes.txt" in msg[0].children and cleared is None
    # dropping the same file again does not duplicate its layers
    again, _, _ = add_layers(contents[:1], ["napa.zip"], layers)
    assert len(again) == 2
    assert "domain (1 polygon)" in show_layers(again).children
    assert clear_layers(1) == ([], None)
