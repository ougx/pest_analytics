"""
Map layers (zipped shapefiles or GeoJSON uploaded on the Upload page) and spatial statistics
for the residual map.
"""
import functools
import json
import os
import zipfile
import numpy as np
from utils import upload_path

SIMPLIFY_DEG = 0.00005   # simplification tolerance in degrees (about 5 m), keeps layers light in the browser
LAYER_EXTENSIONS = (".shp", ".geojson", ".json")


def layer_kind(geom_types):
    """how a layer is drawn: polygons as a light fill with an outline, lines as lines, points as circles"""
    if all("Point" in g for g in geom_types):
        return "circle"
    if all("Polygon" in g for g in geom_types):
        return "fill"
    return "line"


def read_layers(key, ext):
    """
    The layers in an uploaded file as [{"name", "kind", "count", "geojson"}] in latitude/longitude.
    A zip may hold several shapefiles (or GeoJSON files). Raises ValueError for unusable files.
    """
    path = upload_path(key, ext)
    if path is None:
        raise ValueError("the uploaded file is no longer available on the server; upload it again")
    return _read_layers_cached(path)


@functools.lru_cache(maxsize=32)
def _read_layers_cached(path):
    import geopandas as gpd   # imported here: slow to import and only needed for map layers

    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(LAYER_EXTENSIONS)]
        if not names:
            raise ValueError("no shapefile (.shp) or GeoJSON file found in the zip")
        sources = [(os.path.splitext(os.path.basename(n))[0], f"zip://{path}!{n}") for n in names]
    else:
        sources = [(None, path)]

    layers = []
    for name, source in sources:
        gdf = gpd.read_file(source)
        name = name or "layer"
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if gdf.empty:
            continue
        gdf = gdf.set_geometry(gdf.geometry.force_2d())
        if gdf.crs is None:
            minx, miny, maxx, maxy = gdf.total_bounds
            if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
                raise ValueError(f"{name} has no coordinate system (.prj file) and is not in latitude/longitude")
            gdf = gdf.set_crs(4326)
        gdf = gdf.to_crs(4326)
        gdf = gdf.set_geometry(gdf.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True))
        layers.append(dict(
            name=name,
            kind=layer_kind(gdf.geom_type.unique()),
            count=len(gdf),
            geojson=json.loads(gdf[["geometry"]].to_json()),
        ))
    if not layers:
        raise ValueError("the file has no features")
    return layers
