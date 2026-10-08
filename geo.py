"""
Map layers (zipped shapefiles or GeoJSON uploaded on the Upload page) and spatial statistics
for the residual map.
"""
import functools
import json
import os
import re
import zipfile
import numpy as np
from utils import upload_path

SIMPLIFY_DEG = 0.00005   # simplification tolerance in degrees (about 5 m), keeps layers light in the browser
LAYER_EXTENSIONS = (".shp", ".geojson", ".json")

# Map layer categories, in drawing order (first at the bottom), each with a fixed style so maps look
# the same in every project. fill: polygon fill color/opacity; line: outline or line color/width/dash;
# circle: point color/radius. "words" guess the category from the layer name.
LAYER_CATEGORIES = {
    "Model domain": dict(fill=("#bdbdbd", 0.35), line=("#737373", 1.5, None), circle=("#737373", 3),
                         words=["domain", "model", "active", "grid", "ibound"]),
    "Zone":         dict(fill=None, line=("#e31a1c", 1, None), circle=("#e31a1c", 3),
                         words=["zone", "zones", "geology", "geologic"]),
    "Lake":         dict(fill=("#a6cee3", 0.7), line=("#1f78b4", 1, None), circle=("#1f78b4", 3),
                         words=["lake", "lakes", "reservoir", "pond", "waterbody", "water_body"]),
    "Boundary":     dict(fill=None, line=("#252525", 1.5, [4, 2]), circle=("#252525", 3),
                         words=["boundary", "basin", "subbasin", "county", "watershed", "huc", "outline"]),
    "Stream":       dict(fill=("#1f78b4", 0.5), line=("#1f78b4", 1.5, None), circle=("#1f78b4", 3),
                         words=["stream", "streams", "river", "rivers", "creek", "creeks", "sfr", "channel", "flowline"]),
    "Fault":        dict(fill=None, line=("#6a3d9a", 1.5, [2, 2]), circle=("#6a3d9a", 3),
                         words=["fault", "faults"]),
    "Points":       dict(fill=None, line=("#252525", 1, None), circle=("#252525", 3),
                         words=["well", "wells", "point", "points", "site", "sites", "gage", "gauge"]),
    "Other":        dict(fill=("#ff7f0e", 0.2), line=("#ff7f0e", 1.5, None), circle=("#ff7f0e", 3), words=[]),
}


def guess_category(name, kind):
    """layer category from words in its name; point layers default to Points, others to Other"""
    words = set(re.split(r"[^a-z]+", name.lower()))
    for category, style in LAYER_CATEGORIES.items():
        if words & set(style["words"]):
            return category
    return "Points" if kind == "circle" else "Other"


def category_layers(geojson, kind, category):
    """plotly map layers drawing one map layer in its category's style"""
    style = LAYER_CATEGORIES.get(category, LAYER_CATEGORIES["Other"])
    out = []
    if kind == "circle":
        color, radius = style["circle"]
        return [dict(sourcetype="geojson", source=geojson, type="circle", color=color, circle=dict(radius=radius),
                     below="traces")]
    if kind == "fill" and style["fill"]:
        color, opacity = style["fill"]
        out.append(dict(sourcetype="geojson", source=geojson, type="fill", color=color, opacity=opacity, below="traces"))
    color, width, dash = style["line"]
    line = dict(width=width) if dash is None else dict(width=width, dash=dash)
    out.append(dict(sourcetype="geojson", source=geojson, type="line", color=color, line=line, below="traces"))
    return out


def morans_i(lat, lon, values, k=8, permutations=999, seed=0):
    """
    Moran's I of site values with row-standardized k-nearest-neighbour weights, and a pseudo
    p-value from random permutations (one-sided, in the direction of the observed I).
    Returns (I, p, expected I) or None with too few sites.
    """
    lat, lon, z = (np.asarray(a, dtype=float) for a in (lat, lon, values))
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(z)
    lat, lon, z = lat[ok], lon[ok], z[ok]
    n = len(z)
    if n < k + 2 or np.allclose(z, z[0]):
        return None
    # local planar coordinates (km) are accurate enough to find neighbours
    x = (lon - lon.mean()) * np.cos(np.radians(lat.mean())) * 111.32
    y = (lat - lat.mean()) * 110.57
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(d, np.inf)
    neighbours = np.argsort(d, axis=1)[:, :k]
    W = np.zeros((n, n))
    W[np.repeat(np.arange(n), k), neighbours.ravel()] = 1 / k

    def stat(Z):   # rows of Z are (permuted) centred values
        return np.einsum("ij,ij->i", Z, Z @ W.T) / np.einsum("ij,ij->i", Z, Z)

    z = z - z.mean()
    observed = stat(z[None, :])[0]
    rng = np.random.default_rng(seed)
    perm = stat(np.array([rng.permutation(z) for _ in range(permutations)]))
    expected = -1 / (n - 1)
    extreme = perm >= observed if observed >= expected else perm <= observed
    return observed, (extreme.sum() + 1) / (permutations + 1), expected


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
