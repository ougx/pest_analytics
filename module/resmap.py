from dash import html, dcc, Input, Output, State, ALL, callback, no_update
import dash_mantine_components as dmc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from utils import parse_zip, read_sites
from plotting import message_figure
from layout import update_groups, update_weights, about, graph
from geo import read_layers, morans_i, LAYER_CATEGORIES, guess_category, category_layers

name = __name__.removeprefix("module.") + "-"
fullname = "Residual Map"

md = dcc.Markdown('''

    This page maps the residuals (measured - modelled) of each site to show spatial bias.
    Sites are located with `site.csv` in the zip file (columns `site`, `lat`, `lon` in decimal degrees, plus any
    columns to show on hover); `site` must match the values of the selected site column of `obs.csv` (e.g. `grpSite`).

    Each marker is one site, summarizing its weighted observations (weight > 0) in the selected groups and years:

    * **Mean residual** (bias): red = under-predicted (measured > modelled), blue = over-predicted.
    * **Weighted mean residual**: mean of weight × residual, i.e. the bias as PEST sees it.
    * **RMSE** and **mean absolute residual**: size of the misfit regardless of sign.
    * **Change in RMSE since iteration 0**: negative (blue) where the calibration improved the fit.

    Hovering a site also shows its share of the total measurement objective function: the site's
    $\\sum (w_i r_i)^2$ divided by the $\\phi$ of all weighted observations of the iteration (without regularization,
    and regardless of the group/year filters). With every group and year selected, the shares of all sites add up to
    100% (less any weighted observations that have no site).

    Above the map, Moran's I measures whether neighbouring sites have similar values of the mapped statistic
    (8 nearest neighbours; p-value from 999 random permutations). A significant positive I for the mean residual means
    the bias is spatially clustered, which usually points to a zone, boundary or structural problem rather than noise.

    Click a site to open its time series. Map layers (shapefiles or GeoJSON) are added on the Upload Data page.
    Basemap tiles are loaded by the browser from the selected provider (an internet connection is needed);
    choose "None" to work offline.
    ''',
    mathjax=True)

# basemap name -> (tile URL, attribution); drawn as a raster layer so its transparency can be set
BASEMAPS = {
    "OpenStreetMap": ("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap contributors"),
    "Light gray (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                          "Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "Topographic (USGS)": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
                           "USGS The National Map"),
    "Satellite (USGS)": ("https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
                         "USGS The National Map: Orthoimagery"),
    "Satellite (Esri)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         "Esri, Maxar, Earthstar Geographics"),
    "None": None,
}

STATS = {
    "mean": "Mean residual",
    "wmean": "Weighted mean residual",
    "rmse": "RMSE",
    "mae": "Mean absolute residual",
    "drmse": "Change in RMSE since iteration 0",
}
SIGNED = {"mean", "wmean", "drmse"}


def in_sentence(label):
    """a label as used inside a sentence: lower case unless it starts with an acronym (RMSE)"""
    first = label.split()[0]
    return label if first.isupper() else label[0].lower() + label[1:]


layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Residuals of Iteration:", data=["0"], value="0", style={"width": "150px"}),
        dmc.Select(id=name+"site", label="Site Column:", data=["grpSite"], value="grpSite", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        dmc.MultiSelect(id=name+"groups", label="Observation Groups:", data=[], value=[], style={"width": "300px"}),
        dmc.Stack([
            dmc.Text("Years:", size="sm", fw=500),
            dmc.RangeSlider(id=name+"years", min=1900, max=2100, value=[1900, 2100], step=1, minRange=0,
                            disabled=True, style={"width": "260px"}),
        ], gap=4),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"stat", label="Color by:", data=[{"value": k, "label": v} for k, v in STATS.items()],
                   value="mean", style={"width": "260px"}),
        dmc.Select(id=name+"size", label="Size by:", data=["Observations", "RMSE", "Fixed"], value="Observations",
                   style={"width": "150px"}),
        dmc.Button("Plot Map", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"basemap", label="Basemap:", data=list(BASEMAPS), value="OpenStreetMap", style={"width": "200px"}),
        dmc.Stack([
            dmc.Text("Basemap opacity:", size="sm", fw=500),
            dmc.Slider(id=name+"opacity", min=0, max=100, step=5, value=70, style={"width": "200px"}),
        ], gap=4),
        html.Div(id=name+"layer-switches"),
    ], mt=10, align="flex-end"),
    dmc.Text(id=name+"note", size="sm", c="dimmed", mt=10),
    graph(name+"plot", message_figure("Load data with site.csv, then click 'Plot Map'."), fullname,
          style={"margin-bottom": "90px"}),
])


@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Output(name+"site", "data"),
    Output(name+"site", "value"),
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Output(name+"groups", "data"),
    Output(name+"groups", "value"),
    Output(name+"years", "min"),
    Output(name+"years", "max"),
    Output(name+"years", "value"),
    Output(name+"years", "disabled"),
    Input("pest-data", "data"),
)
def fill_options(data):
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, False)
    if not pst or not reis:
        return [no_update] * 12
    iters = [str(i) for i in sorted(reis)]
    sites, site = update_groups(data, "grpSite")
    weights, weight = update_weights(data, "weight")
    groups = sorted(g for g in obs.obsgrp.dropna().unique() if not g.startswith("regul"))
    years = obs_years(obs)
    if years is None:
        year_props = [no_update, no_update, no_update, True]
    else:
        year_props = [int(years.min()), int(years.max()), [int(years.min()), int(years.max())], False]
    return [iters, iters[-1], sites, site, weights, weight, groups, groups] + year_props


@callback(
    Output(name+"layer-switches", "children"),
    Input("map-layers", "data"),
)
def show_layer_switches(layers):
    if not layers:
        return dmc.Text("No map layers (add them on the Upload Data page).", size="sm", c="dimmed")
    return dmc.Group([dmc.Text("Layers:", size="sm", fw=500)] + [
        dmc.Switch(id={"type": name+"layer", "index": i}, label=l["name"], checked=True, size="sm")
        for i, l in enumerate(layers)
    ], gap="md")


def obs_years(obs):
    """years of the observations (from the obs.csv time column), or None"""
    if "time" not in obs.columns:
        return None
    years = pd.to_datetime(obs["time"], errors="coerce").dt.year
    return None if years.isna().all() else years


def site_stats(rei, site_col, weight, groups=None, years=None):
    """residual statistics per site of the weighted observations in the given groups and years"""
    df = rei[rei[weight] > 0]
    if groups:
        df = df[df.obsgrp.isin(groups)]
    if years and "time" in df.columns:
        y = pd.to_datetime(df["time"], errors="coerce").dt.year
        df = df[y.between(*years)]
    df = df[df[site_col].notna()]
    r, w = df.residual, df[weight]
    g = pd.DataFrame(dict(r=r, wr=w * r, r2=r**2, ar=r.abs(), phi=(w * r)**2, site=df[site_col].astype(str).str.lower()))
    stats = g.groupby("site").agg(n=("r", "size"), mean=("r", "mean"), wmean=("wr", "mean"),
                                  r2=("r2", "mean"), mae=("ar", "mean"), phi=("phi", "sum"))
    stats["rmse"] = np.sqrt(stats.pop("r2"))
    total = total_phi(rei, weight)
    stats["phi_share"] = stats.phi / total if total > 0 else np.nan
    return stats


def total_phi(rei, weight):
    """measurement objective function of the iteration: sum of (w r)^2 without regularization"""
    df = rei[(rei[weight] > 0) & ~rei.obsgrp.astype(str).str.startswith("regul")]
    return float(((df[weight] * df.residual)**2).sum())


@callback(
    Output(name+"plot", "figure"),
    Output(name+"note", "children"),
    Input(name+"update-plot", "n_clicks"),
    Input(name+"basemap", "value"),
    Input(name+"opacity", "value"),
    Input({"type": name+"layer", "index": ALL}, "checked"),
    State("pest-data", "data"),
    State("map-layers", "data"),
    State(name+"iter", "value"),
    State(name+"site", "value"),
    State(name+"weight", "value"),
    State(name+"groups", "value"),
    State(name+"years", "value"),
    State(name+"years", "disabled"),
    State(name+"stat", "value"),
    State(name+"size", "value"),
)
def plot(clicks, basemap, opacity, layer_on, data, layers, iiter, site_col, weight, groups, years, no_years,
         stat, size_by):
    if not clicks:
        return message_figure("Load data with site.csv, then click 'Plot Map'."), None
    if not data:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Map'."), None
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, False)
    if obs is None or not reis:
        return message_figure("No .pst or .rei files found in the zip file."), None
    try:
        sites = read_sites(data)
    except ValueError as e:
        return message_figure(f"site.csv: {e}"), None
    if sites is None:
        return message_figure("Add site.csv (site, lat, lon) to the zip file to map the residuals."), None
    if site_col not in obs.columns:
        return message_figure(f"Column {site_col} not found."), None

    n = int(iiter)
    years = None if no_years else years
    stats = site_stats(reis[n], site_col, weight, groups, years)
    if stat == "drmse":
        base = site_stats(reis[min(reis)], site_col, weight, groups, years)
        stats["drmse"] = stats.rmse - base.rmse.reindex(stats.index)
    located = stats.join(sites.set_index("site"), how="inner")
    unlocated = len(stats) - len(located)
    note = (f"{len(located)} sites mapped, together {located.phi_share.sum():.1%} of the total measurement φ "
            f"of iteration {iiter}" + (f"; {unlocated} {site_col} values have no coordinates in site.csv" if unlocated else ""))
    if located.empty:
        return message_figure(f"None of the {site_col} values match the site column of site.csv."), note

    on = [l for l, checked in zip(layers or [], layer_on) if checked]
    return map_figure(located, stat, size_by, basemap, opacity, on, iiter, site_col), [note, html.Br(), moran_note(located, stat)]


def moran_note(located, stat):
    """one-line spatial autocorrelation summary of the mapped statistic"""
    result = morans_i(located.lat, located.lon, located[stat])
    if result is None:
        return "Moran's I: too few sites."
    i, p, expected = result
    if p < 0.05 and i > expected:
        verdict = "neighbouring sites have similar values: the misfit is spatially clustered"
    elif p < 0.05:
        verdict = "neighbouring sites tend to differ (spatially dispersed)"
    else:
        verdict = "no significant spatial pattern"
    return f"Moran's I of {in_sentence(STATS[stat])} = {i:.2f} (p = {p:.3f}; 8 nearest neighbours): {verdict}."


def map_figure(located, stat, size_by, basemap, opacity, layers, iiter, site_col):
    values = located[stat]
    if stat in SIGNED:
        lim = np.nanquantile(np.abs(values), 0.95) or 1
        color = dict(colorscale="RdBu_r", cmin=-lim, cmid=0, cmax=lim)
    else:
        color = dict(colorscale="Reds", cmin=0, cmax=np.nanquantile(values, 0.95) or 1)
    if size_by == "Fixed":
        size = 12
    else:
        s = np.sqrt((located.n if size_by == "Observations" else located.rmse).astype(float))
        size = 7 + 18 * (s - s.min()) / ((s.max() - s.min()) or 1)

    located = located.copy()
    located["phi_text"] = [f"{s:.2%} (φ = {p:.4g})" for s, p in zip(located.phi_share, located.phi)]
    extra = [c for c in located.columns
             if c not in ("n", "mean", "wmean", "mae", "rmse", "drmse", "lat", "lon", "phi", "phi_share", "phi_text")
             and located[c].notna().any()]
    shown = ["n", "phi_text", "mean", "wmean", "rmse", "mae"] + (["drmse"] if "drmse" in located else []) + extra
    labels = {"n": "observations", "phi_text": "share of total φ", **{k: in_sentence(v) for k, v in STATS.items()}}
    hover = "<b>%{customdata[0]}</b><br>" + "<br>".join(
        f"{labels.get(c, c)}: %{{customdata[{i + 1}]}}" for i, c in enumerate(shown)) + "<extra></extra>"
    customdata = np.column_stack([located.index.astype(str)] + [
        located[c].round(3).astype(str) if located[c].dtype.kind == "f" else located[c].astype(str) for c in shown])

    fig = go.Figure(go.Scattermap(
        lat=located.lat, lon=located.lon, mode="markers",
        marker=dict(size=size, color=values, showscale=True, opacity=0.9,
                    colorbar=dict(title=dict(text=STATS[stat], side="right")), **color),
        customdata=customdata, hovertemplate=hover, name=STATS[stat],
    ))
    lat_span = located.lat.max() - located.lat.min()
    lon_span = located.lon.max() - located.lon.min()
    zoom = float(np.clip(np.log2(360 / max(lon_span, lat_span * 1.3, 0.01)) - 0.6, 1, 15))
    fig.update_layout(
        map=dict(style="white-bg", center=dict(lat=located.lat.mean(), lon=located.lon.mean()), zoom=zoom,
                 layers=basemap_layers(basemap, opacity) + shape_layers(layers)),
        height=750, margin=dict(l=0, r=0, t=40, b=0),
        title=f"{STATS[stat]} by {site_col}, residuals of iteration {iiter}",
        uirevision=site_col,   # keep the user's pan/zoom when toggling layers or the basemap
    )
    return fig


def basemap_layers(basemap, opacity):
    tiles = BASEMAPS.get(basemap)
    if not tiles:
        return []
    url, attribution = tiles
    return [dict(sourcetype="raster", source=[url], sourceattribution=attribution, below="traces",
                 opacity=(opacity if opacity is not None else 100) / 100)]


def shape_layers(layers):
    """map layers for the switched-on uploaded layers, styled and ordered by their category"""
    order = list(LAYER_CATEGORIES)
    def rank(l):
        category = l.get("category") or guess_category(l["name"], l["kind"])
        return order.index(category) if category in order else len(order)
    out = []
    for l in sorted(layers, key=rank):
        try:
            geojson = read_layers(l["key"], l["ext"])[l["index"]]["geojson"]
        except (ValueError, IndexError, KeyError):
            continue   # upload no longer on the server
        out += category_layers(geojson, l["kind"], l.get("category") or guess_category(l["name"], l["kind"]))
    return out


@callback(
    Output("main-tabs", "value", allow_duplicate=True),
    Output("tsplot-grp", "value", allow_duplicate=True),
    Output("tsplot-site", "value", allow_duplicate=True),
    Output("tsplot-update-plot", "n_clicks"),
    Input(name+"plot", "clickData"),
    State(name+"site", "value"),
    State("tsplot-update-plot", "n_clicks"),
    prevent_initial_call=True,
)
def open_time_series(click, site_col, n_clicks):
    """clicking a site opens its time series"""
    if not click:
        return no_update, no_update, no_update, no_update
    site = click["points"][0]["customdata"][0]
    return "tsplot", site_col, site, (n_clicks or 0) + 1
