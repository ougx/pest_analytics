from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from utils import parse_zip, read_sites
from plotting import message_figure
from layout import about, graph
from module.resmap import BASEMAPS, basemap_layers, shape_layers
import compare

name = __name__.removeprefix("module.") + "-"
fullname = "Fit Changes"

BETTER, WORSE = "#2166ac", "#b2182b"

md = dcc.Markdown('''

    How the fit changed between the loaded run (the reference) and another run added under "Runs to compare" on the
    Upload Data page, observation by observation (matched by name; observations in only one run and regularization are left out).

    **Use the same weights for both runs.** When the weights changed between runs, each run's own $\\phi$ is measured on a
    different scale: lowering many weights lowers $\\phi$ without any change in the fit. By default both runs are scored with
    the reference run's weights; the other choices are the other run's weights, each run's own weights (what PEST reported)
    or the original weights from obs.csv (`weight_orig`).

    - **Objective function by group**: $\\sum (w r)^2$ per group for both runs (the largest groups).
    - **RMSE by group**: RMSE in measured units of the observations weighted in both runs, one point per group; points
      below the 1:1 line fit better in the second run. Group by a site column (e.g. `grpSite`) to compare sites.
    - **Map of the change in RMSE**: with a site column and site.csv in the reference zip, red sites fit worse in the second
      run and blue ones better.

    Each run is represented by its best iteration (lowest measurement $\\phi$ with its own weights) unless you choose another.

    ''', mathjax=True)

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"run", label="Other run:", description="add runs on the Upload Data page",
                   data=[], placeholder="no runs added", style={"width": "220px"}),
        dmc.Select(id=name+"direction", label="Changes from:", data=[compare.FORWARD, compare.BACKWARD],
                   value=compare.FORWARD, style={"width": "220px"}),
        dmc.Select(id=name+"ref-iter", label="Reference iteration:", data=[compare.BEST, compare.LAST],
                   value=compare.BEST, style={"width": "170px"}),
        dmc.Select(id=name+"other-iter", label="Other run iteration:", data=[compare.BEST, compare.LAST],
                   value=compare.BEST, style={"width": "170px"}),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"weights", label="Score both runs with:", data=compare.WEIGHTS, value=compare.WEIGHTS[0],
                   style={"width": "230px"}),
        dmc.Select(id=name+"grp", label="Grouped by:", data=["obsgrp"], value="obsgrp", style={"width": "150px"}),
        dmc.Select(id=name+"basemap", label="Basemap:", data=list(BASEMAPS), value="Light gray (Esri)",
                   style={"width": "180px"}),
        dmc.Button("Compare Fit", id=name+"update", color="blue"),
    ], mt=10, align="flex-end"),
    html.Div(id=name+"summary", style={"margin-top": "20px"}),
    graph(name+"phi", message_figure("Add a run to compare on the Upload Data page, then click 'Compare Fit'."),
          "Objective Function by Group"),
    graph(name+"rmse", message_figure(), "RMSE by Group"),
    graph(name+"map", message_figure(), "Change in RMSE by Site", style={"margin-bottom": "90px"}),
])


@callback(
    Output(name+"run", "data"),
    Output(name+"run", "value"),
    Input("compare-runs", "data"),
    State(name+"run", "value"),
)
def fill_runs(runs, current):
    options = [{"value": str(i), "label": r["name"]} for i, r in enumerate(runs or [])]
    values = [o["value"] for o in options]
    return options, current if current in values else (values[-1] if values else None)


@callback(
    Output(name+"grp", "data"),
    Output(name+"grp", "value"),
    Input("pest-data", "data"),
)
def fill_groups(data):
    if not data:
        return no_update, no_update
    obs = parse_zip(data, False)[2]
    if obs is None:
        return no_update, no_update
    cols = ["obsgrp"] + [c for c in obs.columns if c.lower().startswith("grp")]
    return cols, "grpSite" if "grpSite" in cols else "obsgrp"


def iteration_options(data):
    reis = parse_zip(data, False, True)[4] if data else None
    return [compare.BEST, compare.LAST] + [str(i) for i in sorted(reis or {})]


@callback(
    Output(name+"ref-iter", "data"),
    Output(name+"other-iter", "data"),
    Input("pest-data", "data"),
    Input("compare-runs", "data"),
    Input(name+"run", "value"),
)
def fill_iterations(data, runs, run):
    other = runs[int(run)] if runs and run is not None and int(run) < len(runs) else None
    return iteration_options(data), iteration_options(other)


def phi_figure(fit, label_from, label_to, top=25):
    f = fit.assign(big=fit[["phi_from", "phi_to"]].max(axis=1)).sort_values("big", ascending=False)
    shown = f.head(top)
    fig = go.Figure()
    for col, label, color in (("phi_from", label_from, "darkgray"), ("phi_to", label_to, BETTER)):
        fig.add_trace(go.Bar(
            x=shown.index, y=shown[col], name=label, marker_color=color,
            customdata=100 * shown[col] / f[col].sum(),
            hovertemplate="%{x}<br>φ = %{y:.4g} (%{customdata:.1f}% of the total)<extra>%{fullData.name}</extra>",
        ))
    total_from, total_to = f.phi_from.sum(), f.phi_to.sum()
    change = f"{100 * (total_to / total_from - 1):+.1f}%" if total_from > 0 else ""
    fig.update_layout(
        barmode="group", height=480,
        title=f"Objective function by group: {total_from:.4g} → {total_to:.4g} ({change})"
              + (f"; the {top} largest of {len(f)} groups" if len(f) > top else ""),
        xaxis=dict(title="Group", tickangle=-45 if len(shown) > 8 else 0), yaxis_title="φ = Σ (w r)²",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    return fig


def rmse_figure(fit, label_from, label_to):
    f = fit[fit.n > 0]
    if not len(f):
        return message_figure("No observation is weighted in both runs.")
    fig = go.Figure()
    lo, hi = f[["rmse_from", "rmse_to"]].min().min(), f[["rmse_from", "rmse_to"]].max().max()
    lo = lo if lo > 0 else hi / 1e3
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash"),
                             name="1:1", hoverinfo="skip"))
    hover = ("<b>%{text}</b> (%{customdata[0]} obs.)<br>RMSE: %{x:.4g} → %{y:.4g}"
             "<br>mean residual: %{customdata[1]:.4g} → %{customdata[2]:.4g}"
             "<br>φ: %{customdata[3]:.4g} → %{customdata[4]:.4g}<extra></extra>")
    for mask, label, color in ((f.drmse <= 0, f"Better or same in {label_to}", BETTER),
                               (f.drmse > 0, f"Worse in {label_to}", WORSE)):
        d = f[mask]
        fig.add_trace(go.Scatter(
            x=d.rmse_from, y=d.rmse_to, mode="markers", name=f"{label} ({len(d)})", text=d.index,
            marker=dict(color=color, size=9, line=dict(width=0.5, color="white")),
            customdata=d[["n", "mean_from", "mean_to", "phi_from", "phi_to"]].values, hovertemplate=hover,
        ))
    fig.update_layout(
        height=560, title="RMSE by group (measured units, observations weighted in both runs)",
        xaxis=dict(title=f"RMSE, {label_from}", type="log"), yaxis=dict(title=f"RMSE, {label_to}", type="log"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    return fig


def map_figure(fit, sites, grp, basemap, layers, label_from, label_to):
    if sites is None:
        return message_figure("No site.csv in the reference zip: no map.")
    located = fit[fit.n > 0].join(sites.drop_duplicates("site").set_index("site")[["lat", "lon"]], how="inner")
    if not len(located):
        return message_figure(f"No {grp} value is a site in site.csv: group by a site column (e.g. grpSite) for the map.")
    lim = np.nanquantile(located.drmse.abs(), 0.95) or 1
    fig = go.Figure(go.Scattermap(
        lat=located.lat, lon=located.lon, mode="markers", text=located.index,
        marker=dict(size=12, color=located.drmse, colorscale="RdBu_r", cmin=-lim, cmid=0, cmax=lim, opacity=0.9,
                    showscale=True, colorbar=dict(title=dict(text="Change in RMSE", side="right"))),
        customdata=located[["n", "rmse_from", "rmse_to"]].values,
        hovertemplate="<b>%{text}</b> (%{customdata[0]} obs.)<br>RMSE: %{customdata[1]:.4g} → %{customdata[2]:.4g}"
                      "<br>change: %{marker.color:+.4g}<extra></extra>",
    ))
    span = max(located.lon.max() - located.lon.min(), 1.3 * (located.lat.max() - located.lat.min()), 0.01)
    fig.update_layout(
        map=dict(style="white-bg", center=dict(lat=located.lat.mean(), lon=located.lon.mean()),
                 zoom=float(np.clip(np.log2(360 / span) - 0.6, 1, 15)),
                 layers=basemap_layers(basemap, 100) + shape_layers(layers or [])),
        height=650, margin=dict(l=0, r=0, t=40, b=0),
        title=f"Change in RMSE by {grp}, {label_from} → {label_to} (red: worse)",
    )
    return fig


@callback(
    Output(name+"summary", "children"),
    Output(name+"phi", "figure"),
    Output(name+"rmse", "figure"),
    Output(name+"map", "figure"),
    Input(name+"update", "n_clicks"),
    State("pest-data", "data"),
    State("compare-runs", "data"),
    State(name+"run", "value"),
    State(name+"direction", "value"),
    State(name+"ref-iter", "value"),
    State(name+"other-iter", "value"),
    State(name+"weights", "value"),
    State(name+"grp", "value"),
    State(name+"basemap", "value"),
    State("map-layers", "data"),
    prevent_initial_call=True,
    running=[(Output(name+"update", "loading"), True, False)],
)
def compare_fit(n_clicks, data, runs, run, direction=compare.FORWARD, ref_iter=compare.BEST,
                other_iter=compare.BEST, weights=compare.WEIGHTS[0], grp="obsgrp", basemap="None", layers=None):
    def only(message, color="yellow"):
        return [dmc.Alert(message, color=color, variant="light"), message_figure(message), message_figure(),
                message_figure()]
    if not n_clicks:
        return [no_update] * 4
    if not data:
        return only("No data loaded. Load the reference run on the Upload Data page.")
    if not runs or run is None or int(run) >= len(runs):
        return only("Add another run under 'Runs to compare' on the Upload Data page, then choose it here.")
    other = runs[int(run)]
    ref_name, _, obs_ref, _, reis_ref, _ = parse_zip(data, False, True)
    other_name, _, obs_other, _, reis_other, _ = parse_zip(other, False, True)
    if not reis_ref or not reis_other:
        return only("Both runs need residual files (.rei or .hp).", "red")
    i_ref, i_other = compare.pick_iteration(reis_ref, ref_iter), compare.pick_iteration(reis_other, other_iter)
    if i_ref is None or i_other is None:
        return only("The chosen iteration is not in that run.", "red")
    grp = grp if grp in obs_ref.columns else "obsgrp"
    to_is_ref = direction == compare.BACKWARD
    try:
        w_from, w_to = compare.run_weights(weights, obs_ref, obs_other, to_is_ref)
    except ValueError as e:
        return only(str(e), "red")
    side = {"ref": (reis_ref[i_ref], f"{ref_name} (iteration {i_ref})"),
            "other": (reis_other[i_other], f"{other_name} (iteration {i_other})")}
    (rei_from, label_from), (rei_to, label_to) = (side["other"], side["ref"]) if to_is_ref else (side["ref"], side["other"])
    paired = compare.paired_residuals(rei_from, rei_to, w_from, w_to, obs_ref.set_index("obsname")[grp])
    if not len(paired):
        return only("The two runs have no observation in common.", "red")
    fit = compare.fit_by_group(paired)

    phi_from, phi_to = fit.phi_from.sum(), fit.phi_to.sum()
    own = {k: compare.measurement_phi(r) for k, (r, _) in side.items()}
    text = (f"{label_from} → {label_to}, scored with {weights[0].lower() + weights[1:]}: φ {phi_from:.4g} → {phi_to:.4g} "
            f"({100 * (phi_to / phi_from - 1):+.1f}%) over the "
            f"{int(((paired.weight_from > 0) | (paired.weight_to > 0)).sum())} weighted observations in both runs. "
            f"With their own weights PEST reported {own['other' if to_is_ref else 'ref']:.4g} → "
            f"{own['ref' if to_is_ref else 'other']:.4g}.")
    better = int((fit[fit.n > 0].drmse < 0).sum())
    summary = [dmc.Alert(text, color="blue", variant="light", mb=5),
               dmc.Text(f"RMSE better in {better} of {int((fit.n > 0).sum())} groups ({grp}).", size="sm", c="dimmed")]
    try:
        sites = read_sites(data)
    except ValueError:
        sites = None
    return [summary, phi_figure(fit, label_from, label_to), rmse_figure(fit, label_from, label_to),
            map_figure(fit, sites, grp, basemap, layers, label_from, label_to)]
