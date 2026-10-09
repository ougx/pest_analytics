from dash import html, dcc, Input, Output, State, callback
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import parse_zip
from pst import get_identifiability, get_jco_rei
from fosm import linear_uncertainty, measurement_rows, prior_information_rows, noise_scale
from plotting import message_figure
from layout import update_weights, update_jco_iter, about, graph

name = __name__.removeprefix("module.") + "-"
fullname = "Parameter Uncertainty"

EIGTHRESH = 5e-7   # identifiability shown alongside, at PEST's default cutoff

md = dcc.Markdown('''

    How much calibrating against the observations reduces each parameter's uncertainty, by linear
    (first-order, second-moment) analysis of a Jacobian matrix, as in PEST's PREDUNC utilities or pyEMU's Schur complement.

    The prior covariance $C_{prior}$ is diagonal: each parameter's bounds span the selected number of standard deviations
    (in log10 units for log-transformed parameters). Measurement noise comes from the weights: a weight $w$ means a noise
    standard deviation of $1/w$. The posterior covariance is

    $$
                   C_{post} = (J^TQJ + C_{prior}^{-1})^{-1}
    $$

    and the plot shows the uncertainty reduction $1 - \\sigma_{post}/\\sigma_{prior}$: 0 means the data say nothing
    about the parameter, 100% that they fix it completely.

    - **Noise "scaled to the fit"** (the default) multiplies the noise variance by $\\phi/n$ of the weighted measurements (at the
      residuals where the Jacobian was computed), so the noise is consistent with the misfit actually achieved.
      With weights that are much larger than 1/noise, the plain weights overstate the reduction.
    - **Regularization**: prior-information equations (and regul* observation groups) add their own constraint on
      top of the prior from the bounds; leave them out to see what the measurements alone achieve.
    - **No sensitivity**: an all-zero Jacobian column (for example a derivative run that failed and was forgiven)
      gives no reduction. Such parameters are marked separately; use a Jacobian where they were computed.
    - Identifiability (diamonds; see the Parameter Identifiability page) is shown for comparison at EIGTHRESH = 5e-7.
      Identifiability ignores the prior and the noise; the reduction accounts for both.

    **Reference:**

    Doherty, J., Hunt, R.J., 2009. Two statistics for evaluating parameter identifiability and error reduction. Journal of Hydrology 366, 119–127. https://doi.org/10.1016/j.jhydrol.2008.12.018

    ''',
    mathjax=True)

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Jacobian Iteration:", description=".jco.N, at iteration N-1 parameters", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        dmc.NumberInput(id=name+"nstd", label="Bounds span (σ):", description="prior standard deviations", value=4, min=1, max=10, step=0.5, style={"width": "150px"}),
        dmc.Select(id=name+"noise", label="Noise:", description="measurement noise std", data=["Scaled to the fit (φ/n)", "From weights (1/w)"], value="Scaled to the fit (φ/n)", style={"width": "200px"}),
        dmc.Select(id=name+"show", label="Show:", data=["Parameters", "Parameter Groups"], value="Parameters", style={"width": "170px"}),
        dmc.Switch(id=name+"regul", label="Include regularization", checked=False, style={"width": "200px"}),
        dmc.Button("Plot Uncertainty", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    graph(name+"plot", message_figure("Load data, then click 'Plot Uncertainty'."), fullname, style={"margin-bottom": "90px"}),
])


@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Input( "pest-data", "data"),
)
def fill_iterations(data):
    return update_jco_iter(data)


@callback(
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Input( "pest-data", "data"),
)
def upload_weight_columns(data):
    return update_weights(data, "weight")


def std_text(std, partrans):
    """a standard deviation for hover text; log-transformed parameters also as a factor"""
    return f"{std:.3g} log10 (×{10**std:.3g})" if partrans == "log" else f"{std:.3g}"


@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"weight", "value"),
    State( name+"nstd", "value"),
    State( name+"noise", "value"),
    State( name+"show", "value"),
    State( name+"regul", "checked"),
)
def plot(clicks, data, iiter, weight="weight", nstd=4, noise="Scaled to the fit (φ/n)", show="Parameters", regul=False):
    if clicks is None or clicks == 0:
        return message_figure("Load data, then click 'Plot Uncertainty'.")
    if data is None or data == {}:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Uncertainty'.")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, True)
    if obs is None:
        return message_figure("No .pst file found in zip file.")
    if not jcos:
        return message_figure("No .jco file found in zip file.")
    k = int(iiter)
    jco = jcos[k]
    rei = get_jco_rei(reis, k)
    scaled = noise.startswith("Scaled")
    if rei is None and (scaled or regul):
        return message_figure(f"No .rei file for .jco.{k}: needed for noise scaled to the fit and for regularization.")

    measured = measurement_rows(obs, weight)
    nmeas = int(measured.index.isin(jco.index).sum())
    rows = measurement_rows(obs, weight, include_regul=regul)
    if regul:
        prior = prior_information_rows(obs, rei)
        rows = pd.concat([rows, prior[~prior.index.isin(rows.index)]])
    nregul = int(rows.index.isin(jco.index).sum()) - nmeas
    scale = noise_scale(obs, rei, weight) if scaled else 1.0
    df = linear_uncertainty(jco, par, rows, float(nstd or 4), scale, measured=measured.index)
    ident = get_identifiability(obs, jco, EIGTHRESH, weight_col=weight)[0]
    df["identifiability"] = ident.reindex(df.index)

    nzero = int((~df.sensitive).sum())
    sens = df[df.sensitive]
    title = (f".jco.{k}: {nmeas} weighted measurements"
             + (f" + {nregul} regularization rows" if regul and nregul else "")
             + (" (no regularization rows: the residuals for this Jacobian hold none)" if regul and not nregul else "")
             + f"; noise {'× √(φ/n) = ×' + format(scale**0.5, '.3g') if scaled else 'from weights'}"
             + f"; median reduction {100 * sens.reduction.median():.0f}%"
              + (f"<br>{nzero} of {len(df)} parameters have no measurement sensitivity in this Jacobian" if nzero else ""))
    if show == "Parameter Groups":
        return group_figure(df, title)
    return parameter_figure(df, title)


def parameter_figure(df, title):
    df = df.sort_values(["sensitive", "reduction"], ascending=False)
    groups = list(dict.fromkeys(df.pargrp))
    colors = dict(zip(groups, (px.colors.qualitative.Plotly * (len(groups) // 10 + 1))))
    fig = go.Figure()
    for g in groups:
        d = df[(df.pargrp == g) & df.sensitive]
        if len(d) == 0:
            continue
        fig.add_trace(go.Bar(
            x=100 * d.reduction, y=d.index, orientation="h", name=g, marker_color=colors[g],
            customdata=[[g, std_text(r.prior_std, r.partrans), std_text(r.post_std, r.partrans)] for r in d.itertuples()],
            hovertemplate="%{y} (%{customdata[0]})<br>reduction: %{x:.1f}%<br>prior σ: %{customdata[1]}"
                          "<br>posterior σ: %{customdata[2]}<extra></extra>",
        ))
    d = df[~df.sensitive]
    if len(d):
        fig.add_trace(go.Scatter(
            x=100 * d.reduction, y=d.index, mode="markers", name="No measurement sensitivity in this Jacobian",
            marker=dict(symbol="x", color="grey", size=8), customdata=d.pargrp,
            hovertemplate="%{y} (%{customdata}): all-zero Jacobian column for the measurements"
                          "<br>reduction: %{x:.1f}%<extra></extra>",
        ))
    d = df[df.identifiability.notna()]
    fig.add_trace(go.Scatter(
        x=100 * d.identifiability, y=d.index, mode="markers", name=f"Identifiability (EIGTHRESH {EIGTHRESH:g}), %",
        marker=dict(symbol="diamond", color="black", size=7),
        hovertemplate="%{y}<br>identifiability: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, yref="container", y=1, yanchor="top", pad=dict(t=15)),
        height=18 * len(df) + 240, barmode="overlay",
        xaxis=dict(title="Uncertainty reduction, 1 - σ_post/σ_prior (%)", range=[-3, 103], side="top"),
        yaxis=dict(title="Parameter", categoryorder="array", categoryarray=list(df.index), range=[len(df) - 0.5, -0.5]),
        legend=dict(orientation="h", y=-30 / (18 * len(df) + 220), yanchor="top"),
        margin=dict(t=170),
    )
    return fig


def group_figure(df, title):
    """mean, min and max reduction per group, over the parameters with measurement sensitivity"""
    counts = df.groupby("pargrp", sort=False).agg(n=("reduction", "size"),
                                                  nzero=("sensitive", lambda s: int((~s).sum())))
    summary = df[df.sensitive].groupby("pargrp", sort=False).agg(
        mean=("reduction", "mean"), lo=("reduction", "min"), hi=("reduction", "max"),
        ident=("identifiability", "mean"))
    summary = counts.join(summary).sort_values("mean", ascending=False)
    labels = [f"{r.Index}<br>({r.n - r.nzero} of {r.n})" for r in summary.itertuples()]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=100 * summary["mean"], name="Mean reduction (%), bar: min to max",
        error_y=dict(type="data", symmetric=False, array=100 * (summary.hi - summary["mean"]),
                     arrayminus=100 * (summary["mean"] - summary.lo), color="grey"),
        customdata=summary[["n", "nzero"]].values,
        hovertemplate="mean %{y:.1f}% (bar: min to max)<br>%{customdata[0]} parameters; "
                      "%{customdata[1]} without measurement sensitivity are left out<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=100 * summary.ident, mode="markers", name=f"Mean identifiability (EIGTHRESH {EIGTHRESH:g}), %",
        marker=dict(symbol="diamond", color="black", size=9),
        hovertemplate="mean identifiability %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=600,
        xaxis=dict(title="Parameter group (parameters with measurement sensitivity)"),
        yaxis=dict(title="Uncertainty reduction, 1 - σ_post/σ_prior (%)", range=[0, 102]),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=110),
    )
    return fig
