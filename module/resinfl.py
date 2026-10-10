import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
from utils import parse_zip
from plotting import colorscales, message_figure, normal_scores, qq_line
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pst import get_weighted_jco, get_jco_rei
from layout import update_groups, update_weights, update_jco_iter, about, graph

name = __name__.removeprefix("module.") + "-"
fullname = "Residual Analysis"

md = dcc.Markdown('''

    This page investigates model misfit (residuals). Residuals are plotted against observation sensitivity and leverage.
    Observation leverage is the POTENTIAL of an observation affecting the inverse result.
    It can be calculated as the diagonal elements of the hat matrix $H$:
    $$
                  H = (Q^{1/2}J)(J^TQJ)^{-1}(J^TQ^{1/2})
    $$
    where $Q$ is the weight matrix; $J$ is the jacobian matrix. Note that $(J^TQJ)^{-1}$ is the parameter covariance matrix.


    The figure below plots observation leverage (X axis) vs. absolute standardized residual (Y axis), sized by the observation sensitivity and colored by raw residual.
    `.jco.N` is computed at the parameters of iteration N-1, so it is paired with the residuals in `.rei.(N-1)` (`.hp` for N = 1).
    Standardized Residual is calculated as follows:
    $$
        r^*_j = \\frac{r_j}{\\sqrt{\\sigma^2(1 - H_{jj})}}
    $$

    In this figure, observations can be divided into four quadrant:

    1. Top-right (HIGH leverage + HIGH residual): these are the most problematic observations that strongly influence parameters but poorly fit.
    The reasons could be

                  * bad data
                  * wrong weights
                  * model structural error

    2. Top-left (LOW leverage + HIGH residual): these observations are poorly fit but with low impact.
    They are likely noise or outliers. It is safe to downweight or remove.

    3. Bottom-right (HIGH leverage + LOW residual): these observations are working and well-fit.
    They are critical observations and should NOT be removed.

    4. Bottom-left (LOW leverage + LOW residual): these are unimportant observations that could be redundant or provide low information.

    ''',
    mathjax=True)

# However if its identifiability is unity, this does not mean that its
#    estimation takes place without any potential for error. What it does indicate is that all of its
#    potential for error is inherited from noise in the measurement dataset.

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Jacobian Iteration:", description=".jco.N with .rei.(N-1) residuals", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"grp"   , label="Grouped by:" , data=["obsgrp"], value="obsgrp", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"palette", label="Color Scale:", data=colorscales, value=colorscales[0], style={"width": "150px"}),
        dmc.Switch(id=name+"invcmap", label="Inverse colormap", checked=True, style={"width": "150px"}),
        dmc.Select(id=name+"xscale", label="X Scale:", data=["Linear", "Log"], value="Log", style={"width": "150px"}),
        dmc.Select(id=name+"yscale", label="Y Scale:", data=["Linear", "Log"], value="Linear", style={"width": "150px"}),
        dmc.Button("Plot Residual", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    # dmc.MultiSelect(
    #     label="Select observation group(s) to analyze:",
    #     id=name+"obsgrp",
    #     value=[],
    #     data=["obs1", "obs2", ],
    #     mb=10,
    #     searchable=True,
    # ),
    graph(name+"plot", message_figure("Load data, then click 'Plot Residual'."), fullname + ""),
    html.H4("Weighted residual distribution", style={"margin-top": "30px"}),
    dmc.Text("Weighted residuals w(measured - modelled) of the weighted observations (regularization excluded), by the "
             "'Grouped by' column above. If the weights were 1 / noise standard deviation and the noise were independent "
             "and normal, they would follow a standard normal distribution: the Q-Q points would lie on a line of slope 1 "
             "through the origin. A steeper line means more misfit than the weights allow; bent tails, outliers or "
             "steps point to a few targets or groups. Head residuals are usually correlated in time, so read this as a "
             "check for outliers and skew rather than as a formal test.", size="sm", c="dimmed"),
    dmc.Group([
        dmc.Select(id=name+"dist-iter", label="Residual Iteration:", description=".rei.N (0: .hp)",
                   data=["0"], value="0", style={"width": "150px"}),
        dmc.Button("Plot Distribution", id=name+"dist-update", color="blue"),
    ], mt=10, align="flex-end"),
    graph(name+"dist", message_figure("Load data, then click 'Plot Distribution'."), "Weighted Residual Distribution",
          style={"margin-bottom":"90px"}),
])



# @callback(
#     Output(name+"obsgrp", "data"),
#     Output(name+"obsgrp", "value"),
#     Input ("pest-data", "data"),
# )
# def upload_pest(data):
#     if data is None or data == {}:
#         return no_update
#     contents = data.get("contents")
#     filename = data.get("filename")
#     pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, False)
#     if obs is None:
#         return no_update
#     else:
#         oo = obs[obs.weight>0]
#         data = [gp for gp in oo["group"].unique() if not gp.startswith("regul")]
#     return [data,data]

@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Input( "pest-data", "data"),
)
def fill_iterations(data):
    return update_jco_iter(data, need_rei=True)

@callback(
    Output(name+"grp"   , "data"),
    Output(name+"grp"   , "value"),
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Input( "pest-data", "data"),
)
def upload_grp_weight(data):
    return update_groups(data, "grpType") + update_weights(data, "weight")


@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"grp", "value"),
    State( name+"weight", "value"),
    State( name+"palette", "value"),
    State(name+"invcmap", "checked"),
    State(name+"xscale", "value"),
    State(name+"yscale", "value"),
    # State(name+"obsgrp", "value"),
)
def plot(clicks, data, iiter, grp, weight, palette, invcmap, xscale, yscale, ): # obsgrp
    if clicks is None or clicks == 0:
        return message_figure("Load data, then click 'Plot Residual'.")
    if data is None or data == {}:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Residual'.")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, True)
    if obs is None:
        return message_figure("No .pst file found in zip file.")
    if not jcos:
        return message_figure("No .jco file found in zip file.")
    if not reis:
        return message_figure("No .rei file found in zip file.")
    if int(iiter) not in jcos:
        return message_figure(f"No .jco file found for iteration {iiter}.")
    rei = get_jco_rei(reis, int(iiter))
    if rei is None:
        return message_figure(f"No .rei file found for iteration {int(iiter)-1} or {iiter}.")

    jco0 = jcos[int(iiter)]
    WJ = get_weighted_jco(obs, jco0, weight_col=weight, )


    H = (WJ @ np.linalg.pinv(WJ.T @ WJ)).values @ WJ.T

    leverage = np.diag(H)
    sensitivity = np.sqrt(np.diag(WJ @ WJ.T))

    rei = rei.set_index("obsname").loc[WJ.index]
    r  = rei.residual
    rw = rei.residual * rei[weight]

    sigma2 = np.sum(rw**2) / (len(r) - WJ.shape[1])
    std_res = r / np.sqrt(sigma2 * (1 - leverage))

    rr = pd.DataFrame(dict(
        Observed=rei.measured,
        Simulated=rei.modelled,
        Residual=r,
        Weight=rei[weight],
        Sensitivity=sensitivity,
        Leverage=leverage,
        Std_Residual=std_res,
        ObsGroup=rei[grp],
    ))
    rr["text"] = [f"{i}<br>"+r.to_string().replace("\n", "<br>") for i, r in rr.iterrows()]

    fig = go.Figure()
    for gp, df in rr.groupby("ObsGroup"):
        fig.add_trace(go.Scatter(
                x=df.Leverage,
                y=np.abs(df.Std_Residual),
                text=df["text"],
                mode="markers",
                hoverinfo="text",
                name=gp,
                marker=dict(
                    color=df.Residual,
                    size=df.Sensitivity,
                    colorscale=palette + ('_r' if invcmap else ""),
                    cmin=np.quantile(df.Residual, 0.05), cmid=0, cmax=np.quantile(df.Residual, 0.95),
                    sizemode='area',
                    sizeref=2.*max(df.Sensitivity)/(40.**2),
                    sizemin=4
                )
            )
        )
    fig.update_layout(
        height=700,
        xaxis_title="Observation Leverage",
        yaxis_title="Absolute Standardized Residual",
        hoverlabel=dict(
            font_family="Monospace"
        )
    )
    if xscale == "Log":
        fig.update_xaxes(type="log")
    if yscale == "Log":
        fig.update_yaxes(type="log")



    return fig


@callback(
    Output(name+"dist-iter", "data"),
    Output(name+"dist-iter", "value"),
    Input( "pest-data", "data"),
)
def fill_residual_iterations(data):
    if not data:
        return no_update, no_update
    reis = parse_zip(data, False, True)[4]
    if not reis:
        return no_update, no_update
    iters = [str(i) for i in sorted(reis)]
    return iters, iters[-1]


@callback(
    Output(name+"dist", "figure"),
    Input( name+"dist-update", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"dist-iter", "value"),
    State( name+"grp", "value"),
    State( name+"weight", "value"),
)
def plot_distribution(clicks, data, iiter, grp="obsgrp", weight="weight"):
    if not clicks:
        return message_figure("Load data, then click 'Plot Distribution'.")
    if not data:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Distribution'.")
    reis = parse_zip(data, False, True)[4]
    if not reis or int(iiter) not in reis:
        return message_figure("No residuals for that iteration.")
    rei = reis[int(iiter)]
    grp = grp if grp in rei.columns else "obsgrp"
    df = rei[(rei[weight] > 0) & ~rei.obsgrp.astype(str).str.startswith("regul")]
    if len(df) < 3:
        return message_figure("Too few weighted observations.")
    return distribution_figure(df.obsname.values, (df[weight] * df.residual).values, df[grp].astype(str).values,
                               int(iiter), grp)


def distribution_figure(names, wr, groups, iiter, grp, max_groups=10):
    from scipy.stats import norm
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.08,
                        subplot_titles=("Histogram by group", "Normal Q-Q plot"))
    counts = pd.Series(groups).value_counts()
    shown = list(counts.index[:max_groups])
    colors = px.colors.qualitative.Plotly
    color = {g: colors[i % len(colors)] for i, g in enumerate(shown)}
    bins = dict(start=float(np.min(wr)), end=float(np.max(wr)), size=float(np.ptp(wr)) / 40 or 1)
    for g in shown:
        fig.add_trace(go.Histogram(x=wr[groups == g], name=g, legendgroup=g, marker_color=color[g], opacity=0.6,
                                   xbins=bins, hovertemplate=f"{g}<br>%{{x}}: %{{y}}<extra></extra>"), row=1, col=1)
    mu, sd = float(np.mean(wr)), float(np.std(wr))
    x = np.linspace(np.min(wr), np.max(wr), 200)
    fig.add_trace(go.Scatter(x=x, y=norm.pdf(x, mu, sd) * len(wr) * bins["size"], mode="lines", name="Normal fit",
                             line=dict(color="black", dash="dot"), hoverinfo="skip"), row=1, col=1)

    z, ordered, order = normal_scores(wr)
    g_sorted, n_sorted = groups[order], names[order]
    for g in shown + (["other groups"] if len(counts) > max_groups else []):
        mask = np.isin(g_sorted, shown, invert=True) if g == "other groups" else g_sorted == g
        fig.add_trace(go.Scattergl(x=z[mask], y=ordered[mask], mode="markers", name=g, legendgroup=g,
                                   showlegend=g == "other groups", marker=dict(color=color.get(g, "lightgray"), size=5),
                                   text=n_sorted[mask], hovertemplate="%{text}<br>w r = %{y:.4g}<br>z = %{x:.2f}<extra></extra>"),
                      row=1, col=2)
    a, b = qq_line(z, ordered)
    ends = np.array([z[0], z[-1]])
    fig.add_trace(go.Scatter(x=ends, y=a + b * ends, mode="lines", name=f"Quartile line (slope {b:.3g})",
                             line=dict(color="black", dash="dash"), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=ends, y=ends, mode="lines", name="Standard normal (slope 1)",
                             line=dict(color="gray", dash="dot"), hoverinfo="skip"), row=1, col=2)
    fig.update_layout(
        barmode="overlay", height=520,
        title=f"Weighted residuals of iteration {iiter}: n = {len(wr)}, mean {mu:.3g}, std {sd:.3g}"
              + (f"; the {max_groups} largest of {len(counts)} {grp} groups shown by colour" if len(counts) > max_groups else ""),
        legend=dict(orientation="h", y=-0.15, yanchor="top"),
    )
    fig.update_xaxes(title_text="Weighted residual", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Standard normal quantile", row=1, col=2)
    fig.update_yaxes(title_text="Weighted residual", row=1, col=2)
    return fig
