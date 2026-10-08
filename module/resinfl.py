import numpy as np
import pandas as pd
from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import plotly.express as px
from utils import parse_zip
import plotly.graph_objects as go
from pst import get_weighted_jco, get_jco_rei
from layout import update_groups, update_weights, about

name = __name__.lstrip("module.") + "-"
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
        r^*_j = \\frac{r_j}{\sqrt{\sigma^2(1 - H_{jj})}}
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
        dmc.Select(id=name+"iter", label="Iteration:", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"grp"   , label="Grouped by:" , data=["obsgrp"], value="obsgrp", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"palette", label="Color Palette:", data=["Jet","Plotly3","Viridis","BrBG","RdBu","RdYlBu"], value="Jet", style={"width": "150px"}),
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
    dcc.Loading(dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, )), style={"margin-bottom":"90px"})),
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
    Input( "pest-data", "data"),
)
def upload_pest(data):
    if data is None or data == {}:
        return no_update
    contents = data.get("contents")
    filename = data.get("filename")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, True)
    if pst and jcos:
        return [str(i) for i in jcos.keys()]
    else:
        return no_update

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
        return px.scatter(title="Upload data and click 'Plot Residual' to see plot")
    if data is None or data == {}:
        return px.scatter(title="No data uploaded. Please upload data and click 'Plot Residual' to see plot")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, True)
    if obs is None:
        return px.scatter(title="No .pst file found in zip file.")
    if not jcos:
        return px.scatter(title="No .jco file found in zip file.")
    if not reis:
        return px.scatter(title="No .rei file found in zip file.")
    if int(iiter) not in jcos:
        return px.scatter(title=f"No .jco file found for iteration {iiter}.")
    rei = get_jco_rei(reis, int(iiter))
    if rei is None:
        return px.scatter(title=f"No .rei file found for iteration {int(iiter)-1} or {iiter}.")

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
