from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go
from plotting import palettes
from utils import parse_zip, isiterable
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

name = __name__.lstrip("module.") + "-"
fullname = "Observation Sensitivity"

md = dcc.Markdown('''

    Observation sensitivity is similar to parameter sensitivity but for observations or observation groups.

    Please refer to the parameter sensitivty page for more explanation including its calculation.

    ''',
    mathjax=True)

data = [
    {"group": "hk", "items": ["hk1", "hk2", "hk3"]},
    {"group": "vk", "items": ["vk1", "vk2"]}
]
layout = html.Div([
    html.H3(fullname),
    md,
    dmc.Group([
        dmc.Select(id=name+"stype", label="Sensitivity Type:", data=["L1 Norm", "L2 Norm"], value="L2 Norm", style={"width": "150px"}),
        dmc.Select(id=name+"otype", label="Observation Type:",
                   data=["Observations", "Obs Groups"], value="Observations", style={"width": "150px"}),
        dmc.Select(id=name+"nobs", label="Number of Observations:", data=["5", "10", "20", "30", "40", "50"], value="50", style={"width": "150px"}),
        dmc.Select(id=name+"palette", label="Color Palette:", data=list(palettes.keys()), value="Jet", style={"width": "150px"}),
        dmc.Button("Plot Sensitivity", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    dmc.MultiSelect(
        label="Select observation(s)/group(s) to analyze:",
        id=name+"obsselect",
        value=[],
        data=data,
        mb=10,
    ),
    dcc.Loading(dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, )), style={"margin-bottom":"90px"})),
])

@callback(
    Output(name+"obsselect", "data"),
    Input ("pest-data", "data"),
    Input (name+"otype", "value"),
)
def upload_pest(data, otype):
    if data is None or data == {}:
        return no_update
    contents = data.get("contents")
    filename = data.get("filename")
    pst, par, obs, pars, reis, jcos = parse_zip(contents, False, False, False)
    if obs is None:
        return no_update
    else:
        oo = obs[obs.weight>0]
        if otype == "Observations":
            data = []
            for gp, df in oo.groupby("group"):
                data.append({"group": gp, "items": df.obsname.tolist()})
        else:
            data = [gp for gp in oo["group"].unique() if not gp.startswith("regul")]
    return data

@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"stype", "value"),
    State( name+"otype", "value"),
    State(name+"nobs", "value"),
    State(name+"palette", "value"),
    State(name+"obsselect", "value"),
)
def plot(clicks, data, stype, otype, nobs, palette, obsselect):
    if clicks is None or clicks == 0:
        return px.scatter(title="Upload data and click 'Plot Sensitivity' to see plot")
    if data is None or data == {}:
        return px.scatter(title="No data uploaded. Please upload data and click 'Plot Sensitivity' to see plot")
    contents = data.get("contents")
    filename = data.get("filename")
    pst, par, obs, pars, reis, jcos = parse_zip(contents, True, True, True)
    if obs is None:
        return px.scatter(title="No .pst file found in zip")
    if not jcos:
        return px.scatter(title="No .jco file found in zip")
    if not reis:
        return px.scatter(title="No .rei file found in zip")
    if not obsselect:
        return px.scatter(title="No observation/group is selected.")
    if not isiterable(obsselect):
        obsselect = [obsselect]


    ddf = {} # aggregated sensitivity
    sdf = {} # (raw)signed sensitivity
    if otype == "Observations":
        oo = obs[obs["obsname"].isin(obsselect)]
    else:
        oo = obs[obs["group"].isin(obsselect)]
    obsselect = oo.obsname.tolist()
    for i, jco0 in jcos.items():
        df = jco0.loc[oo.obsname]
        sdf[f"Iter {i}"] = (oo["weight"].values[:, None] * df).sum(axis=0)
        if stype == "L1 Norm":
            ddf[f"Iter {i}"] =  (oo["weight"].values[:, None] * df .abs()).sum(axis=0)
        else:
            ddf[f"Iter {i}"] = ((oo["weight"].values[:, None] * df)**2).sum(axis=0)
    sdf = pd.DataFrame(sdf, )
    ddf = pd.DataFrame(ddf, )

    if stype == "L2 Norm":
        ddf = ddf**0.5

    tot = (ddf / ddf.max(axis=0).values[None,:]).sum(axis=1)
    tot = tot.sort_values(ascending=False).iloc[:int(nobs)]
    ddf = ddf.loc[tot.index]
    sdf = sdf.loc[tot.index]
    sdf = pd.DataFrame(np.where(sdf>0, "triangle-up", "triangle-down"), columns=ddf.columns)

    ddf["Iter 0"] = 0
    sdf["Iter 0"] = "circle"

    colors = px.colors.sample_colorscale(palettes[palette], max(ddf.shape[1], len(reis)))

    fig = make_subplots(
        rows=1, cols=2, column_widths=[1, 0.05*len(obsselect)],
        subplot_titles=("Observation Sensitivity", "Observation Values")
    )

    # add parameter update plot
    for i in range(len(reis)):
        c = colors[i]
        df = reis[i]
        fig.add_trace(go.Scatter(
            x=obsselect,
            y=df.set_index("name").loc[obsselect, "modelled"],
            mode="markers",
            name=i,
            marker=dict(color=c),
            showlegend=False,
        ), row=1, col=2)


    # add sensitivity plot
    for i, c in enumerate(ddf.sort_index(axis=1).columns):
        if i == 0:
            parval = par.set_index("parname").loc[ddf.index, "parval0"]
        else:
            parval = pars[i].set_index("parname").loc[ddf.index, "parval"]
        fig.add_trace(go.Scatter(
            x=ddf.index,
            y=ddf[c],
            text=parval.to_string(index=False, header=False).split("\n"),
            mode="lines+markers",
            name=c,
            hovertemplate='Sen:%{y}; Modeled:%{text}',
            line=dict(color=colors[i]),
            marker=dict(size=10, symbol=sdf[c])
        ))
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(title="Parameter",tickangle=90),
        yaxis=dict(title="Observation Sensitivity"),
    )

    return fig
