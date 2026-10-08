

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
from utils import parse_zip, isiterable
from layout import about, graph
from plotting import palettes, message_figure


name = __name__.removeprefix("module.") + "-"
fullname = "Parameter Evolution"


md = dcc.Markdown('''

    This page investigates Parameter update.

    ''',
    mathjax=True)

layout = dmc.Stack(
    [
        html.H3(fullname),
        about(md),
        dmc.Group([
            dmc.Select(id=name+"valcol", label="Value Type:", data=["Raw", "Relative", "Absolute", "Normalized"],
                       value="Relative", persistence=True, style={"width": "150px"}),
            dmc.MultiSelect(id=name+"pargroup", label="Parameter Group:", data=[], value=[],
                            searchable=True, clearable=True, style={"width": "420px"}),
            dmc.MultiSelect(id=name+"partrans", label="Transformation Type:", data=["none", "log", "tied", "fixed"],
                            value=["none", "log", "tied"], persistence=True, style={"width": "320px"}),
        ], mt=10, align="flex-end"),
        dmc.Group([
            dmc.Switch(id=name+"nearbound", label="Highlight Near Bound", size="md", color="blue", pb=10, checked=True),
            dmc.Select(id=name+"bndlimit", label="Bound Limit:", data="100% 99% 98% 95% 90%".split(), value="98%", style={"width": "150px"}),
        ], mt=10, align="flex-end"),
        dmc.Group([
            dmc.Select(id=name+"palette", label="Iteration Colors:", data=list(palettes.keys()), value="Jet", style={"width": "150px"}),
            dmc.Select(id=name+"vscale", label="Value Scale:", data=["linear", "log"], value="linear", style={"width": "150px"}),
            dmc.Select(id=name+"layout", label="Layout:", data=["Horizontal", "Vertical"], value="Horizontal", style={"width": "150px"}),
            dmc.Button("Plot Parameter", id=name+"update-plot", color="blue"),
        ], mt=10, align="flex-end"),
        graph(name+"plot", message_figure("Load data, then click 'Plot Parameter'."), fullname + "", style={"margin-bottom":"90px"}),
    ]
)

# ---------------------------
# Upload callback
# ---------------------------


# ---------------------------
# Plot callback
# ---------------------------

@callback(
    Output(name+"pargroup", "data"),
    Output(name+"pargroup", "value"),
    Input ("pest-data", "data"),
)
def upload_pest(data):
    if data is None or data == {}:
        return [no_update, no_update]
    pst, par, obs, pars, jcos, reis = parse_zip(data, False, False, False)
    if par is None:
        return [no_update, no_update]
    else:
        pargroups = [p for p in par.pargrp.unique() if p.strip() != ""]
    return [pargroups, pargroups]

@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"valcol", "value"),
    State( name+"pargroup", "value"),
    State( name+"partrans", "value"),
    State( name+"nearbound", "checked"),
    State(name+"bndlimit", "value"),
    State(name+"palette", "value"),
    State(name+"vscale", "value"),
    State(name+"layout", "value"),
)
def update_plot(clicks, data, valcol, pargroup, partrans, nearbound, bndlimit, palette, vscale, layout):
    # print("pargroup value:", pargroup)
    if clicks is None or clicks == 0:
        return message_figure("Load data, then click 'Plot Parameter'.")

    if data is None:
        return message_figure("Load data, then click 'Plot Parameter'.")

    pst, par, obs, pars, reis, jcos = parse_zip(data, True)

    if par is None:
        return message_figure("No .pst file found in zip")

    if pars is None or len(pars) == 0:
        return message_figure("No .par file found in zip")

    niter = len(pars)
    dfs = []
    bl = float(bndlimit.strip("%"))/100
    for i in range(niter+1):
        df = par.set_index("parname")
        if i == 0:
            p = par
            df["Raw"] = p.set_index("parname")["parval0"]
        else:
            p = pars[i]
            df["Raw"] = p.set_index("parname")["parval"]
        df["Absolute"] = df["Raw"] * df["scale"] + df["offset"]
        df["Relative"] = df["Raw"] / df["parval0"]
        df["parvalnorm"] = (df.Raw - df.parlbnd) / (df.parubnd - df.parlbnd)
        df["parvallognorm"] = (np.log10(df.Raw) - np.log10(df.parlbnd)) / (np.log10(df.parubnd) - np.log10(df.parlbnd))
        df["Normalized"] = np.where(df.partrans=="log", df.parvallognorm, df.parvalnorm)
        df["nearbound"] = (df["Normalized"] > bl) | (df["Normalized"] < 1 - bl)
        df["iteration"] = i
        dfs.append(df)
    df = pd.concat(dfs)
    dff = df[
        df.pargrp.isin(pargroup) &
        df.partrans.isin(partrans) &
        # (df.nearbound == nearbound) &
        True
    ].reset_index()

    n_par = (dff["iteration"]==0).sum()
    height = max(600, n_par * 18)

    fig = go.Figure()
    colors = px.colors.sample_colorscale(palettes[palette], niter+1)
    for ii, dfi in dff.groupby("iteration"):
        fig.add_trace(go.Scatter(
            x=dfi["parname"] if layout=="Horizontal" else dfi[valcol],
            y=dfi[valcol] if layout=="Horizontal" else dfi["parname"],
            mode="markers",
            marker=dict(
                color=colors[ii],
                size=8,
                symbol=np.where(dfi["nearbound"], "diamond" if nearbound else "circle", "circle"),
            ),
            name=f"Iteration {ii}"
        ))

    if layout == "Horizontal":
        if vscale == "log":
            fig.update_yaxes(type="log")
        fig.update_layout(
            height=700,
            margin=dict(l=150, r=50, t=50, b=50),
            xaxis=dict(type="category", automargin=True),
            xaxis_title="Parameter Name",
            xaxis_range=[-3, n_par+2],
            yaxis_title="Parameter " + valcol + " Value",
        )
    else:
        if vscale == "log":
            fig.update_xaxes(type="log")
        fig.update_layout(
            height=height,
            margin=dict(l=50, r=150, t=50, b=50),
            yaxis=dict(type="category", automargin=True),
            xaxis_title="Parameter " + valcol + " Value",
            yaxis_title="Parameter Name",
            yaxis_range=[n_par+2, -3,],
        )

    return fig
