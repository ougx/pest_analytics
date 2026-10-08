from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
from plotting import palettes, residual_stats, stat_names, message_figure, iteration_colors, select_iterations
import pandas as pd
from utils import parse_zip, isiterable
from pst import get_obssim, get_obscor
from layout import update_weights, update_groups, update_obs, update_rei_iter, about, graph
from plotly.subplots import make_subplots
import numpy as np

name = __name__.removeprefix("module.") + "-"
fullname = "Time Series Plot"

md = dcc.Markdown('''

    This Tab allows you to make time series plots of the observed and modeled values.
    To use this function, the `obs.csv` file needs to be included in the zip file.
    `obs.csv` must include two columns: `obsname` defining the observation names that match those in the `pst` file;
    `time` column defining the timestamps for each observations.

    Observations could be regrouped by the columns with names starting with `grp`. For example,

    ```
    | obsname    | time | grpSite | grpLayer   | grpType |
    | ---------- | ---- | ------- | ---------- | ------- |
    | well1_1991 | 1991 | well    | unconfined | head    |
    | well1_1992 | 1992 | well    | unconfined | head    |
    | well1_1993 | 1993 | well    | unconfined | head    |
    | well1_1994 | 1994 | well    | unconfined | head    |
    | well2_1991 | 1991 | wel2    | unconfined | head    |
    | well2_1992 | 1992 | wel2    | unconfined | head    |
    | well2_1993 | 1993 | wel2    | unconfined | head    |
    | well2_1994 | 1994 | wel2    | unconfined | head    |
    | well3_1990 | 1990 | wel3    | confined   | head    |
    | well3_1991 | 1991 | wel3    | confined   | head    |
    | well3_1992 | 1992 | wel3    | confined   | head    |
    | well3_1993 | 1993 | wel3    | confined   | head    |
    | well3_1994 | 1994 | wel3    | confined   | head    |
    | gage1_1990 | 1990 | gage1   | top        | flow    |
    | gage1_1991 | 1991 | gage1   | top        | flow    |
    | gage1_1992 | 1992 | gage1   | top        | flow    |
    | gage1_1993 | 1993 | gage1   | top        | flow    |
    | gage1_1994 | 1994 | gage1   | top        | flow    |

    ```
    The time series can plot by the columns "grpSite".

    The time column is automatically parsed by the `pandas.to_datetime` function.
    Please check if time stamps in this column can be correctly parsed before uploading the file.
    ''',
    mathjax=True)


layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"grp", label="Site Column:", data=["grpSite"], value="grpSite", style={"width": "150px"}),
        dmc.Select(id=name+"site", label="Select Site:", data=["Well1"], value="Well1", style={"width": "150px"}, searchable=True),
        # dmc.Switch(id=name+"norm", label="Normalized", checked=True, style={"width": "150px"}),
        dmc.MultiSelect(id=name+"iters", label="Iterations:", data=["All"], value=[], clearable=True,
                        style={"width": "260px"}),
        dmc.Select(id=name+"zeroweight", label="Zero Weight:", data=["Hide", "Greyout", "DoNothing"], value="DoNothing", style={"width": "150px"}),
    ], mt=10, align="flex-end"),
    dmc.Group([
        dmc.Select(id=name+"plottype", label="Plot Type:", data=["Lines+Markers", "Lines", "Markers", ], value="Lines+Markers", style={"width": "150px"}),
        dmc.Select(id=name+"palette", label="Iteration Colors:", data=list(palettes.keys()), value="Jet", style={"width": "150px"}),
        dmc.Switch(id=name+"invcmap", label="Inverse colormap", checked=False, style={"width": "150px"}),
        dmc.Select(id=name+"hovermode", label="Hover:", data=[
            {"value": "closest", "label": "Closest point"},
            {"value": "x unified", "label": "Unified by date (all series)"},
        ], value="closest", style={"width": "220px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        dmc.Button("Plot Time Series", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    graph(name+"plot", message_figure("Load data, then click 'Plot Time Series'."), fullname + "", style={"margin-bottom":"90px"}),
])




@callback(
    Output(name+"grp", "data"),
    Output(name+"grp", "value"),
    Input( "pest-data", "data"),
)
def upload_pest(data):
    return update_groups(data, "grpSite")

@callback(
    Output(name+"site", "data"),
    State( "pest-data", "data"),
    Input( name+"grp" , "value"),
)
def upload_pest(data, grp):
    return update_obs(data, "Observation Group", grp, filter_zeroweight=False)

@callback(
    Output(name+"site", "value"),
    Input(name+"site", "data"),
    State(name+"site", "value"),
)
def upload_pest(data, site):
    # keep a site chosen elsewhere (e.g. clicked on the Residual Map) if it is still an option
    return site if site in data else data[0]

@callback(
    Output(name+"iters", "data"),
    Output(name+"iters", "value"),
    Input( "pest-data", "data"),
)
def upload_iterations(data):
    # default: the last iteration only (plotting every iteration of a large dataset is slow)
    return update_rei_iter(data)


@callback(
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Input( "pest-data", "data"),
)
def upload_weight_columns(data):
    return update_weights(data, "weight")


@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"grp", "value"),
    State( name+"site", "value"),
    State( name+"zeroweight", "value"),
    State( name+"plottype", "value"),
    State( name+"palette", "value"),
    State( name+"invcmap", "checked"),
    State( name+"iters", "value"),
    State( name+"hovermode", "value"),
    State( name+"weight", "value"),
)
def plot(clicks, data, grp, site, zeroweight, plottype, palette, invcmap, iters=None, hovermode="closest", weight="weight"):
    if clicks is None or clicks == 0:
        return message_figure("Load data, then click 'Plot Time Series'.")
    if data is None or data == {}:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Time Series'.")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, False)
    if obs is None:
        return message_figure("No .pst file found in zip file.")
    if not reis:
        return message_figure("No .rei file found in zip file.")
    if "time" not in obs.columns:
        return message_figure("No 'time' column found. Add obs.csv with obsname and time columns to the zip file.")


    color_of = iteration_colors(reis, palette, invcmap)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        # vertical_spacing=0.1,
        specs=[[{"type": "xy"}], [{"type": "table"}]]
    )

    ts = obs[obs[grp]==site]
    if zeroweight=="Hide":
        ts = ts[ts[weight]>0]
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(ts["time"]),
        y=ts["obsval"],
        text=ts["obsname"],
        mode=plottype.lower(),
        name="Observed",
        hovertemplate='%{text}: %{y}',
        line=dict(color="black"),
        marker=dict(color=["black" if w>0 else "grey" for w in ts[weight]] if zeroweight=="Greyout" else "black")
    ))

    sims = {}
    for i, df in select_iterations(reis, iters):
        color = color_of[i]
        #print(df)
        ts = df[df[grp]==site]
        if zeroweight=="Hide":
            ts = ts[ts[weight]>0]
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(ts["time"]),
            y=ts["modelled"],
            text=ts["obsname"],
            mode=plottype.lower(),
            name="Iter " + str(i),
            hovertemplate='%{text}: %{y}',
            line=dict(color=color),
            marker=dict(color=[color if w>0 else "grey" for w in ts[weight]] if zeroweight=="Greyout" else color)
        ))
        sims["Iter " + str(i)] = residual_stats(ts, weight)

    sims = pd.DataFrame(sims)
    sims.index = stat_names

    # Add table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Statistics<br>(weight > 0)"] +list(sims.columns),
                fill_color='royalblue',
                line_color='darkslategray',
                align="center",
                font=dict(size=12, color="white", weight="bold"),
            ),
            cells=dict(
                values=[l.split() for l in sims.T.to_string(index=False).splitlines()],
                fill_color=['paleturquoise'] +  ['white']*(sims.shape[1]),
                line_color='darkslategray',
                align="center"
            )
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=700,
        hovermode=hovermode,
    )
    return fig
