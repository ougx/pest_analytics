from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import parse_zip
from plotting import message_figure
from layout import about, graph
import compare

name = __name__.removeprefix("module.") + "-"
fullname = "Parameter Changes"

ADJUSTABLE, ALL = "Adjustable (log / none)", "All, with tied and fixed"

md = dcc.Markdown('''

    How the estimated parameters differ between the loaded run (the reference) and another run added under
    "Runs to compare" on the Upload Data page, matched by name (a pilot point that moved but kept its name is the same parameter).

    - **Change by parameter**: $\\log_{10}$ of the ratio for parameters log-transformed in both runs (1 = ten times larger),
      the relative change $(p_2 - p_1)/|p_1|$ for the others. Crosses mark parameters on a bound in the second run.
    - **Values**: log-transformed parameters, one run against the other, with the 1:1 line.

    Each run is represented by its best iteration (lowest measurement $\\phi$, from the residual files) unless you choose
    another; iteration 0 is the initial values in the control file. The Setup Changes page lists changed bounds, transforms
    and initial values.

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
        dmc.Select(id=name+"which", label="Parameters:", data=[ADJUSTABLE, ALL], value=ADJUSTABLE,
                   style={"width": "200px"}),
        dmc.Button("Compare Parameters", id=name+"update", color="blue"),
    ], mt=10, align="flex-end"),
    html.Div(id=name+"summary", style={"margin-top": "20px"}),
    graph(name+"change", message_figure("Add a run to compare on the Upload Data page, then click 'Compare Parameters'."),
          "Parameter Changes"),
    graph(name+"values", message_figure(), "Parameter Values", style={"margin-bottom": "90px"}),
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


def iteration_options(data):
    pars = parse_zip(data, True)[3] if data else None
    return [compare.BEST, compare.LAST, "0"] + [str(i) for i in sorted(pars or {})]


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


def run_iteration(data, choice):
    """(iteration, parameter values) of a run for an iteration choice"""
    pst, par, obs, pars, reis, jcos = parse_zip(data, True, True)
    if choice == compare.LAST:
        i = max(pars) if pars else 0
    elif choice in (None, compare.BEST):
        i = compare.pick_iteration(reis, compare.BEST) if reis else (max(pars) if pars else 0)
    else:
        i = int(choice)
    return i, compare.parameter_values(par, pars, i)


def change_figure(m, label_from, label_to):
    m = m.sort_values(["pargrp", "change"], ascending=[True, False])
    groups = list(dict.fromkeys(m.pargrp))
    colors = dict(zip(groups, px.colors.qualitative.Plotly * (len(groups) // 10 + 1)))
    fig = go.Figure()
    hover = ("%{y} (%{customdata[0]}, %{customdata[1]})<br>%{customdata[2]:.4g} → %{customdata[3]:.4g}"
             "<br>change: %{x:.3g}<extra></extra>")
    for g in groups:
        d = m[m.pargrp == g]
        fig.add_trace(go.Bar(
            x=d.change, y=d.index, orientation="h", name=g, marker_color=colors[g],
            customdata=np.c_[d.pargrp, np.where(d.log, "log10 ratio", "relative change"), d.value_from, d.value_to],
            hovertemplate=hover,
        ))
    d = m[m.at_bound_to]
    if len(d):
        fig.add_trace(go.Scatter(x=d.change, y=d.index, mode="markers", name=f"On a bound in {label_to}",
                                 marker=dict(symbol="x", color="black", size=8), hoverinfo="skip"))
    fig.update_layout(
        height=18 * len(m) + 220, barmode="overlay",
        title=dict(text=f"Change by parameter, {label_from} → {label_to}", yref="container", y=1, yanchor="top",
                   pad=dict(t=15)),
        xaxis=dict(title="log10(new / old) for log-transformed parameters, (new - old) / |old| otherwise", side="top"),
        yaxis=dict(title="Parameter", categoryorder="array", categoryarray=list(m.index), range=[len(m) - 0.5, -0.5]),
        legend=dict(orientation="h", y=-30 / (18 * len(m) + 220), yanchor="top"),
        margin=dict(t=130),
    )
    return fig


def values_figure(m, label_from, label_to):
    d = m[m.log]
    if not len(d):
        return message_figure("No parameter is log-transformed in both runs.")
    lo, hi = d[["value_from", "value_to"]].min().min(), d[["value_from", "value_to"]].max().max()
    fig = go.Figure(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="gray", dash="dash"),
                               name="1:1", hoverinfo="skip"))
    colors = px.colors.qualitative.Plotly
    for i, (g, gd) in enumerate(d.groupby("pargrp")):
        fig.add_trace(go.Scatter(
            x=gd.value_from, y=gd.value_to, mode="markers", name=g, text=gd.index,
            marker=dict(color=colors[i % len(colors)], size=9, line=dict(width=0.5, color="white"),
                        symbol=np.where(gd.at_bound_to, "x", "circle")),
            hovertemplate="%{text}<br>%{x:.4g} → %{y:.4g}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(height=600, title="Values of log-transformed parameters (crosses: on a bound in the second run)",
                      xaxis=dict(title=label_from, type="log"), yaxis=dict(title=label_to, type="log"))
    return fig


@callback(
    Output(name+"summary", "children"),
    Output(name+"change", "figure"),
    Output(name+"values", "figure"),
    Input(name+"update", "n_clicks"),
    State("pest-data", "data"),
    State("compare-runs", "data"),
    State(name+"run", "value"),
    State(name+"direction", "value"),
    State(name+"ref-iter", "value"),
    State(name+"other-iter", "value"),
    State(name+"which", "value"),
    prevent_initial_call=True,
    running=[(Output(name+"update", "loading"), True, False)],
)
def compare_parameters(n_clicks, data, runs, run, direction=compare.FORWARD, ref_iter=compare.BEST,
                       other_iter=compare.BEST, which=ADJUSTABLE):
    def only(message, color="yellow"):
        return [dmc.Alert(message, color=color, variant="light"), message_figure(message), message_figure()]
    if not n_clicks:
        return [no_update] * 3
    if not data:
        return only("No data loaded. Load the reference run on the Upload Data page.")
    if not runs or run is None or int(run) >= len(runs):
        return only("Add another run under 'Runs to compare' on the Upload Data page, then choose it here.")
    other = runs[int(run)]
    i_ref, v_ref = run_iteration(data, ref_iter)
    i_other, v_other = run_iteration(other, other_iter)
    if v_ref is None or v_other is None:
        return only(f"No .par file for the chosen iteration ({i_ref if v_ref is None else i_other}).", "red")
    side = {"ref": (v_ref, f"{parse_zip(data, False)[0]} (iteration {i_ref})"),
            "other": (v_other, f"{other['name']} (iteration {i_other})")}
    to_is_ref = direction == compare.BACKWARD
    (v_from, label_from), (v_to, label_to) = (side["other"], side["ref"]) if to_is_ref else (side["ref"], side["other"])
    m = compare.paired_parameters(v_from, v_to)
    only_from = v_from.index.difference(v_to.index)
    only_to = v_to.index.difference(v_from.index)
    trans = m[m.partrans_from != m.partrans_to]
    if which == ADJUSTABLE:
        m = m[m.partrans_from.isin(["log", "none"]) & m.partrans_to.isin(["log", "none"])]
    if not len(m):
        return only("No parameter to compare.")

    big = int((m.change.abs() > np.where(m.log, np.log10(2), 1)).sum())
    text = [f"{len(m)} parameters compared, {label_from} → {label_to}. ",
            f"{big} changed by more than a factor of 2 (or 100% for untransformed ones); "
            f"{int(m.at_bound_from.sum())} on a bound in {label_from}, {int(m.at_bound_to.sum())} in {label_to}."]
    if len(trans):
        text.append(f" Transform changed for {len(trans)}: " + ", ".join(
            f"{p} ({a} → {b})" for p, a, b in trans[["partrans_from", "partrans_to"]].head(8).itertuples())
            + (", ..." if len(trans) > 8 else "") + ".")
    if len(only_from) or len(only_to):
        text.append(f" Only in {label_from}: {len(only_from)}; only in {label_to}: {len(only_to)}.")
    summary = dmc.Alert("".join(text), color="blue", variant="light")
    return [summary, change_figure(m, label_from, label_to), values_figure(m, label_from, label_to)]
