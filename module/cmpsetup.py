from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import parse_zip
from plotting import message_figure
from layout import about, graph
import compare

name = __name__.removeprefix("module.") + "-"
fullname = "Setup Changes"

FORWARD, BACKWARD = compare.FORWARD, compare.BACKWARD

md = dcc.Markdown('''

    What changed in the control file between the loaded run (the reference) and another run added under
    "Runs to compare" on the Upload Data page. The control files are compared with
    [makepst](https://github.com/ougx/makePst) (`makepst diff`): parameters, observations and prior information
    matched by name, column by column, numbers with a tolerance (so reformatting is not a change), plus
    parameter groups, control values, `++` options, template / instruction files, the command line and the
    comment lines at the top of the file (a run log, if you keep one there).

    The weight plot shows each changed observation weight as the ratio new / old, against time when an
    obs.csv with a time column is in either zip. Choose the direction to read the changes from the older run
    to the newer one.

    ''')

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"run", label="Other run:", description="add runs on the Upload Data page",
                   data=[], placeholder="no runs added", style={"width": "260px"}),
        dmc.Select(id=name+"direction", label="Changes from:", data=[FORWARD, BACKWARD], value=FORWARD,
                   style={"width": "230px"}),
        dmc.Button("Compare Setup", id=name+"update", color="blue"),
    ], mt=10, align="flex-end"),
    html.Div(id=name+"summary", style={"margin-top": "20px"}),
    html.H4("Observations", style={"margin-top": "20px"}),
    html.Div(id=name+"obs-summary"),
    graph(name+"weights", message_figure(), "Weight Changes", style={"margin-bottom": "20px"}),
    graph(name+"obs-table", message_figure(), "Observation Changes", style={"margin-bottom": "20px"}),
    html.H4("Parameters", style={"margin-top": "20px"}),
    html.Div(id=name+"par-summary"),
    graph(name+"par-table", message_figure(), "Parameter Changes", style={"margin-bottom": "20px"}),
    html.H4("Prior information", style={"margin-top": "20px"}),
    graph(name+"prior-table", message_figure(), "Prior Information Changes", style={"margin-bottom": "20px"}),
    html.H4("Other control-file changes", style={"margin-top": "20px"}),
    html.Div(id=name+"other", style={"margin-bottom": "90px"}),
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


def html_table(df, index=True):
    """a small DataFrame as a Mantine table"""
    df = df.reset_index() if index else df
    body = [[("" if pd.isna(v) else f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)) for v in row]
            for row in df.itertuples(index=False)]
    return dmc.Table(data={"head": [str(c) for c in df.columns], "body": body},
                     striped=True, highlightOnHover=True, withTableBorder=True, fz="sm", mb=10)


def table_figure(df, empty):
    """a long table as a figure (scrollable, with CSV download), or a message if there is nothing"""
    if df is None or not len(df):
        return message_figure(empty, height=120)
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color="royalblue", font=dict(color="white"),
                    align="left"),
        cells=dict(values=[df[c].astype(str).tolist() for c in df.columns], align="left"),
    ))
    fig.update_layout(height=min(600, 60 + 26 * len(df)), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def weight_figure(changes, label_from, label_to):
    """ratio new / old of every changed weight, against time if known; zero weights at the plot edges"""
    both = changes[changes.status == "changed"]
    w = both[~np.isclose(both.weight_from, both.weight_to, rtol=1e-6)].copy()
    if not len(w):
        return message_figure("No observation weight changed.", height=120)
    if "time" in w.columns and w.time.notna().any():
        w["x"] = pd.to_datetime(w.time, errors="coerce")
        xtitle = "Time (from obs.csv)"
    else:
        w["x"] = np.arange(len(w))
        xtitle = "Observation (order in the control file)"
    finite = w.weight_ratio[np.isfinite(w.weight_ratio) & (w.weight_ratio > 0)]
    lo = finite.min() / 3 if len(finite) else 0.1
    hi = finite.max() * 3 if len(finite) else 10
    w["y"] = w.weight_ratio.where(w.weight_from > 0, hi).where(w.weight_to > 0, lo)
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (g, d) in enumerate(w.groupby("obsgrp")):
        fig.add_trace(go.Scattergl(
            x=d.x, y=d.y, mode="markers", name=g, marker=dict(color=colors[i % len(colors)], size=6),
            text=d.index, customdata=d[["weight_from", "weight_to"]].values,
            hovertemplate=f"%{{text}}<br>weight {label_from}: %{{customdata[0]:.4g}}"
                          f"<br>weight {label_to}: %{{customdata[1]:.4g}}<extra>%{{fullData.name}}</extra>",
        ))
    for y, text in ((lo, "set to 0"), (hi, "raised from 0")):
        if (w.y == y).any():
            fig.add_hline(y=y, line=dict(color="grey", dash="dot"), annotation_text=text, annotation_position="right")
    fig.update_layout(
        title=f"{len(w)} weights changed: ratio {label_to} / {label_from}",
        height=450, xaxis_title=xtitle, yaxis=dict(title="Weight ratio", type="log"),
        hovermode="closest",
    )
    return fig


@callback(
    Output(name+"summary", "children"),
    Output(name+"obs-summary", "children"),
    Output(name+"weights", "figure"),
    Output(name+"obs-table", "figure"),
    Output(name+"par-summary", "children"),
    Output(name+"par-table", "figure"),
    Output(name+"prior-table", "figure"),
    Output(name+"other", "children"),
    Input(name+"update", "n_clicks"),
    State("pest-data", "data"),
    State("compare-runs", "data"),
    State(name+"run", "value"),
    State(name+"direction", "value"),
    prevent_initial_call=True,
    running=[(Output(name+"update", "loading"), True, False)],
)
def compare_setup(n_clicks, data, runs, run, direction=FORWARD):
    def only(message, color="yellow"):
        return [dmc.Alert(message, color=color, variant="light")] + [None] + [message_figure()] * 2 + [None] \
            + [message_figure()] * 2 + [None]
    if not n_clicks:
        return [no_update] * 8
    if compare.makepst is None:
        return only("The Compare pages need makepst: pip install makepst.", "red")
    if not data:
        return only("No data loaded. Load the reference run on the Upload Data page.")
    if not runs or run is None or int(run) >= len(runs):
        return only("Add another run under 'Runs to compare' on the Upload Data page, then choose it here.")
    ref, other = data, runs[int(run)]
    ref_name = parse_zip(ref, False)[0]
    other_name = other["name"]
    try:
        pst_ref, pst_other = compare.run_pst(ref), compare.run_pst(other)
    except Exception as e:
        return only(f"Could not read the control files: {e}", "red")
    if pst_ref is None or pst_other is None:
        return only("The control file of one of the runs could not be found.", "red")
    obs_ref, obs_other = parse_zip(ref, False)[2], parse_zip(other, False)[2]
    if direction == BACKWARD:
        (pst_from, obs_from, label_from), (pst_to, obs_to, label_to) = \
            (pst_other, obs_other, other_name), (pst_ref, obs_ref, ref_name)
    else:
        (pst_from, obs_from, label_from), (pst_to, obs_to, label_to) = \
            (pst_ref, obs_ref, ref_name), (pst_other, obs_other, other_name)

    diff = compare.setup_diff(pst_from, pst_to)

    # summary: counts, makepst's one-line summary and the new run-log lines
    counts = compare.summary_counts(pst_from, pst_to).rename(columns={"from": label_from, "to": label_to})
    summary = [dmc.Alert(f"{label_from} → {label_to}: {diff.summary()}", color="blue", variant="light", mb=10),
               html_table(counts.rename_axis("Item"))]
    log = diff.comments
    if len(log):
        summary += [dmc.Text(f"Comment lines (run log) {'added' if (log.change == 'added').all() else 'changed'} "
                             f"in {label_to}:", size="sm", fw=500),
                    html.Pre("\n".join(f"{'+' if c == 'added' else '-'} {t}" for c, t in zip(log.change, log.comment)),
                             style={"fontSize": "12px", "overflowX": "auto"})]

    # observations
    changes = compare.observation_changes(obs_from, obs_to)
    groups = compare.observation_group_summary(changes)
    obs_summary = html_table(groups.rename_axis("Group")) if len(groups) else None
    obs_rows = diff.obs.assign(group=diff.obs.OBSNME.str.lower().map(changes.obsgrp))
    obs_rows = obs_rows[["OBSNME", "group", "change", "column", "old", "new"]]

    # parameters
    par_groups = pst_to.par.set_index(pst_to.par.PARNME.str.lower()).PARGP
    par_groups = pd.concat([par_groups, pst_from.par.set_index(pst_from.par.PARNME.str.lower()).PARGP])
    par_groups = par_groups[~par_groups.index.duplicated()]
    par_counts = compare.change_counts(diff.par, "PARNME", par_groups)
    par_summary = html_table(par_counts, index=False) if len(par_counts) else \
        dmc.Text("No parameter changed.", size="sm", c="dimmed")
    par_rows = diff.par.assign(group=diff.par.PARNME.str.lower().map(par_groups))
    par_rows = par_rows[["PARNME", "group", "change", "column", "old", "new"]]

    # prior information: show the equation of added and removed ones
    eq = pd.concat([p.prior.set_index(p.prior.PINME.str.lower()) for p in (pst_to, pst_from)])
    eq = eq[~eq.index.duplicated()]
    prior_rows = diff.prior.copy()
    key = prior_rows.PINME.str.lower()
    prior_rows["group"] = key.map(eq.OBGNME)
    whole = prior_rows.change.isin(["added", "removed"])
    prior_rows.loc[whole, "column"] = "EQ, WEIGHT"
    text = key.map(lambda k: f"{eq.EQ.get(k, '')}  (weight {eq.WEIGHT.get(k, '')})")
    prior_rows.loc[prior_rows.change == "added", "new"] = text
    prior_rows.loc[prior_rows.change == "removed", "old"] = text
    prior_rows = prior_rows[["PINME", "group", "change", "column", "old", "new"]]

    # everything else makepst compares
    other = []
    titles = {"control": "Control values", "pestpp": "++ options", "pargp": "Parameter groups",
              "io": "Template / instruction files", "cmd": "Model command line"}
    for t, title in titles.items():
        df = getattr(diff, t)
        if len(df):
            other += [dmc.Text(title, size="sm", fw=500), html_table(df, index=False)]
    if not other:
        other = [dmc.Text("No other changes.", size="sm", c="dimmed")]

    return [summary, obs_summary, weight_figure(changes, label_from, label_to),
            table_figure(obs_rows, "No observation changed."), par_summary,
            table_figure(par_rows, "No parameter changed."),
            table_figure(prior_rows, "No prior-information equation changed."), other]
