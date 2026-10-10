from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotting import message_figure
from layout import about, graph
from rec import read_rec

name = __name__.removeprefix("module.") + "-"
fullname = "Run Record"

md = dcc.Markdown('''

    What the PEST / PEST_HP run record (`case.rec` in the zip) says about each optimisation iteration.
    Iteration N of the record computes `.jco.N` at the parameters of iteration N-1 and produces `.par.N`.

    - **Objective function**: measurement $\\phi$ by observation group at the start of each iteration, with the
      target measurement $\\phi$ of regularisation mode (FRACPHIM / PHIMLIM) and, below, the regularisation $\\phi$
      and the regularisation weight factor.
    - **Marquardt lambda tests**: every parameter upgrade PEST tried, as $\\phi$ / starting $\\phi$ against $\\lambda$
      (colour: upgrade fraction, PEST_HP). Crosses at the top are tests whose model run failed. The circled test is
      the one PEST accepted, identified by matching its measurement $\\phi$ to the next iteration's starting value
      (the record does not say which it was, so the last iteration has none).
    - **Model runs**: runs per iteration, split into Jacobian runs and lambda tests (failed ones apart). Jacobian
      runs are the change in "model calls so far" less the lambda tests.
    - The table lists everything by iteration, including the parameters PEST reported as having no effect on the
      observations (an all-zero Jacobian column) or frozen.

    ''', mathjax=True)

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Button("Read Run Record", id=name+"update", color="blue"),
    ], mt=10, align="flex-end"),
    html.Div(id=name+"summary", style={"margin-top": "20px"}),
    graph(name+"phi", message_figure("Load data, then click 'Read Run Record'."), "Objective Function by Iteration"),
    graph(name+"lambda", message_figure(), "Lambda Tests"),
    graph(name+"runs", message_figure(), "Model Runs"),
    graph(name+"table", message_figure(), "Run Record by Iteration", style={"margin-bottom": "90px"}),
])


def phi_figure(r):
    it = r["iterations"]
    groups = r["groups"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06,
                        specs=[[{}], [{"secondary_y": True}]],
                        subplot_titles=("Measurement φ at the start of each iteration", "Regularisation"))
    meas = groups[~groups.group.str.lower().str.startswith("regul")]
    colors = px.colors.qualitative.Plotly
    for i, (g, d) in enumerate(meas.groupby("group", sort=False)):
        fig.add_trace(go.Bar(x=d.iteration, y=d.phi, name=g, marker_color=colors[i % len(colors)],
                             hovertemplate="iteration %{x}<br>φ = %{y:.4g}<extra>%{fullData.name}</extra>"), row=1, col=1)
    if len(meas) == 0:
        fig.add_trace(go.Bar(x=it.index, y=it.phi_measurement, name="measurement φ"), row=1, col=1)
    if it.phi_target.notna().any():
        fig.add_trace(go.Scatter(x=it.index, y=it.phi_target, mode="lines+markers", name="Target measurement φ",
                                 line=dict(color="black", dash="dash")), row=1, col=1)
    if it.phi_regularization.notna().any():
        fig.add_trace(go.Scatter(x=it.index, y=it.phi_regularization, mode="lines+markers", name="Regularisation φ",
                                 line=dict(color="#7f7f7f")), row=2, col=1)
        fig.add_trace(go.Scatter(x=it.index, y=it.weight_factor, mode="lines+markers", name="Regularisation weight factor",
                                 line=dict(color="#9467bd", dash="dot")), row=2, col=1, secondary_y=True)
    fig.update_layout(barmode="stack", height=650, legend=dict(orientation="h", y=-0.12, yanchor="top"))
    fig.update_yaxes(title_text="φ", row=1, col=1)
    fig.update_yaxes(title_text="Regularisation φ", row=2, col=1)
    fig.update_yaxes(title_text="Weight factor", type="log", row=2, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Iteration", dtick=1, row=2, col=1)
    return fig


def lambda_figure(r):
    lam, it = r["lambdas"], r["iterations"]
    if not len(lam):
        return message_figure("No lambda tests in the run record.")
    iters = sorted(lam.iteration.unique())
    ncol = min(4, len(iters))
    nrow = int(np.ceil(len(iters) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.12,
                        subplot_titles=[f"Iteration {i}" for i in iters])
    ok = lam[~lam.failed]
    top = ok.phi_ratio.max() * 1.03 if len(ok) else 1.1
    has_fraction = lam.fraction.notna().any()
    for k, i in enumerate(iters):
        row, col = k // ncol + 1, k % ncol + 1
        d = ok[ok.iteration == i]
        fig.add_trace(go.Scatter(
            x=d["lambda"], y=d.phi_ratio, mode="markers", name="lambda test", showlegend=False,
            marker=dict(size=7, color=d.fraction if has_fraction else "#1f77b4", colorscale="Viridis",
                        cmin=lam.fraction.min(), cmax=lam.fraction.max(), showscale=has_fraction and k == 0,
                        colorbar=dict(title="Upgrade<br>fraction", len=0.5, y=1, yanchor="top")),
            customdata=np.c_[d.fraction, d.phi],
            hovertemplate="λ = %{x:.4g}, fraction %{customdata[0]}<br>φ = %{customdata[1]:.4g} "
                          "(%{y:.3f} of starting φ)<extra></extra>",
        ), row=row, col=col)
        f = lam[(lam.iteration == i) & lam.failed]
        if len(f):
            fig.add_trace(go.Scatter(
                x=f["lambda"], y=[top] * len(f), mode="markers", name="model run failed", showlegend=k == 0,
                marker=dict(symbol="x", color="#d62728", size=7), customdata=f.fraction,
                hovertemplate="λ = %{x:.4g}, fraction %{customdata}: model run failed<extra></extra>",
            ), row=row, col=col)
        a = it.loc[i] if i in it.index else None
        if a is not None and not np.isnan(a.accepted_test):
            t = lam[(lam.iteration == i) & (lam.test == a.accepted_test)].iloc[0]
            fig.add_trace(go.Scatter(
                x=[t["lambda"]], y=[t.phi_ratio], mode="markers", name="accepted", showlegend=k == 0,
                marker=dict(symbol="circle-open", color="black", size=16, line=dict(width=2)), hoverinfo="skip",
            ), row=row, col=col)
        fig.add_hline(y=1, line=dict(color="gray", dash="dot"), row=row, col=col)
    fig.update_xaxes(type="log", title_text="λ")
    fig.update_yaxes(title_text="φ / starting φ", col=1)
    fig.update_layout(height=330 * nrow + 60, legend=dict(orientation="h", y=1.04, yanchor="bottom", x=1, xanchor="right"))
    return fig


def runs_figure(r):
    it = r["iterations"].dropna(subset=["calls"])
    if not len(it):
        return message_figure("The record has a single iteration: no model-run counts.")
    fig = go.Figure()
    for col, label, color in (("jacobian_runs", "Jacobian runs", "#1f77b4"),
                              ("ok_lambda_runs", "Lambda tests", "#2ca02c"),
                              ("failed_lambda_runs", "Lambda tests that failed", "#d62728")):
        y = it.lambda_runs - it.failed_lambda_runs if col == "ok_lambda_runs" else it[col]
        fig.add_trace(go.Bar(x=it.index, y=y, name=label, marker_color=color,
                             hovertemplate="iteration %{x}: %{y} runs<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="stack", height=420, xaxis=dict(title="Iteration", dtick=1), yaxis_title="Model runs",
                      title=f"Model runs by iteration ({int(it.calls.sum())} in the complete iterations)",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig


def table_figure(r):
    it = r["iterations"].copy()
    ev = r["events"]
    def names(i, event):
        p = ev[(ev.iteration == i) & (ev.event == event)].parameter.tolist()
        return ", ".join(p[:6]) + (f" (+{len(p) - 6})" if len(p) > 6 else "")
    fmt = lambda v: "" if pd.isna(v) else f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)
    df = pd.DataFrame({
        "iteration": it.index,
        "model calls so far": it.model_calls.map(fmt),
        "measurement φ": it.phi_measurement.map(fmt),
        "target φ": it.phi_target.map(fmt),
        "regularisation φ": it.phi_regularization.map(fmt),
        "weight factor": it.weight_factor.map(fmt),
        "lambda tests (failed)": [f"{a} ({b})" for a, b in zip(it.lambda_runs, it.failed_lambda_runs)],
        "accepted λ (fraction)": [f"{fmt(a)} ({fmt(b)})" if pd.notna(a) else "" for a, b in zip(it.accepted_lambda, it.accepted_fraction)],
        "max factor change": [f"{fmt(a)} {b}" if pd.notna(a) else "" for a, b in
                              zip(it.get("max_factor_change", pd.Series(np.nan, it.index)),
                                  it.get("max_factor_parameter", pd.Series("", it.index)))],
        "no effect": [names(i, "no effect") for i in it.index],
        "frozen": [names(i, "frozen") for i in it.index],
    })
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{c}</b>" for c in df.columns], fill_color="royalblue", font=dict(color="white"), align="left"),
        cells=dict(values=[df[c].astype(str).tolist() for c in df.columns], align="left"),
        columnwidth=[70, 80, 95, 85, 100, 80, 90, 120, 130, 170, 170],
    ))
    fig.update_layout(height=min(700, 120 + 45 * len(df)), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def summary(r):
    it, lam, s = r["iterations"], r["lambdas"], r["settings"]
    last = int(it.index.max()) if len(it) else 0
    status = (f"{r['name']}: optimisation complete after {last} iterations." if r["complete"] else
              f"{r['name']}: the record stops during iteration {last} (run still going, stopped, or failed).")
    failed = int(lam.failed.sum())
    parts = [status, f" {len(lam)} lambda tests, {failed} with a failed model run ({100 * failed / max(len(lam), 1):.0f}%)."]
    settings = []
    for key, label in (("noptmax", "NOPTMAX"), ("phiredstp", "PHIREDSTP"), ("fracphim", "FRACPHIM"), ("phimlim", "PHIMLIM"),
                       ("forgive_lambda_failures", "forgive failed lambda runs"),
                       ("forgive_jacobian_failures", "forgive failed Jacobian runs")):
        if key in s:
            try:
                value = f"{float(s[key]):g}"
            except ValueError:
                value = s[key]
            settings.append(f"{label} {value}")
    out = [dmc.Alert("".join(parts), color="blue", variant="light", mb=5)]
    if settings:
        out.append(dmc.Text("Settings: " + "; ".join(settings) + ".", size="sm", c="dimmed"))
    if s.get("forgive_jacobian_failures", "").lower().startswith("y"):
        out.append(dmc.Text("Failed Jacobian runs are forgiven: PEST records them as all-zero Jacobian columns "
                            "(see 'no effect' in the table).", size="sm", c="dimmed"))
    return out


@callback(
    Output(name+"summary", "children"),
    Output(name+"phi", "figure"),
    Output(name+"lambda", "figure"),
    Output(name+"runs", "figure"),
    Output(name+"table", "figure"),
    Input(name+"update", "n_clicks"),
    State("pest-data", "data"),
    prevent_initial_call=True,
    running=[(Output(name+"update", "loading"), True, False)],
)
def read_record(n_clicks, data):
    def only(message, color="yellow"):
        return [dmc.Alert(message, color=color, variant="light")] + [message_figure(message)] + [message_figure()] * 3
    if not n_clicks:
        return [no_update] * 5
    if not data:
        return only("No data loaded. Load data on the Upload Data page, then click 'Read Run Record'.")
    try:
        r = read_rec(data)
    except Exception as e:
        return only(f"Could not read the run record: {e}", "red")
    if r is None:
        return only("No .rec file in the zip file. Add case.rec to the zip to see the run record.")
    if not len(r["iterations"]):
        return only(f"{r['name']} has no optimisation iterations yet.")
    return [summary(r), phi_figure(r), lambda_figure(r), runs_figure(r), table_figure(r)]
