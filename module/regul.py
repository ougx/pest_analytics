from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import parse_zip
from plotting import message_figure
from layout import about, graph
from rec import read_rec
import compare

name = __name__.removeprefix("module.") + "-"
fullname = "Regularisation"

md = dcc.Markdown('''

    How the regularisation (prior information, Tikhonov constraints) and the fit to the measurements were traded off.

    - **Trade-off**: every parameter upgrade PEST tried (the lambda tests in the run record `case.rec`), as measurement
      $\\phi$ against regularisation $\\phi$, one colour per iteration; the circled ones were accepted. PEST reports the
      regularisation $\\phi$ with the `.pst` weights adjusted by IREGADJ, without the regularisation weight factor, so
      points within an iteration are comparable but iterations only roughly. The dashed line is the target measurement
      $\\phi$ of the last iteration (FRACPHIM / PHIMLIM).
    - **Regularisation misfit by group**: $\\sum (w r)^2$ of the prior-information equations (and regul* observation
      groups) in each `.rei` file, with the **`.pst` weights**, so that iterations compare. (The weights in the `.rei`
      files include PEST_HP's weight factor and IREGADJ adjustments, which change every iteration.)
    - **Equations**: for one iteration, the prior-information equations with the largest $(w r)^2$: which constraints
      the calibration pulled hardest against, e.g. a preferred value or a homogeneity condition between pilot points.

    The equations and their `.pst` weights are read with makepst; without it the `.rei` weights are used.

    ''', mathjax=True)

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Residual Iteration:", description="for the equations (.rei.N)",
                   data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"top", label="Equations shown:", data=["20", "50", "All"], value="20", style={"width": "150px"}),
        dmc.Button("Plot Regularisation", id=name+"update", color="blue"),
    ], mt=10, align="flex-end"),
    html.Div(id=name+"summary", style={"margin-top": "20px"}),
    graph(name+"tradeoff", message_figure("Load data, then click 'Plot Regularisation'."), "Regularisation Trade-off"),
    graph(name+"groups", message_figure(), "Regularisation Misfit by Group"),
    graph(name+"equations", message_figure(), "Regularisation Equations", style={"margin-bottom": "90px"}),
])


@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Input("pest-data", "data"),
)
def fill_iterations(data):
    if not data:
        return no_update, no_update
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True)
    if not reis:
        return no_update, no_update
    iters = [str(i) for i in sorted(reis) if (~reis[i].obsname.isin(obs.obsname)).any() or
             reis[i].obsgrp.astype(str).str.startswith("regul").any()]
    return (iters, iters[-1]) if iters else (no_update, no_update)


def prior_weights(data):
    """(.pst weights and equations of the prior information by lower-case name, or None without makepst)"""
    try:
        pst = compare.run_pst(data)
    except ImportError:
        return None
    if pst is None or not len(pst.prior):
        return None
    p = pst.prior.assign(name=pst.prior.PINME.str.lower()).set_index("name")
    return p[["EQ", "WEIGHT", "OBGNME"]].rename(columns={"EQ": "equation", "WEIGHT": "weight_pst", "OBGNME": "group"})


def regularisation_rows(rei, obs, prior):
    """
    Regularisation rows of a residual file: the prior-information equations (rows that are not observations)
    and the observations in regul* groups, with the residual, the .rei weight, the .pst weight (the observation
    weight for regul* observations; the .rei weight if the .pst weights are unknown) and (w_pst r)^2.
    """
    is_prior = ~rei.obsname.isin(obs.obsname)
    is_regul_obs = rei.obsname.isin(obs.obsname) & rei.obsgrp.astype(str).str.startswith("regul")
    df = rei[is_prior | is_regul_obs][["obsname", "obsgrp", "residual", "weight"]].rename(
        columns={"obsname": "name", "obsgrp": "group", "weight": "weight_rei"}).set_index("name")
    df["equation"] = ""
    df["weight_pst"] = df.weight_rei
    if prior is not None:
        known = df.index.intersection(prior.index)
        df.loc[known, "weight_pst"] = prior.loc[known, "weight_pst"].astype(float)
        df.loc[known, "equation"] = prior.loc[known, "equation"]
    w = obs.set_index("obsname").weight
    regul_obs = df.index.intersection(w.index)
    df.loc[regul_obs, "weight_pst"] = w.loc[regul_obs]
    df["phi"] = (df.weight_pst * df.residual) ** 2
    return df


def tradeoff_figure(r):
    lam, it = r["lambdas"], r["iterations"]
    ok = lam[~lam.failed & lam.meas.notna() & lam.regul.notna()]
    if not len(ok):
        return message_figure("The run record has no lambda tests with both measurement and regularisation φ.")
    fig = go.Figure()
    colors = px.colors.sample_colorscale("Viridis", max(len(it), 2))
    color = dict(zip(it.index, colors))
    for i, d in ok.groupby("iteration"):
        fig.add_trace(go.Scatter(
            x=d.meas, y=d.regul, mode="markers", name=f"Iteration {i}", marker=dict(color=color.get(i), size=6, opacity=0.6),
            customdata=np.c_[d["lambda"], d.fraction],
            hovertemplate="λ %{customdata[0]:.4g}, fraction %{customdata[1]}<br>measurement φ %{x:.4g}<br>"
                          "regularisation φ %{y:.4g}<extra>%{fullData.name}</extra>",
        ))
    acc = it.dropna(subset=["accepted_test"])
    pts = [ok[(ok.iteration == i) & (ok.test == t)] for i, t in acc.accepted_test.items()]
    pts = pd.concat(pts) if pts else ok.iloc[:0]
    if len(pts):
        fig.add_trace(go.Scatter(
            x=pts.meas, y=pts.regul, mode="lines+markers+text", name="Accepted", text=[str(i) for i in pts.iteration],
            textposition="top center", line=dict(color="black", width=1),
            marker=dict(symbol="circle-open", size=14, color="black", line=dict(width=2)), hoverinfo="skip",
        ))
    target = it.phi_target.dropna()
    if len(target):
        fig.add_vline(x=target.iloc[-1], line=dict(color="#d62728", dash="dash"),
                      annotation_text=f"target measurement φ {target.iloc[-1]:.4g}", annotation_position="top left")
    fig.update_layout(height=520, title="Measurement vs. regularisation φ of every lambda test (run record)",
                      xaxis_title="Measurement φ", yaxis=dict(title="Regularisation φ (as reported by PEST)", type="log"),
                      legend=dict(orientation="h", y=-0.15, yanchor="top"))
    return fig


def groups_figure(by_iter):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for k, g in enumerate(by_iter.columns):
        fig.add_trace(go.Bar(x=by_iter.index, y=by_iter[g], name=g, marker_color=colors[k % len(colors)],
                             hovertemplate="iteration %{x}<br>%{y:.4g}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="stack", height=420, title="Regularisation misfit Σ (w r)² by group, with the .pst weights",
                      xaxis=dict(title="Iteration (.rei.N)", dtick=1), yaxis_title="Σ (w r)²",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    return fig


def equations_figure(rows, iiter, top):
    d = rows.sort_values("phi", ascending=False)
    total = d.phi.sum()
    d = d if top is None else d.head(top)
    colors = px.colors.qualitative.Plotly
    groups = list(dict.fromkeys(rows.group))
    fig = go.Figure()
    for k, g in enumerate(groups):
        e = d[d.group == g]
        if not len(e):
            continue
        fig.add_trace(go.Bar(
            x=e.phi, y=e.index, orientation="h", name=g, marker_color=colors[k % len(colors)],
            customdata=np.c_[e.equation, e.residual, e.weight_pst, e.weight_rei, 100 * e.phi / total],
            hovertemplate="<b>%{y}</b>: %{customdata[0]}<br>residual %{customdata[1]:.4g}, .pst weight %{customdata[2]:.4g} "
                          "(.rei %{customdata[3]:.3g})<br>(w r)² = %{x:.4g}, %{customdata[4]:.1f}% of the group total"
                          "<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        barmode="overlay", height=22 * len(d) + 200,
        title=f"Prior-information equations with the largest (w r)², .rei.{iiter} (.pst weights; total {total:.4g})",
        xaxis=dict(title="(w r)²", side="top"),
        yaxis=dict(categoryorder="array", categoryarray=list(d.index), range=[len(d) - 0.5, -0.5]),
        legend=dict(orientation="h", y=-30 / (22 * len(d) + 200), yanchor="top"), margin=dict(t=110),
    )
    return fig


@callback(
    Output(name+"summary", "children"),
    Output(name+"tradeoff", "figure"),
    Output(name+"groups", "figure"),
    Output(name+"equations", "figure"),
    Input(name+"update", "n_clicks"),
    State("pest-data", "data"),
    State(name+"iter", "value"),
    State(name+"top", "value"),
    prevent_initial_call=True,
    running=[(Output(name+"update", "loading"), True, False)],
)
def plot(n_clicks, data, iiter, top="20"):
    def only(message, color="yellow"):
        return [dmc.Alert(message, color=color, variant="light")] + [message_figure(message)] + [message_figure()] * 2
    if not n_clicks:
        return [no_update] * 4
    if not data:
        return only("No data loaded. Load data on the Upload Data page, then click 'Plot Regularisation'.")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True)
    if obs is None or not reis:
        return only("No .pst or residual files in the zip file.", "red")
    prior = prior_weights(data)
    rows = {i: regularisation_rows(rei, obs, prior) for i, rei in sorted(reis.items())}
    rows = {i: r for i, r in rows.items() if len(r)}
    if not rows:
        return only("The residual files have no regularisation rows (prior information or regul* groups).")
    by_iter = pd.DataFrame({i: r.groupby("group").phi.sum() for i, r in rows.items()}).T.fillna(0)
    n = int(iiter) if iiter is not None and int(iiter) in rows else max(rows)

    try:
        record = read_rec(data)
    except Exception:
        record = None
    tradeoff = tradeoff_figure(record) if record is not None and len(record["iterations"]) else \
        message_figure("No run record (.rec) in the zip file: no trade-off plot.")
    text = (f"{len(rows[n])} regularisation equations; {len(prior) if prior is not None else 0} prior-information "
            f"equations in the .pst" + ("" if prior is not None else " (read without makepst: .rei weights used)") + ".")
    if record is not None and len(record["iterations"]):
        it = record["iterations"]
        last = it.index.max()
        text += (f" Run record: measurement φ {it.phi_measurement.iloc[0]:.4g} → {it.loc[last, 'phi_measurement']:.4g}, "
                 f"target {it.phi_target.dropna().iloc[-1]:.4g}; weight factor {it.weight_factor.iloc[0]:.3g} → "
                 f"{it.loc[last, 'weight_factor']:.3g}." if it.phi_target.notna().any() else "")
    return [dmc.Alert(text, color="blue", variant="light"), tradeoff, groups_figure(by_iter),
            equations_figure(rows[n], n, None if top == "All" else int(top))]
