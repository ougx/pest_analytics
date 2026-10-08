from dash import html, dcc, Input, Output, State, callback, no_update, ctx
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from utils import parse_zip
from pst import get_obs_pull, get_par_tension, get_jco_rei
from layout import update_groups, update_weights, update_jco_iter, about

name = __name__.lstrip("module.") + "-"
fullname = "Observation Conflict"

UP   = "#d62728"   # observation wants the parameter to increase
DOWN = "#1f77b4"   # observation wants the parameter to decrease
UP_LINK   = "rgba(214, 39, 40, 0.5)"
DOWN_LINK = "rgba(31, 119, 180, 0.5)"

md = dcc.Markdown('''

    This page finds observations (targets) that are in conflict, i.e. targets that pull the same parameter in opposite directions.
    Two targets that are both sensitive to a parameter are not necessarily in conflict; they are only in conflict if their residuals
    ask for opposite changes. The pull of observation $i$ on parameter $j$ is its contribution to the negative gradient of the objective function:
    $$
                  P_{ij} = -\\frac{\\partial \\phi_i}{\\partial p_j} = 2 w_i^2 r_i J_{ij},
    $$
    where $w_i$ is the weight, $r_i$ = measured - modelled is the residual, and $J$ is the Jacobian matrix. $P_{ij} > 0$ (red) means target $i$ wants
    parameter $j$ to increase; $P_{ij} < 0$ (blue) means it wants $j$ to decrease.
    For log-transformed parameters the Jacobian is with respect to $\\log_{10}p$, so pulls are comparable between parameters.
    `.jco.N` is computed at the parameters of iteration N-1, so it is paired with the residuals in `.rei.(N-1)` (`.hp` for N = 1).

    Pulls are summed over the selected observation grouping, and the conflict on each parameter is measured by its tension:
    $$
                  T_j = \\sum_k |P_{kj}| - \\left|\\sum_k P_{kj}\\right| = 2\\min(\\text{pull up}, |\\text{pull down}|),
    $$
    i.e. the pull that cancels out. Grouping by observation group shows conflicts *between* groups (e.g. heads vs. flows); conflicts *within* a group cancel
    before the tension is computed. Group by `obsname` to see conflicts between individual targets.

    1. The bar chart ranks parameters by tension. The net pull (diamond) is what is left; at convergence it is balanced by regularization and parameter bounds.
    2. The heatmap shows the share of the pull on each parameter that comes from each group. A column with both strong red and strong blue cells is a conflict.
    3. The Sankey diagram drills into the selected parameters (click a bar or a heatmap cell to add one) and shows which targets pull each way.
    4. Target Focus starts from a chosen observation or group $F$ instead (click a heatmap cell to focus on its row). It shows what $F$ wants,
       ranked by how strongly the other targets pull the opposite way, and which targets oppose it. Target $k$ opposes $F$ by
       $$
                  \\text{oppose}_k = \\frac{1}{\\sum_j |P_{Fj}|}\\sum_{j:\\,P_{Fj}P_{kj}<0} \\min(|P_{Fj}|, |P_{kj}|),
       $$
       the share of the focus pull it opposes (agreement is the same sum over parameters pulled in the same direction).
       A focus opposed by many unrelated targets points to the focus itself (questionable data, too much weight, or a local feature the model
       cannot represent); a focus opposed mainly by one site or group points to a structural conflict between the two.

    Persistent tension after convergence often indicates model structural error (e.g. a missing zone or a wrong boundary condition) or
    inconsistent data/weights, rather than a poorly estimated parameter. The analysis is linear and only valid near the parameters where
    the Jacobian was computed; a conflict on one parameter may be resolved by another parameter.
    ''',
    mathjax=True)


layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Jacobian Iteration:", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        dmc.Select(id=name+"grp", label="Obs Grouped by:", data=["obsgrp"], value="obsgrp", style={"width": "150px"}),
        dmc.Select(id=name+"plevel", label="Parameter Level:", data=["Parameters", "Par Groups"], value="Parameters", style={"width": "150px"}),
        dmc.Select(id=name+"npar", label="Number of Parameters:", data=["10", "20", "30", "50", "100"], value="30", style={"width": "150px"}),
        dmc.Select(id=name+"ngrp", label="Number of Obs Groups:", data=["10", "20", "30", "50", "100"], value="30", style={"width": "150px"}),
        dmc.Button("Plot Conflicts", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    dcc.Loading(dcc.Graph(id=name+"bar", figure=go.Figure(layout=dict(height=450, )))),
    dcc.Loading(dcc.Graph(id=name+"heat", figure=go.Figure(layout=dict(height=450, )))),
    html.H4("Drill-down", style={"margin-top": "20px"}),
    dmc.Group([
        dmc.MultiSelect(id=name+"sankey-par", label="Parameters (click a bar or heatmap cell to add):",
                        data=[], value=[], searchable=True, clearable=True, style={"width": "600px"}),
        dmc.Select(id=name+"sankey-grp", label="Targets:", data=["obsname"], value="obsname", style={"width": "150px"}),
        dmc.Select(id=name+"nlinks", label="Targets per Parameter:", data=["5", "10", "20", "40"], value="10", style={"width": "150px"}),
        dmc.Switch(id=name+"normalize", label="Normalize per parameter", checked=True, style={"width": "200px"}),
    ], mt=10, align="flex-end"),
    dcc.Loading(dcc.Graph(id=name+"sankey", figure=go.Figure(layout=dict(height=650, )))),
    html.H4("Target Focus", style={"margin-top": "20px"}),
    dmc.Group([
        dmc.Select(id=name+"focus-type", label="Focus by:", data=["obsname"], value="obsname", style={"width": "150px"}),
        dmc.MultiSelect(id=name+"focus", label="Focus target(s) (or click a heatmap cell):",
                        data=[], value=[], searchable=True, clearable=True, limit=200, style={"width": "500px"}),
        dmc.Select(id=name+"vs-grp", label="Compare against:", data=["obsname"], value="obsname", style={"width": "150px"}),
        dmc.Select(id=name+"fnpar", label="Number of Parameters:", data=["10", "20", "30", "50"], value="20", style={"width": "150px"}),
        dmc.Select(id=name+"fnopp", label="Number of Targets:", data=["10", "20", "30", "50"], value="20", style={"width": "150px"}),
    ], mt=10, align="flex-end"),
    dcc.Loading(dcc.Graph(id=name+"fbar", figure=go.Figure(layout=dict(height=550, )))),
    dcc.Loading(dcc.Graph(id=name+"fopp", figure=go.Figure(layout=dict(height=400, )))),
    dcc.Loading(dcc.Graph(id=name+"fsankey", figure=go.Figure(layout=dict(height=650, )), style={"margin-bottom":"90px"})),
])


@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Input( "pest-data", "data"),
)
def upload_pest(data):
    return update_jco_iter(data, need_rei=True)


@callback(
    Output(name+"grp"       , "data"),
    Output(name+"grp"       , "value"),
    Output(name+"sankey-grp", "data"),
    Output(name+"focus-type", "data"),
    Output(name+"vs-grp"    , "data"),
    Output(name+"weight"    , "data"),
    Output(name+"weight"    , "value"),
    Input( "pest-data", "data"),
)
def upload_grp_weight(data):
    grps, grp = update_groups(data)
    if grps is not no_update:
        grps = ["obsname"] + grps
    return [grps, grp, grps, grps, grps] + update_weights(data, "weight")


def load_pull(data, iiter, weight, plevel):
    """
    Load PEST data and return (pull per observation, obs table, message).
    Pull is None if the data cannot be analyzed; message then explains why.
    """
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, True, True)
    if obs is None:
        return None, None, "No .pst file found in zip file."
    if not jcos:
        return None, None, "No .jco file found in zip file."
    if not reis:
        return None, None, "No .rei or .hp file found in zip file."
    n = int(iiter)
    if n not in jcos:
        return None, None, f"No .jco file found for iteration {n}."
    rei = get_jco_rei(reis, n)
    if rei is None:
        return None, None, f"No .rei file found for iteration {n-1} or {n}."

    pull = get_obs_pull(obs, jcos[n], rei, weight_col=weight)
    if plevel == "Par Groups":
        pargrp = par.set_index("parname").pargrp.reindex(pull.columns).fillna("other")
        pull = pull.T.groupby(pargrp.values).sum().T
    return pull, obs, ""


def group_pull(pull, obs, grp):
    """Sum observation pulls by an observation grouping column ("obsname": no grouping)."""
    if grp == "obsname":
        return pull
    key = obs.set_index("obsname")[grp].reindex(pull.index).astype(str)
    return pull.groupby(key.values).sum()


def get_group_pull(data, iiter, weight, grp, plevel):
    """
    Load PEST data and return (pull summed by observation grouping, message).
    Pull is None if the data cannot be analyzed; message then explains why.
    """
    pull, obs, msg = load_pull(data, iiter, weight, plevel)
    if pull is None:
        return None, msg
    return group_pull(pull, obs, grp), ""


@callback(
    Output(name+"bar"       , "figure"),
    Output(name+"heat"      , "figure"),
    Output(name+"sankey-par", "data"),
    Output(name+"sankey-par", "value"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"weight", "value"),
    State( name+"grp", "value"),
    State( name+"plevel", "value"),
    State( name+"npar", "value"),
    State( name+"ngrp", "value"),
)
def plot(clicks, data, iiter, weight, grp, plevel, npar, ngrp):
    if clicks is None or clicks == 0:
        msg = px.scatter(title="Upload data and click 'Plot Conflicts' to see plot")
        return msg, go.Figure(), no_update, no_update
    if data is None or data == {}:
        msg = px.scatter(title="No data uploaded. Please upload data and click 'Plot Conflicts' to see plot")
        return msg, go.Figure(), no_update, no_update
    G, msg = get_group_pull(data, iiter, weight, grp, plevel)
    if G is None:
        return px.scatter(title=msg), go.Figure(), no_update, no_update

    T = get_par_tension(G).sort_values("tension", ascending=False)
    top = T.iloc[:int(npar)]
    pars = top.index.tolist()
    return tension_figure(top, grp), share_figure(G[pars], int(ngrp), grp), T.index.tolist(), pars[:5]


def tension_figure(top, grp):
    customdata = np.c_[top.tension, top.ratio, top.net]
    hover = "%{x}<br>%{fullData.name}: %{y:.4g}<br>tension: %{customdata[0]:.4g}<br>conflict ratio: %{customdata[1]:.2f}<extra></extra>"
    fig = go.Figure([
        go.Bar(x=top.index, y=top.up  , name="Pull up (wants increase)"  , marker_color=UP  , customdata=customdata, hovertemplate=hover),
        go.Bar(x=top.index, y=top.down, name="Pull down (wants decrease)", marker_color=DOWN, customdata=customdata, hovertemplate=hover),
        go.Scatter(x=top.index, y=top.net, name="Net pull", mode="markers",
                   marker=dict(symbol="diamond", size=10, color="black"),
                   hovertemplate="%{x}<br>net pull: %{y:.4g}<extra></extra>"),
    ])
    fig.update_layout(
        barmode="relative",
        height=450,
        title=f"Parameters ranked by tension between targets grouped by {grp}",
        yaxis_title="Pull (-dφ/dp)",
        xaxis=dict(tickangle=90),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    return fig


def share_figure(G, ngrp, grp):
    rows = G.abs().sum(axis=1).sort_values(ascending=False).index[:ngrp]
    share = G / G.abs().sum().replace(0, np.nan)
    share = share.loc[rows]
    zmax = np.nanmax(np.abs(share.values)) if share.size else 1
    fig = go.Figure(go.Heatmap(
        z=share.values,
        x=share.columns,
        y=share.index.astype(str),
        customdata=G.loc[rows].values,
        colorscale="RdBu_r",
        zmin=-zmax, zmid=0, zmax=zmax,
        hovertemplate="%{y} → %{x}<br>share: %{z:.1%}<br>pull: %{customdata:.4g}<extra></extra>",
        colorbar=dict(title=dict(text="Share of pull on parameter", side="right"), tickformat=".0%"),
    ))
    fig.update_layout(
        height=max(450, 22 * len(rows) + 200),
        title=f"Share of pull on each parameter by {grp} (red: wants increase, blue: wants decrease)",
        xaxis=dict(tickangle=90),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


@callback(
    Output(name+"sankey-par", "value", allow_duplicate=True),
    Input( name+"bar" , "clickData"),
    Input( name+"heat", "clickData"),
    State( name+"sankey-par", "value"),
    prevent_initial_call=True,
)
def add_par(barclick, heatclick, selected):
    click = barclick if ctx.triggered_id == name+"bar" else heatclick
    if not click:
        return no_update
    p = click["points"][0]["x"]
    selected = selected or []
    return no_update if p in selected else selected + [p]


@callback(
    Output(name+"sankey", "figure"),
    Input( name+"sankey-par", "value"),
    Input( name+"sankey-grp", "value"),
    Input( name+"nlinks", "value"),
    Input( name+"normalize", "checked"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"weight", "value"),
    State( name+"plevel", "value"),
)
def plot_sankey(selected, grp, nlinks, normalize, data, iiter, weight, plevel):
    if not selected:
        return px.scatter(title="Select parameter(s) above, or click a bar or heatmap cell, to see which targets pull them")
    if data is None or data == {}:
        return px.scatter(title="No data uploaded.")
    G, msg = get_group_pull(data, iiter, weight, grp, plevel)
    if G is None:
        return px.scatter(title=msg)
    selected = [p for p in selected if p in G.columns]
    if not selected:
        return px.scatter(title="Selected parameter(s) not found. Click 'Plot Conflicts' again.")
    return sankey_figure(G[selected], int(nlinks), normalize, grp)


def sankey_figure(G, nlinks, normalize, grp):
    T = get_par_tension(G)
    links = []  # (source, parameter, pull, share of the parameter's total |pull|)
    for p in G.columns:
        col = G[p]
        col = col[col != 0]
        total = col.abs().sum()
        top = col.abs().nlargest(nlinks).index
        links += [(str(k), p, v, v / total) for k, v in col[top].items()]
        rest = col.drop(top)
        for v in (rest[rest > 0].sum(), rest[rest < 0].sum()):
            if v != 0:
                links.append((f"other {grp}", p, v, v / total))

    sources = list(dict.fromkeys(l[0] for l in links))
    nodes = sources + list(G.columns)
    idx = {n: i for i, n in enumerate(nodes)}
    total = G.abs().sum().replace(0, np.nan)
    par_labels = [f"{p}  ▲{T.up[p]/total[p]:.0%} ▼{-T.down[p]/total[p]:.0%}" for p in G.columns]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=sources + par_labels,
            color=["lightgray"] * len(sources) + ["dimgray"] * len(G.columns),
            pad=8,
            thickness=15,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=[idx[l[0]] for l in links],
            target=[idx[l[1]] for l in links],
            value=[abs(l[3] if normalize else l[2]) for l in links],
            color=[UP_LINK if l[2] > 0 else DOWN_LINK for l in links],
            customdata=[[l[1], "increase" if l[2] > 0 else "decrease", abs(l[3]), l[2]] for l in links],
            hovertemplate="%{source.label} → %{customdata[0]}<br>wants %{customdata[1]}"
                          "<br>share of pull on %{customdata[0]}: %{customdata[2]:.1%}"
                          "<br>pull: %{customdata[3]:.4g}<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=max(650, 18 * len(sources)),
        title="Targets pulling the selected parameter(s): red wants increase, blue wants decrease (▲/▼: share of pull up/down)",
    )
    return fig


# ---------------------------------------------------------------------------
# Target focus: what does a chosen observation/group want, and who opposes it
# ---------------------------------------------------------------------------

OPPOSE = "#ff7f0e"
AGREE  = "#2ca02c"


@callback(
    Output(name+"focus", "data"),
    Input( name+"focus-type", "value"),
    Input( "pest-data", "data"),
    State( name+"weight", "value"),
)
def update_focus_options(ftype, data, weight):
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, False)
    if obs is None:
        return no_update
    weight = weight if weight in obs.columns else "weight"
    obs = obs[(obs[weight] > 0) & ~obs.obsgrp.str.startswith("regul")]
    if ftype == "obsname":
        return [{"group": str(gp), "items": df.obsname.tolist()} for gp, df in obs.groupby("obsgrp")]
    return sorted(obs[ftype].dropna().astype(str).unique().tolist())


@callback(
    Output(name+"focus-type", "value"),
    Output(name+"focus", "value"),
    Input( name+"heat", "clickData"),
    State( name+"grp", "value"),
    prevent_initial_call=True,
)
def focus_from_heatmap(click, grp):
    if not click:
        return no_update, no_update
    return grp, [str(click["points"][0]["y"])]


@callback(
    Output(name+"fbar"   , "figure"),
    Output(name+"fopp"   , "figure"),
    Output(name+"fsankey", "figure"),
    Input( name+"focus-type", "value"),
    Input( name+"focus", "value"),
    Input( name+"vs-grp", "value"),
    Input( name+"fnpar", "value"),
    Input( name+"fnopp", "value"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"weight", "value"),
    State( name+"plevel", "value"),
)
def plot_focus(ftype, focus, vsgrp, npar, nopp, data, iiter, weight, plevel):
    if not focus:
        msg = px.scatter(title="Select the target(s) to focus on, or click a heatmap cell")
        return msg, go.Figure(), go.Figure()
    if data is None or data == {}:
        msg = px.scatter(title="No data uploaded.")
        return msg, go.Figure(), go.Figure()
    pull, obs, msg = load_pull(data, iiter, weight, plevel)
    if pull is None:
        return px.scatter(title=msg), go.Figure(), go.Figure()

    if ftype == "obsname":
        key = pull.index.to_series()
    else:
        key = obs.set_index("obsname")[ftype].reindex(pull.index).astype(str)
    mask = key.isin(focus).values
    F = pull[mask].sum()
    if not mask.any() or (F == 0).all():
        msg = px.scatter(title="The selected target(s) have no pull (zero weight or zero residual).")
        return msg, go.Figure(), go.Figure()
    others = pull[~mask]
    O = group_pull(others, obs, vsgrp)
    label = ", ".join(focus) if len(focus) <= 3 else f"{len(focus)} {ftype} selected"

    stats = get_focus_stats(F, others.sum())
    pars = stats[stats.focus != 0].iloc[:int(npar)].index.tolist()
    return (
        focus_bar_figure(stats.loc[pars], label),
        opponent_figure(get_opponents(F, O), int(nopp), label, vsgrp),
        focus_sankey_figure(F[pars[:15]], O[pars[:15]], label, vsgrp),
    )


def get_focus_stats(F, R):
    """
    Per parameter: focus pull F, pull of all other targets R, and the opposed pull
    min(|F|, |R|) where they point in opposite directions. Sorted by opposed pull, then |F|.
    """
    opposed = np.where(np.sign(F) * np.sign(R) < 0, np.minimum(F.abs(), R.abs()), 0)
    stats = pd.DataFrame(dict(focus=F, rest=R, opposed=opposed, absF=F.abs()))
    stats["share"] = stats.absF / stats.absF.sum()
    return stats.sort_values(["opposed", "absF"], ascending=False)


def get_opponents(F, O):
    """
    For each other target/group k: the share of the focus pull it opposes and agrees with,
    sum_j min(|F_j|, |O_kj|) over parameters with opposite (or same) direction, / sum_j |F_j|,
    and the parameter carrying most of the opposition.
    """
    m = np.minimum(O.abs().values, F.abs().values[None, :])
    s = np.sign(O.values) * np.sign(F.values)[None, :]
    opp = np.where(s < 0, m, 0)
    total = F.abs().sum()
    res = pd.DataFrame(dict(
        oppose=opp.sum(axis=1) / total,
        agree=np.where(s > 0, m, 0).sum(axis=1) / total,
        main_par=O.columns[opp.argmax(axis=1)],
    ), index=O.index.astype(str))
    res.loc[res.oppose == 0, "main_par"] = ""
    return res.sort_values("oppose", ascending=False)


def focus_bar_figure(stats, label):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"Pull of focus: {label}", "Pull of all other targets"))
    opposed = np.where(stats.opposed > 0, "yes", "no")
    fig.add_trace(go.Bar(
        x=stats.index, y=stats.focus, name="Focus",
        marker_color=np.where(stats.focus > 0, UP, DOWN),
        customdata=np.c_[stats.share, opposed],
        hovertemplate="%{x}<br>focus pull: %{y:.4g}<br>share of focus pull: %{customdata[0]:.1%}"
                      "<br>opposed by others: %{customdata[1]}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=stats.index, y=stats.rest, name="All other targets",
        marker_color=np.where(stats.rest > 0, UP, DOWN), opacity=0.6,
        customdata=opposed,
        hovertemplate="%{x}<br>pull of others: %{y:.4g}<br>opposes focus: %{customdata}<extra></extra>",
    ), row=2, col=1)
    fig.update_layout(
        height=550, showlegend=False,
        title="What the focus wants (red: increase, blue: decrease); parameters the others pull the opposite way come first",
    )
    fig.update_xaxes(tickangle=90, row=2, col=1)
    fig.update_yaxes(title_text="Pull (-dφ/dp)")
    return fig


def opponent_figure(opp, nopp, label, vsgrp):
    opp = opp[(opp.oppose > 0) | (opp.agree > 0)].iloc[:nopp]
    hover = "%{y}<br>%{fullData.name}: %{x:.1%} of focus pull<br>main conflict on: %{customdata}<extra></extra>"
    fig = go.Figure([
        go.Bar(y=opp.index, x=opp.oppose, name="Opposes focus", orientation="h",
               marker_color=OPPOSE, customdata=opp.main_par, hovertemplate=hover),
        go.Bar(y=opp.index, x=opp.agree, name="Agrees with focus", orientation="h",
               marker_color=AGREE, customdata=opp.main_par, hovertemplate=hover),
    ])
    fig.update_layout(
        barmode="group",
        height=max(400, 28 * len(opp) + 150),
        title=f"Which {vsgrp} oppose {label} (share of the focus pull they oppose / agree with)",
        xaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def focus_sankey_figure(F, O, label, vsgrp, ntargets=5):
    """
    Focus -> parameters -> other targets. Each parameter's ribbon is split among the
    other targets by their share of the pull on that parameter.
    """
    F = F[F != 0]
    left = F.abs() / F.abs().sum()
    links = []  # (source key, target key, width, color, hover text)
    for p, f in F.items():
        want = "increase" if f > 0 else "decrease"
        links.append((("focus",), ("par", p), left[p], UP_LINK if f > 0 else DOWN_LINK,
                      f"{label} → {p}<br>focus wants {want}<br>share of focus pull: {left[p]:.1%}<br>pull: {f:.4g}"))
        col = O[p][O[p] != 0]
        if col.empty:
            continue
        total = col.abs().sum()
        top = col.abs().nlargest(ntargets).index
        items = [(str(k), v) for k, v in col[top].items()]
        rest = col.drop(top)
        items += [(f"other {vsgrp}", v) for v in (rest[rest > 0].sum(), rest[rest < 0].sum()) if v != 0]
        for k, v in items:
            want = "increase" if v > 0 else "decrease"
            rel = "agrees with" if v * f > 0 else "opposes"
            links.append((("par", p), ("tgt", k), left[p] * abs(v) / total, UP_LINK if v > 0 else DOWN_LINK,
                          f"{p} → {k}<br>wants {want} ({rel} focus)"
                          f"<br>share of others' pull on {p}: {abs(v) / total:.1%}<br>pull: {v:.4g}"))

    keys = list(dict.fromkeys([l[0] for l in links] + [l[1] for l in links]))
    idx = {k: i for i, k in enumerate(keys)}
    labels = [f"FOCUS: {label}" if k[0] == "focus" else k[1] for k in keys]
    colors = ["black" if k[0] == "focus" else "dimgray" if k[0] == "par" else "lightgray" for k in keys]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=8, thickness=15, hovertemplate="%{label}<extra></extra>"),
        link=dict(
            source=[idx[l[0]] for l in links],
            target=[idx[l[1]] for l in links],
            value=[l[2] for l in links],
            color=[l[3] for l in links],
            customdata=[l[4] for l in links],
            hovertemplate="%{customdata}<extra></extra>",
        ),
    ))
    fig.update_layout(
        height=max(650, 16 * len(keys)),
        title="Focus → parameters → other targets pulling the same parameters (red: wants increase, blue: wants decrease)",
    )
    return fig
