"""
PEST Calibration Diagnostics — Sensitivity Page
================================================
Drop-in Dash page module.  Wire into your AppShell:

    from sensitivity import layout, register_callbacks
    register_callbacks(app)

Expects the dict from pest_reader.read_pest_run().
Runs standalone with mock data:

    python sensitivity.py

Key diagnostics
---------------
  • Composite Scaled Sensitivity (CSS) — overall parameter importance ranking
  • One-Percent Scaled Sensitivity (OSS) heatmap — which obs each param affects
  • Parameter dominance bar — fraction of total sensitivity budget per param
  • Sensitivity vs parameter value scatter — correlation between sensitivity and
    current value (helps detect scale-related issues)
  • Group-level sensitivity aggregation — which param groups drive which obs groups
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MOCK DATA
# ──────────────────────────────────────────────────────────────────────────────

def _mock_data() -> dict:
    rng = np.random.default_rng(7)

    param_groups = {
        "Hydraulic cond.": ["K_hk1", "K_hk2", "K_hk3", "K_hk4"],
        "Storage":         ["Sy1",   "Ss1",   "Sy2"],
        "Recharge":        ["Rch_z1","Rch_z2","Rch_z3"],
        "Boundary":        ["GHB_k", "RIV_k", "DRN_k"],
        "Transport":       ["alphaL","alphaT","Kd"],
    }
    all_params = [p for ps in param_groups.values() for p in ps]
    n_params   = len(all_params)

    obs_groups = ["Heads", "Flows", "Conc.", "Levels"]
    n_obs      = 40

    obs_names = [f"{g[:2].lower()}{i:03d}"
                 for g in obs_groups for i in range(n_obs // len(obs_groups))]

    # CSS — log-uniform spread
    css_raw = np.exp(rng.uniform(-2, 3, n_params))
    css_raw[:4]  *= 3.0    # K params most sensitive
    css_raw[7:10]*= 1.8    # recharge moderately sensitive
    css_raw[10:] *= 0.4    # transport least sensitive

    # OSS matrix (n_obs × n_params)
    oss = rng.normal(0, 1, (len(obs_names), n_params))
    # K params strongly affect heads & flows
    for i, p in enumerate(all_params[:4]):
        oss[:10, i] *= 4
        oss[10:20, i] *= 2
    # Recharge affects heads & levels
    for i, p in enumerate(all_params[7:10]):
        oss[:10, 6 + (i % 4)] *= 2.5
        oss[30:, 6 + (i % 4)] *= 1.5
    # Transport only affects Conc
    for i in range(3):
        oss[20:30, 10 + i] *= 5
        oss[:20, 10 + i] *= 0.1

    # Parameter current values
    param_values = {
        p: float(np.exp(rng.uniform(-1, 2))) for p in all_params
    }

    sensitivity_df = pd.DataFrame({
        "name":  all_params,
        "group": [g for g, ps in param_groups.items() for p in ps],
        "css":   css_raw,
        "oss_mean": np.abs(oss).mean(axis=0),
    })

    oss_df = pd.DataFrame(oss, index=obs_names, columns=all_params)

    # Tag each obs with its group
    obs_group_map = {n: g
                     for g in obs_groups
                     for n in obs_names if n[:2] == g[:2].lower()}

    parameters_df = pd.DataFrame([
        {"name": p, "group": g, "value": param_values[p]}
        for g, ps in param_groups.items() for p in ps
    ])

    return {
        "sensitivity":    sensitivity_df,
        "oss_matrix":     oss_df,
        "parameters":     parameters_df,
        "obs_groups":     obs_groups,
        "param_groups":   list(param_groups.keys()),
        "_obs_group_map": obs_group_map,
    }


PEST_DATA = _mock_data()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  THEME
# ──────────────────────────────────────────────────────────────────────────────

PARAM_GROUP_COLORS = ["#4a90d9", "#2f7a5e", "#c0774a", "#9b7ec8", "#e8a838",
                      "#d94a4a", "#8ab3cc", "#5bab8a"]
OBS_GROUP_COLORS   = ["#4a90d9", "#2f7a5e", "#c0774a", "#e8a838"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=11, color="#555"),
    margin=dict(l=54, r=20, t=36, b=44),
    hoverlabel=dict(bgcolor="white", bordercolor="#ddd", font_size=11),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
    ),
)

def _base(extra_margin=None):
    """BASE_LAYOUT without margin, so callers can override it cleanly."""
    b = {k: v for k, v in BASE_LAYOUT.items() if k != "margin"}
    if extra_margin:
        b["margin"] = extra_margin
    return b

GRID = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=False, linecolor="rgba(0,0,0,0.12)", tickfont=dict(size=10))

CSS_THRESHOLDS = {"high": 1.0, "moderate": 0.1}   # relative to max CSS


def _pgroup_colors(param_groups: list[str]) -> dict[str, str]:
    return {g: PARAM_GROUP_COLORS[i % len(PARAM_GROUP_COLORS)]
            for i, g in enumerate(param_groups)}

def _ogroup_colors(obs_groups: list[str]) -> dict[str, str]:
    return {g: OBS_GROUP_COLORS[i % len(OBS_GROUP_COLORS)]
            for i, g in enumerate(obs_groups)}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _css_bar(sen: pd.DataFrame, param_groups: list[str],
             selected_pgroups: list[str], log_scale: bool = False) -> go.Figure:
    """Horizontal CSS bar chart, sorted descending, coloured by param group."""
    sub = sen[sen["group"].isin(selected_pgroups)].copy()
    sub = sub.sort_values("css", ascending=True)      # ascending → top = highest

    pgc = _pgroup_colors(param_groups)
    max_css = sub["css"].max()

    # Colour by sensitivity tier
    bar_colors = []
    for _, row in sub.iterrows():
        c = pgc.get(row["group"], "#aaa")
        ratio = row["css"] / (max_css + 1e-12)
        if ratio < CSS_THRESHOLDS["moderate"]:
            # desaturate low-sensitivity params
            bar_colors.append(_desaturate(c, 0.35))
        else:
            bar_colors.append(c)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["css"],
        y=sub["name"],
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        customdata=np.stack([sub["group"].values, sub["css"].values], axis=1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Group: %{customdata[0]}<br>"
            "CSS = %{x:.4f}<extra></extra>"
        ),
        showlegend=False,
    ))

    # Threshold annotations
    for thr_name, thr_frac in CSS_THRESHOLDS.items():
        thr_val = max_css * thr_frac
        if sub["css"].min() < thr_val < sub["css"].max():
            fig.add_vline(
                x=thr_val,
                line=dict(color="#d94a4a" if thr_name == "high" else "#e8a838",
                          width=1, dash="dot"),
                annotation_text=thr_name,
                annotation_position="top",
                annotation_font=dict(size=8),
            )

    fig.update_layout(
        **_base(extra_margin=dict(l=80, r=20, t=36, b=44)),
        xaxis={**GRID, "title": "CSS",
               "type": "log" if log_scale else "linear"},
        yaxis={**GRID, "title": None, "tickfont": dict(size=10)},
        height=max(300, 28 * len(sub) + 80),
    )
    return fig


def _css_dominance(sen: pd.DataFrame, param_groups: list[str],
                   selected_pgroups: list[str]) -> go.Figure:
    """Stacked 100% bar showing each group's share of total CSS budget."""
    sub = sen[sen["group"].isin(selected_pgroups)].copy()
    total = sub["css"].sum()
    pgc = _pgroup_colors(param_groups)

    group_sum = (sub.groupby("group")["css"]
                   .sum()
                   .reindex(selected_pgroups)
                   .dropna())

    fig = go.Figure()
    for grp in group_sum.index:
        pct = group_sum[grp] / total * 100
        fig.add_trace(go.Bar(
            name=grp,
            x=[pct],
            y=["Sensitivity budget"],
            orientation="h",
            marker_color=pgc.get(grp, "#aaa"),
            marker_line_width=0,
            text=f"{pct:.1f}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>{grp}</b><br>Share = {pct:.1f}%<extra></extra>",
        ))

    layout_no_margin = {k: v for k, v in BASE_LAYOUT.items() if k != "margin"}
    layout_no_margin["legend"] = dict(
        orientation="h", yanchor="top", y=-0.25,
        xanchor="left", x=0, font=dict(size=9),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        **layout_no_margin,
        barmode="stack",
        xaxis={**GRID, "title": "% of total CSS", "range": [0, 100]},
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=12, r=12, t=36, b=36),
        height=110,
    )
    return fig


def _oss_heatmap(oss: pd.DataFrame, obs_group_map: dict,
                 obs_groups: list[str], param_groups_order: list[str],
                 selected_pgroups: list[str], selected_ogroups: list[str],
                 metric: str = "mean_abs") -> go.Figure:
    """
    OSS aggregated to (obs_group × param) heatmap.
    metric: 'mean_abs' | 'max_abs' | 'sum_abs'
    """
    # Filter columns by selected param groups
    # (we need sensitivity df to map param→group — rebuild from oss columns)
    cols = oss.columns.tolist()

    # Aggregate OSS rows by obs group
    oss_tagged = oss.copy()
    oss_tagged["_obs_group"] = [obs_group_map.get(n, "Other") for n in oss.index]
    sel_obs = [g for g in obs_groups if g in selected_ogroups]

    agg_rows = []
    for og in sel_obs:
        grp_df = oss_tagged[oss_tagged["_obs_group"] == og].drop("_obs_group", axis=1)
        if metric == "mean_abs":
            agg_rows.append(grp_df.abs().mean())
        elif metric == "max_abs":
            agg_rows.append(grp_df.abs().max())
        else:
            agg_rows.append(grp_df.abs().sum())

    if not agg_rows:
        return go.Figure()

    agg = pd.DataFrame(agg_rows, index=sel_obs)

    # Filter columns to selected param groups requires the sensitivity df
    # We just show all params here — caller can pass pre-filtered oss

    z = agg.values
    vmax = np.abs(z).max()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=agg.columns.tolist(),
        y=agg.index.tolist(),
        colorscale="YlOrRd",
        zmin=0, zmax=vmax,
        colorbar=dict(
            title=dict(text=metric.replace("_", " ").title(),
                       font=dict(size=9)),
            thickness=12, len=0.8, tickfont=dict(size=8),
        ),
        hovertemplate=(
            "<b>%{x}</b> → <b>%{y}</b><br>"
            "OSS = %{z:.4f}<extra></extra>"
        ),
        xgap=1, ygap=1,
    ))

    fig.update_layout(
        **_base(extra_margin=dict(l=70, r=20, t=36, b=100)),
        xaxis=dict(tickfont=dict(size=9), tickangle=-40,
                   showgrid=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=10), showgrid=False, zeroline=False),
    )
    return fig


def _css_vs_value(sen: pd.DataFrame, params_df: pd.DataFrame,
                  param_groups: list[str], selected_pgroups: list[str]) -> go.Figure:
    """Scatter: current parameter value vs CSS, log-log axes."""
    sub = sen[sen["group"].isin(selected_pgroups)].merge(
        params_df[["name", "value"]], on="name", how="left", suffixes=(None, "_1")
    )
    pgc = _pgroup_colors(param_groups)
    fig = go.Figure()

    for grp in selected_pgroups:
        g_df = sub[sub["group"] == grp]
        if g_df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g_df["value"],
            y=g_df["css"],
            mode="markers+text",
            name=grp,
            text=g_df["name"],
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                color=pgc.get(grp, "#aaa"),
                size=9, opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Group: {grp}<br>"
                "Value = %{x:.4g}<br>"
                "CSS = %{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**GRID, "title": "Parameter value (log)", "type": "log"},
        yaxis={**GRID, "title": "CSS (log)",             "type": "log"},
    )
    return fig


def _param_ranking_table(sen: pd.DataFrame,
                         param_groups: list[str],
                         selected_pgroups: list[str]) -> html.Div:
    sub = (sen[sen["group"].isin(selected_pgroups)]
           .sort_values("css", ascending=False)
           .reset_index(drop=True))

    max_css = sub["css"].max()
    pgc = _pgroup_colors(param_groups)

    hstyle = {"background": "#f4f5f7", "padding": "6px 10px",
               "fontSize": "11px", "fontWeight": "600", "color": "#555",
               "borderBottom": "1px solid #e8e9ec", "textAlign": "right"}
    cstyle = {"padding": "5px 10px", "fontSize": "11px", "textAlign": "right",
               "fontFamily": "DM Mono, monospace", "borderBottom": "1px solid #f0f1f3"}
    lstyle = {**cstyle, "textAlign": "left", "fontFamily": "DM Sans, sans-serif"}

    def tier_badge(css_val):
        ratio = css_val / (max_css + 1e-12)
        if ratio >= CSS_THRESHOLDS["high"]:
            color, label = "#2f7a5e", "High"
        elif ratio >= CSS_THRESHOLDS["moderate"]:
            color, label = "#e8a838", "Moderate"
        else:
            color, label = "#d94a4a", "Low"
        return html.Span(label, style={
            "background": color, "color": "white",
            "padding": "1px 7px", "borderRadius": "10px",
            "fontSize": "10px", "fontWeight": "600",
        })

    def css_spark(css_val):
        pct = css_val / (max_css + 1e-12) * 100
        return html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"},
                        children=[
                            html.Div(style={
                                "width": f"{max(pct, 3):.0f}%",
                                "maxWidth": "120px",
                                "height": "6px",
                                "background": "#4a90d9",
                                "borderRadius": "3px",
                            }),
                            html.Span(f"{css_val:.4f}", style={
                                "fontFamily": "DM Mono, monospace",
                                "fontSize": "11px",
                            }),
                        ])

    cols = ["Rank", "Parameter", "Group", "CSS", "Sensitivity", "OSS mean"]
    header = html.Tr([html.Th(c, style=hstyle) for c in cols])

    rows = []
    for i, row in sub.iterrows():
        dot = html.Span(style={
            "display": "inline-block", "width": "8px", "height": "8px",
            "borderRadius": "50%",
            "background": pgc.get(row["group"], "#aaa"),
            "marginRight": "6px", "verticalAlign": "middle",
        })
        rows.append(html.Tr([
            html.Td(f"#{i+1}", style={**cstyle, "color": "#999"}),
            html.Td(row["name"], style=lstyle),
            html.Td([dot, row["group"]], style=lstyle),
            html.Td(css_spark(row["css"]), style={**cstyle, "textAlign": "left",
                                                   "minWidth": "180px"}),
            html.Td(tier_badge(row["css"]), style={**cstyle, "textAlign": "center"}),
            html.Td(f"{row['oss_mean']:.4f}", style=cstyle),
        ]))

    return html.Div(
        style={"overflowX": "auto"},
        children=[html.Table(
            style={"width": "100%", "borderCollapse": "collapse"},
            children=[html.Thead(header), html.Tbody(rows)],
        )],
    )


def _desaturate(hex_color: str, factor: float = 0.5) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    L = 0.299 * r + 0.587 * g + 0.114 * b
    r2 = r + factor * (L - r)
    g2 = g + factor * (L - g)
    b2 = b + factor * (L - b)
    return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SECTION HEADER & KPI HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _section(title, subtitle=None):
    return dmc.Stack(gap=0, mb="xs", children=[
        dmc.Text(title, fw=600, size="sm",
                 style={"color": "#333", "letterSpacing": "0.02em"}),
        dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else html.Span(),
    ])

def _kpi(label, value, unit="", icon=None, color="teal"):
    return dmc.Paper(
        shadow="xs", p="md", radius="md", withBorder=True,
        style={"borderColor": "#eee"},
        children=[dmc.Group(justify="space-between", align="flex-start", children=[
            dmc.Stack(gap=2, children=[
                dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=500,
                         style={"letterSpacing": "0.05em"}),
                dmc.Group(align="baseline", gap=4, children=[
                    dmc.Text(value, size="xl", fw=700,
                             style={"fontFamily": "DM Mono, monospace",
                                    "fontSize": "1.5rem"}),
                    dmc.Text(unit, size="sm", c="dimmed"),
                ]),
            ]),
            dmc.ThemeIcon(
                DashIconify(icon=icon, width=18),
                size="lg", radius="md", variant="light", color=color,
            ) if icon else html.Span(),
        ])],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

def layout(data: dict = None) -> dmc.Stack:
    d = data or PEST_DATA
    sen: pd.DataFrame   = d["sensitivity"]
    oss: pd.DataFrame   = d.get("oss_matrix", pd.DataFrame())
    params: pd.DataFrame = d.get("parameters", pd.DataFrame())
    obs_groups          = d.get("obs_groups", [])
    param_groups        = d.get("param_groups",
                                sorted(sen["group"].unique().tolist()))
    obs_group_map       = d.get("_obs_group_map", {})

    # ── summary scalars ───────────────────────────────────────────
    max_css    = sen["css"].max()
    n_high     = int((sen["css"] >= max_css * CSS_THRESHOLDS["high"]).sum())
    n_mod      = int(((sen["css"] >= max_css * CSS_THRESHOLDS["moderate"]) &
                       (sen["css"] <  max_css * CSS_THRESHOLDS["high"])).sum())
    n_low      = len(sen) - n_high - n_mod
    most_sens  = sen.loc[sen["css"].idxmax(), "name"]

    pgroup_chips = [
        dmc.Chip(g, value=g, size="xs", variant="light")
        for g in param_groups
    ]
    ogroup_chips = [
        dmc.Chip(g, value=g, size="xs", variant="light")
        for g in obs_groups
    ]

    return dmc.Stack(
        gap="lg", p="lg",
        children=[

            # ── Page header ───────────────────────────────────────
            dmc.Group(
                justify="space-between", align="flex-end",
                children=[
                    dmc.Stack(gap=2, children=[
                        dmc.Title("Sensitivity Analysis", order=3,
                                  style={"color": "#1a1a1a"}),
                        dmc.Text(
                            "Composite scaled sensitivity (CSS), one-percent scaled "
                            "sensitivity (OSS) heatmap, and parameter ranking",
                            size="sm", c="dimmed",
                        ),
                    ]),
                    dmc.Group(gap="sm", children=[
                        dmc.Select(
                            id="sen-oss-metric",
                            data=[
                                {"value": "mean_abs", "label": "OSS: Mean |value|"},
                                {"value": "max_abs",  "label": "OSS: Max |value|"},
                                {"value": "sum_abs",  "label": "OSS: Sum |value|"},
                            ],
                            value="mean_abs", size="xs", w=160,
                            styles={"input": {"fontSize": "11px"}},
                        ),
                        dmc.Switch(
                            id="sen-log-toggle",
                            label="Log CSS", size="xs", checked=False,
                        ),
                    ]),
                ],
            ),

            # ── Parameter group filter ────────────────────────────
            dmc.Paper(
                shadow="xs", p="sm", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    dmc.Group(gap="md", wrap="wrap", children=[
                        dmc.Stack(gap=4, children=[
                            dmc.Text("Parameter groups", size="xs",
                                     c="dimmed", fw=500),
                            dmc.ChipGroup(
                                id="sen-pgroup-filter",
                                multiple=True,
                                value=param_groups,
                                children=pgroup_chips,
                            ),
                        ]),
                        dmc.Divider(orientation="vertical"),
                        dmc.Stack(gap=4, children=[
                            dmc.Text("Observation groups", size="xs",
                                     c="dimmed", fw=500),
                            dmc.ChipGroup(
                                id="sen-ogroup-filter",
                                multiple=True,
                                value=obs_groups,
                                children=ogroup_chips,
                            ),
                        ]),
                    ]),
                ],
            ),

            dmc.Divider(),

            # ── KPI row ───────────────────────────────────────────
            dmc.SimpleGrid(
                cols={"base": 2, "sm": 4},
                spacing="md",
                children=[
                    _kpi("Parameters",  str(len(sen)),
                         icon="tabler:adjustments-horizontal", color="teal"),
                    _kpi("Most sensitive", most_sens,
                         icon="tabler:flame",          color="red"),
                    _kpi("High sensitivity",  str(n_high),
                         icon="tabler:arrow-up",       color="green"),
                    _kpi("Low sensitivity",   str(n_low),
                         icon="tabler:arrow-down",     color="orange"),
                ],
            ),

            # ── Sensitivity budget (dominance strip) ──────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    _section(
                        "Sensitivity Budget by Parameter Group",
                        "Each segment = share of total CSS across all parameters",
                    ),
                    dcc.Graph(
                        id="sen-dominance",
                        figure=_css_dominance(sen, param_groups, param_groups),
                        config={"displayModeBar": False},
                        style={"height": "130px"},
                    ),
                ],
            ),

            # ── CSS bar + OSS heatmap ─────────────────────────────
            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        span={"base": 12, "md": 5},
                        children=dmc.Paper(
                            shadow="xs", p="md", radius="md", withBorder=True,
                            style={"borderColor": "#eee"},
                            children=[
                                _section(
                                    "Composite Scaled Sensitivity (CSS)",
                                    "Overall parameter importance — higher = more influential",
                                ),
                                dcc.Graph(
                                    id="sen-css-bar",
                                    figure=_css_bar(sen, param_groups, param_groups),
                                    config={"displayModeBar": False},
                                    style={"height": "420px", "overflowY": "auto"},
                                ),
                            ],
                        ),
                    ),
                    dmc.GridCol(
                        span={"base": 12, "md": 7},
                        children=dmc.Paper(
                            shadow="xs", p="md", radius="md", withBorder=True,
                            style={"borderColor": "#eee"},
                            children=[
                                _section(
                                    "OSS Heatmap — Observation Group × Parameter",
                                    "Aggregated one-percent scaled sensitivity. "
                                    "Dark = parameter strongly affects that obs group",
                                ),
                                dcc.Graph(
                                    id="sen-oss-heatmap",
                                    figure=_oss_heatmap(
                                        oss, obs_group_map,
                                        obs_groups, param_groups,
                                        param_groups, obs_groups,
                                    ),
                                    config={"displayModeBar": False},
                                    style={"height": "420px"},
                                ),
                            ],
                        ),
                    ),
                ],
            ),

            # ── CSS vs param value scatter ────────────────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    _section(
                        "CSS vs Parameter Value",
                        "Log-log scatter — detects scale-related sensitivity issues. "
                        "Points far from the cloud may indicate unit or scaling problems",
                    ),
                    dcc.Graph(
                        id="sen-css-value",
                        figure=_css_vs_value(sen, params, param_groups, param_groups)
                        if not params.empty else go.Figure(),
                        config={"displayModeBar": False},
                        style={"height": "300px"},
                    ),
                ],
            ),

            # ── Parameter ranking table ───────────────────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    _section(
                        "Parameter Sensitivity Ranking",
                        "Full ranked table with sensitivity tier, CSS value, "
                        "and mean absolute OSS",
                    ),
                    html.Div(
                        id="sen-ranking-table",
                        children=_param_ranking_table(
                            sen, param_groups, param_groups
                        ),
                    ),
                ],
            ),

            # ── Insight alert ─────────────────────────────────────
            dmc.Alert(
                title="Sensitivity notes",
                color="teal", variant="light", radius="md",
                icon=DashIconify(icon="tabler:bulb", width=18),
                children=dmc.List(
                    size="sm", spacing="xs",
                    children=[
                        dmc.ListItem(
                            f"{n_high} parameter(s) classified as high sensitivity "
                            f"(CSS ≥ {CSS_THRESHOLDS['high']*100:.0f}% of maximum)."
                        ),
                        dmc.ListItem(
                            f"{n_low} parameter(s) have low sensitivity — consider "
                            "fixing these to their prior values to reduce "
                            "over-parameterisation."
                        ),
                        dmc.ListItem(
                            f"Most sensitive parameter: {most_sens}. "
                            "Verify its prior uncertainty is well-constrained."
                        ),
                    ],
                ),
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:
    d           = PEST_DATA
    sen         = d["sensitivity"]
    oss         = d.get("oss_matrix", pd.DataFrame())
    params      = d.get("parameters", pd.DataFrame())
    obs_groups  = d.get("obs_groups", [])
    param_groups= d.get("param_groups",
                        sorted(sen["group"].unique().tolist()))
    obs_gmap    = d.get("_obs_group_map", {})

    @app.callback(
        [Output("sen-css-bar",       "figure"),
         Output("sen-dominance",     "figure"),
         Output("sen-oss-heatmap",   "figure"),
         Output("sen-css-value",     "figure"),
         Output("sen-ranking-table", "children")],
        [Input("sen-pgroup-filter",  "value"),
         Input("sen-ogroup-filter",  "value"),
         Input("sen-oss-metric",     "value"),
         Input("sen-log-toggle",     "checked")],
    )
    def update(sel_pg, sel_og, oss_metric, log_css):
        sp = sel_pg or param_groups
        so = sel_og or obs_groups
        return (
            _css_bar(sen, param_groups, sp, log_scale=bool(log_css)),
            _css_dominance(sen, param_groups, sp),
            _oss_heatmap(oss, obs_gmap, obs_groups, param_groups,
                         sp, so, metric=oss_metric or "mean_abs"),
            _css_vs_value(sen, params, param_groups, sp)
                if not params.empty else go.Figure(),
            _param_ranking_table(sen, param_groups, sp),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  STANDALONE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700"
            "&family=DM+Mono&display=swap"
        ],
    )
    app.layout = dmc.MantineProvider(
        theme={"fontFamily": "DM Sans, sans-serif", "primaryColor": "teal"},
        children=dmc.Container(layout(), size="xl", py="lg"),
    )
    register_callbacks(app)
    app.run(debug=True)
