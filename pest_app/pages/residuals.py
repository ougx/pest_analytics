"""
PEST Calibration Diagnostics — Residuals Page
==============================================
Drop-in Dash page module.  Wire into your AppShell:

    from residuals import layout, register_callbacks
    register_callbacks(app)

Expects the dict from pest_reader.read_pest_run().
Runs standalone with mock data:

    python residuals.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_mantine_components as dmc
from dash_iconify import DashIconify


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MOCK DATA
# ──────────────────────────────────────────────────────────────────────────────

def _mock_residuals() -> dict:
    rng = np.random.default_rng(3)

    groups = {
        "Heads":  60,
        "Flows":  25,
        "Conc.":  30,
        "Levels": 20,
    }

    rows = []
    for grp, n in groups.items():
        obs_true = rng.uniform(1, 100, n)

        if grp == "Heads":
            sim = obs_true + rng.normal(0.2, 0.8, n)      # slight bias
            weight = rng.uniform(0.8, 1.2, n)
        elif grp == "Flows":
            sim = obs_true * rng.uniform(0.92, 1.08, n)
            weight = rng.uniform(0.01, 0.05, n)
        elif grp == "Conc.":
            sim = obs_true + rng.normal(-0.5, 3.0, n)     # noisier
            weight = rng.uniform(0.1, 0.3, n)
        else:
            sim = obs_true + rng.normal(0.0, 0.4, n)
            weight = rng.uniform(0.5, 1.0, n)

        for i in range(n):
            res = obs_true[i] - sim[i]
            rows.append({
                "name":         f"{grp[:2].lower()}{i+1:03d}",
                "group":        grp,
                "obs":          obs_true[i],
                "sim":          sim[i],
                "residual":     res,
                "weight":       weight[i],
                "weighted_res": res * weight[i],
            })

    df = pd.DataFrame(rows)
    return {
        "residuals":  df,
        "obs_groups": list(groups.keys()),
    }


PEST_DATA = _mock_residuals()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  COLOUR SCHEME  (cohesive with objective_function.py)
# ──────────────────────────────────────────────────────────────────────────────

GROUP_COLORS = ["#4a90d9", "#2f7a5e", "#c0774a", "#e8a838", "#8ab3cc",
                "#9b7ec8", "#d94a4a", "#5bab8a"]

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=11, color="#555"),
    margin=dict(l=52, r=20, t=36, b=44),
    hoverlabel=dict(bgcolor="white", bordercolor="#ddd", font_size=11),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
    ),
)

GRID = dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=False, linecolor="rgba(0,0,0,0.12)", tickfont=dict(size=10))


def _color_for_group(groups: list[str]) -> dict[str, str]:
    return {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}


# ──────────────────────────────────────────────────────────────────────────────
# 3.  STATISTICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _residual_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group summary statistics."""
    rows = []
    for grp, sub in df.groupby("group", sort=False):
        wr = sub["weighted_res"]
        r  = sub["residual"]
        rows.append({
            "Group":   grp,
            "N":       len(sub),
            "Mean res":     f"{r.mean():+.3f}",
            "Std res":      f"{r.std():.3f}",
            "Mean |wres|":  f"{wr.abs().mean():.3f}",
            "Max |wres|":   f"{wr.abs().max():.3f}",
            "RMSE":         f"{np.sqrt((r**2).mean()):.3f}",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _obs_vs_sim(df: pd.DataFrame, groups: list[str], selected: list[str]) -> go.Figure:
    sub = df[df["group"].isin(selected)] if selected else df
    colors = _color_for_group(groups)
    fig = go.Figure()

    # 1:1 line
    mn = sub[["obs", "sim"]].min().min()
    mx = sub[["obs", "sim"]].max().max()
    pad = (mx - mn) * 0.05
    fig.add_trace(go.Scatter(
        x=[mn - pad, mx + pad], y=[mn - pad, mx + pad],
        mode="lines",
        line=dict(color="#bbb", width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    # ±1 weighted residual envelope (optional visual guide)
    # (skip if weights are too heterogeneous)

    for grp in [g for g in groups if g in selected]:
        g_df = sub[sub["group"] == grp]
        # Size points by |weighted residual|, scaled between 5–16
        wr_abs = g_df["weighted_res"].abs()
        sz = 5 + 11 * (wr_abs - wr_abs.min()) / (wr_abs.max() - wr_abs.min() + 1e-9)

        fig.add_trace(go.Scatter(
            x=g_df["obs"],
            y=g_df["sim"],
            mode="markers",
            name=grp,
            marker=dict(
                color=colors[grp],
                size=sz,
                opacity=0.72,
                line=dict(width=0.5, color="white"),
            ),
            customdata=np.stack([
                g_df["residual"].values,
                g_df["weighted_res"].values,
                g_df["name"].values,
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[2]}</b><br>"
                "Obs = %{x:.3f}<br>"
                "Sim = %{y:.3f}<br>"
                "Residual = %{customdata[0]:+.3f}<br>"
                "W·Residual = %{customdata[1]:+.3f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**GRID, "title": "Observed"},
        yaxis={**GRID, "title": "Simulated", "scaleanchor": "x", "scaleratio": 1},
    )
    return fig


def _weighted_residuals_bar(df: pd.DataFrame, groups: list[str],
                            selected: list[str], sort_by: str = "group") -> go.Figure:
    sub = df[df["group"].isin(selected)] if selected else df
    colors = _color_for_group(groups)

    if sort_by == "magnitude":
        sub = sub.reindex(sub["weighted_res"].abs().sort_values(ascending=False).index)
    elif sort_by == "value":
        sub = sub.sort_values("weighted_res")
    # else: group order (default)

    fig = go.Figure()
    for grp in [g for g in groups if g in selected]:
        g_df = sub[sub["group"] == grp]
        clr_arr = [
            colors[grp] if v >= 0 else _darken(colors[grp])
            for v in g_df["weighted_res"]
        ]
        fig.add_trace(go.Bar(
            x=g_df["name"],
            y=g_df["weighted_res"],
            name=grp,
            marker_color=clr_arr,
            marker_line_width=0,
            opacity=0.82,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"Group: {grp}<br>"
                "W·Residual = %{y:+.3f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color="#555", width=0.8))

    # ±2σ guide lines
    wr = sub["weighted_res"]
    sig = wr.std()
    for s, dash in [(2 * sig, "dot"), (-2 * sig, "dot")]:
        fig.add_hline(
            y=s, line=dict(color="#d94a4a", width=1, dash=dash),
            annotation_text="±2σ" if s > 0 else "",
            annotation_position="right",
            annotation_font=dict(size=8, color="#d94a4a"),
        )

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="relative",
        xaxis={**GRID, "title": "Observation", "showticklabels": False,
               "showgrid": False},
        yaxis={**GRID, "title": "Weighted residual"},
    )
    return fig


def _residual_histogram(df: pd.DataFrame, groups: list[str],
                        selected: list[str]) -> go.Figure:
    sub = df[df["group"].isin(selected)] if selected else df
    colors = _color_for_group(groups)
    fig = go.Figure()

    for grp in [g for g in groups if g in selected]:
        g_df = sub[sub["group"] == grp]
        fig.add_trace(go.Histogram(
            x=g_df["weighted_res"],
            name=grp,
            marker_color=colors[grp],
            opacity=0.65,
            nbinsx=20,
            hovertemplate=f"{grp}<br>Count = %{{y}}<extra></extra>",
        ))

    # Normal distribution overlay
    wr_all = sub["weighted_res"]
    mu, sigma = wr_all.mean(), wr_all.std()
    x_norm = np.linspace(wr_all.min(), wr_all.max(), 200)
    from scipy.stats import norm as _norm
    pdf = _norm.pdf(x_norm, mu, sigma)
    # Scale to histogram counts
    bin_width = (wr_all.max() - wr_all.min()) / 20
    scale = len(wr_all) * bin_width
    fig.add_trace(go.Scatter(
        x=x_norm, y=pdf * scale,
        name="Normal fit",
        line=dict(color="#333", width=1.5, dash="dot"),
        hoverinfo="skip",
    ))

    fig.add_vline(x=0, line=dict(color="#555", width=0.8, dash="dash"))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="overlay",
        xaxis={**GRID, "title": "Weighted residual"},
        yaxis={**GRID, "title": "Count"},
    )
    return fig


def _qqplot(df: pd.DataFrame, groups: list[str], selected: list[str]) -> go.Figure:
    from scipy.stats import probplot
    sub = df[df["group"].isin(selected)] if selected else df
    wr = sub["weighted_res"].dropna().sort_values().values

    (osm, osr), (slope, intercept, _) = probplot(wr, dist="norm")

    fig = go.Figure()

    # Reference line
    x_line = np.array([osm[0], osm[-1]])
    fig.add_trace(go.Scatter(
        x=x_line, y=slope * x_line + intercept,
        mode="lines",
        line=dict(color="#bbb", width=1.5, dash="dash"),
        name="Normal ref.", showlegend=True,
    ))

    # Points coloured by distance from line
    deviations = np.abs(osr - (slope * osm + intercept))
    fig.add_trace(go.Scatter(
        x=osm, y=osr,
        mode="markers",
        marker=dict(
            color=deviations,
            colorscale="RdYlGn_r",
            size=6,
            opacity=0.8,
            colorbar=dict(
                title=dict(text="Deviation", font=dict(size=9)),
                thickness=10, len=0.6,
                tickfont=dict(size=8),
            ),
        ),
        name="Obs.",
        hovertemplate="Theoretical = %{x:.2f}<br>Sample = %{y:.2f}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**GRID, "title": "Theoretical quantiles"},
        yaxis={**GRID, "title": "Sample quantiles"},
        showlegend=True,
    )
    return fig


def _residual_map(df: pd.DataFrame, groups: list[str], selected: list[str]) -> go.Figure:
    """Bubble plot: observation index vs weighted residual, sized by |wres|."""
    sub = df[df["group"].isin(selected)] if selected else df
    sub = sub.reset_index(drop=True)
    colors = _color_for_group(groups)
    fig = go.Figure()

    for grp in [g for g in groups if g in selected]:
        g_df = sub[sub["group"] == grp].reset_index()
        wr_abs = g_df["weighted_res"].abs()
        sz = 6 + 14 * (wr_abs - wr_abs.min()) / (wr_abs.max() - wr_abs.min() + 1e-9)

        fig.add_trace(go.Scatter(
            x=g_df["index"],
            y=g_df["weighted_res"],
            mode="markers",
            name=grp,
            marker=dict(
                color=[colors[grp] if v >= 0 else "#d94a4a"
                       for v in g_df["weighted_res"]],
                size=sz,
                opacity=0.7,
                line=dict(width=0.4, color="white"),
            ),
            customdata=np.stack([g_df["name"].values, g_df["residual"].values], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Group: {grp}<br>"
                "W·Res = %{y:+.3f}<br>"
                "Residual = %{customdata[1]:+.3f}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color="#999", width=0.8))
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis={**GRID, "title": "Observation index", "showticklabels": False},
        yaxis={**GRID, "title": "Weighted residual"},
    )
    return fig


def _darken(hex_color: str, factor: float = 0.6) -> str:
    """Return a darker shade of a hex colour."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  STATS TABLE COMPONENT
# ──────────────────────────────────────────────────────────────────────────────

def _stats_table(df: pd.DataFrame, groups: list[str], selected: list[str]) -> html.Div:
    sub = df[df["group"].isin(selected)] if selected else df
    stats = _residual_stats(sub)
    colors = _color_for_group(groups)

    header_style = {
        "background": "#f4f5f7",
        "padding": "6px 12px",
        "fontSize": "11px",
        "fontWeight": "600",
        "color": "#555",
        "textAlign": "right",
        "letterSpacing": "0.03em",
        "borderBottom": "1px solid #e8e9ec",
    }
    cell_style = {
        "padding": "5px 12px",
        "fontSize": "11px",
        "textAlign": "right",
        "fontFamily": "DM Mono, monospace",
        "borderBottom": "1px solid #f0f1f3",
        "color": "#333",
    }
    grp_style = {**cell_style, "textAlign": "left", "fontFamily": "DM Sans, sans-serif",
                 "fontWeight": "600"}

    cols = ["Group", "N", "Mean res", "Std res", "Mean |wres|", "Max |wres|", "RMSE"]

    header_row = html.Tr([html.Th(c, style=header_style) for c in cols])

    body_rows = []
    for _, row in stats.iterrows():
        grp = row["Group"]
        dot = html.Span(style={
            "display": "inline-block", "width": "8px", "height": "8px",
            "borderRadius": "50%", "background": colors.get(grp, "#aaa"),
            "marginRight": "6px", "verticalAlign": "middle",
        })
        cells = [html.Td([dot, grp], style=grp_style)]
        for c in cols[1:]:
            cells.append(html.Td(row[c], style=cell_style))
        body_rows.append(html.Tr(cells))

    return html.Div(
        style={"overflowX": "auto"},
        children=[
            html.Table(
                style={"width": "100%", "borderCollapse": "collapse",
                       "fontSize": "12px"},
                children=[html.Thead(header_row), html.Tbody(body_rows)],
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  KPI CARDS  (shared with other pages via inline helper)
# ──────────────────────────────────────────────────────────────────────────────

def _kpi(label, value, unit="", icon=None, color="teal"):
    return dmc.Paper(
        shadow="xs", p="md", radius="md", withBorder=True,
        style={"borderColor": "#eee"},
        children=[
            dmc.Group(justify="space-between", align="flex-start", children=[
                dmc.Stack(gap=2, children=[
                    dmc.Text(label, size="xs", c="dimmed", tt="uppercase",
                             fw=500, style={"letterSpacing": "0.05em"}),
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
            ]),
        ],
    )


def _section(title, subtitle=None):
    return dmc.Stack(gap=0, mb="xs", children=[
        dmc.Text(title, fw=600, size="sm", style={"color": "#333",
                                                   "letterSpacing": "0.02em"}),
        dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else html.Span(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 7.  PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

def layout(data: dict = None) -> dmc.Stack:
    d = data or PEST_DATA
    df: pd.DataFrame = d["residuals"]
    groups: list[str] = d.get("obs_groups", sorted(df["group"].unique().tolist()))

    # ── summary stats ─────────────────────────────────────────────
    wr = df["weighted_res"]
    r  = df["residual"]
    outliers = int((wr.abs() > 2 * wr.std()).sum())
    bias_flag = abs(wr.mean()) > 0.5 * wr.std()

    group_chip_data = [{"value": g, "label": g} for g in groups]

    return dmc.Stack(
        gap="lg",
        p="lg",
        children=[

            # ── Page header ───────────────────────────────────────
            dmc.Group(
                justify="space-between", align="flex-end",
                children=[
                    dmc.Stack(gap=2, children=[
                        dmc.Title("Residuals", order=3,
                                  style={"color": "#1a1a1a"}),
                        dmc.Text(
                            "Observed vs simulated, weighted residual analysis, "
                            "and normality diagnostics",
                            size="sm", c="dimmed",
                        ),
                    ]),
                    dmc.Group(gap="sm", children=[
                        dmc.Select(
                            id="res-sort-select",
                            label=None,
                            data=[
                                {"value": "group",     "label": "Sort: by group"},
                                {"value": "magnitude", "label": "Sort: by magnitude"},
                                {"value": "value",     "label": "Sort: by value"},
                            ],
                            value="group",
                            size="xs", w=160,
                            styles={"input": {"fontSize": "11px"}},
                        ),
                    ]),
                ],
            ),

            # ── Group filter chips ────────────────────────────────
            dmc.Group(gap="xs", children=[
                dmc.Text("Filter groups:", size="xs", c="dimmed", mt=2),
                dmc.ChipGroup(
                    id="res-group-filter",
                    multiple=True,
                    value=groups,
                    children=[
                        dmc.Chip(g, value=g, size="xs", variant="light",
                                 color=GROUP_COLORS[i % len(GROUP_COLORS)]
                                       .replace("#", "").lower()
                                       if False else "teal")
                        for i, g in enumerate(groups)
                    ],
                ),
            ]),

            dmc.Divider(),

            # ── KPI row ───────────────────────────────────────────
            dmc.SimpleGrid(
                cols={"base": 2, "sm": 5},
                spacing="md",
                children=[
                    _kpi("Observations",   str(len(df)),
                         icon="tabler:list-numbers",    color="teal"),
                    _kpi("Mean W·Residual", f"{wr.mean():+.3f}",
                         icon="tabler:minus",           color="blue"),
                    _kpi("Std W·Residual",  f"{wr.std():.3f}",
                         icon="tabler:wave-sine",       color="indigo"),
                    _kpi("RMSE",            f"{np.sqrt((r**2).mean()):.3f}",
                         icon="tabler:chart-area",      color="orange"),
                    _kpi("Outliers (>2σ)",  str(outliers),
                         icon="tabler:alert-triangle",
                         color="red" if outliers > 0 else "teal"),
                ],
            ),

            # ── Bias alert ────────────────────────────────────────
            dmc.Alert(
                title="Systematic bias detected" if bias_flag else "No systematic bias",
                color="orange" if bias_flag else "teal",
                variant="light",
                radius="md",
                icon=DashIconify(
                    icon="tabler:alert-triangle" if bias_flag else "tabler:circle-check",
                    width=16,
                ),
                children=dmc.Text(
                    f"Mean weighted residual = {wr.mean():+.4f}. "
                    + ("The model is systematically over- or under-fitting. "
                       "Check parameter bounds, conceptual model, or observation weights."
                       if bias_flag else
                       "Mean weighted residual is within acceptable range."),
                    size="sm",
                ),
            ) if True else html.Span(),

            # ── Observed vs Simulated  +  Weighted residuals bar ──
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
                                    "Observed vs Simulated",
                                    "Marker size = |weighted residual|; "
                                    "dashed line = 1:1",
                                ),
                                dcc.Graph(
                                    id="res-obs-sim",
                                    figure=_obs_vs_sim(df, groups, groups),
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
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
                                    "Weighted Residuals",
                                    "Red dashed lines = ±2σ threshold",
                                ),
                                dcc.Graph(
                                    id="res-wres-bar",
                                    figure=_weighted_residuals_bar(df, groups, groups),
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                ),
                            ],
                        ),
                    ),
                ],
            ),

            # ── Bubble map  +  Histogram ──────────────────────────
            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        span={"base": 12, "md": 7},
                        children=dmc.Paper(
                            shadow="xs", p="md", radius="md", withBorder=True,
                            style={"borderColor": "#eee"},
                            children=[
                                _section(
                                    "Residual Bubble Map",
                                    "Sized by |weighted residual|; "
                                    "blue = positive, red = negative",
                                ),
                                dcc.Graph(
                                    id="res-bubble",
                                    figure=_residual_map(df, groups, groups),
                                    config={"displayModeBar": False},
                                    style={"height": "260px"},
                                ),
                            ],
                        ),
                    ),
                    dmc.GridCol(
                        span={"base": 12, "md": 5},
                        children=dmc.Paper(
                            shadow="xs", p="md", radius="md", withBorder=True,
                            style={"borderColor": "#eee"},
                            children=[
                                _section(
                                    "Weighted Residual Distribution",
                                    "Overlaid normal-fit curve",
                                ),
                                dcc.Graph(
                                    id="res-histogram",
                                    figure=_residual_histogram(df, groups, groups),
                                    config={"displayModeBar": False},
                                    style={"height": "260px"},
                                ),
                            ],
                        ),
                    ),
                ],
            ),

            # ── Q-Q Plot ──────────────────────────────────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    _section(
                        "Normal Q-Q Plot",
                        "Points lying on the dashed line indicate normally "
                        "distributed weighted residuals — a calibration quality indicator",
                    ),
                    dcc.Graph(
                        id="res-qq",
                        figure=_qqplot(df, groups, groups),
                        config={"displayModeBar": False},
                        style={"height": "240px"},
                    ),
                ],
            ),

            # ── Stats table ───────────────────────────────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    _section(
                        "Per-group Summary Statistics",
                        "RMSE and weighted-residual diagnostics by observation group",
                    ),
                    html.Div(id="res-stats-table",
                             children=_stats_table(df, groups, groups)),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:
    d = PEST_DATA
    df: pd.DataFrame    = d["residuals"]
    groups: list[str]   = d.get("obs_groups", sorted(df["group"].unique()))

    outputs = [
        Output("res-obs-sim",     "figure"),
        Output("res-wres-bar",    "figure"),
        Output("res-bubble",      "figure"),
        Output("res-histogram",   "figure"),
        Output("res-qq",          "figure"),
        Output("res-stats-table", "children"),
    ]

    @app.callback(
        outputs,
        Input("res-group-filter", "value"),
        Input("res-sort-select",  "value"),
    )
    def update_all(selected_groups, sort_by):
        sel = selected_groups or groups
        return (
            _obs_vs_sim(df, groups, sel),
            _weighted_residuals_bar(df, groups, sel, sort_by or "group"),
            _residual_map(df, groups, sel),
            _residual_histogram(df, groups, sel),
            _qqplot(df, groups, sel),
            _stats_table(df, groups, sel),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 9.  STANDALONE RUNNER
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
