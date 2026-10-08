"""
PEST Calibration Diagnostics — Contribution Matrix Page
========================================================
C[i,j] = -r_i * J[i,j] * Δθ_j

Shows:
  • Full contribution heatmap with row & column sum sidebars
  • Row-sum bar (which observations drive the update)
  • Column-sum bar (which parameters are helping vs hurting)
  • Top contributors table
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

try:
    from components.shared import (
        base_layout, GRID, section_header, kpi_card, page_header,
        insight_alert, ogroup_colors, pgroup_colors,
    )
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from components.shared import (
        base_layout, GRID, section_header, kpi_card, page_header,
        insight_alert, ogroup_colors, pgroup_colors,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1.  MOCK DATA
# ──────────────────────────────────────────────────────────────────────────────

def _mock_data() -> dict:
    rng = np.random.default_rng(42)
    n_obs, n_params = 12, 9

    obs_groups = ["Heads"]*5 + ["Flows"]*3 + ["Conc."]*2 + ["Levels"]*2
    param_groups = (["Hydraulic cond."]*3 + ["Storage"]*2 +
                    ["Recharge"]*2 + ["Boundary"]*2)

    obs_labels   = [f"Obs {i+1}" if i < 5 else
                    (f"Obs m-{n_obs-i-1}" if i < n_obs-1 else "Obs m")
                    for i in range(n_obs)]
    param_labels = [f"Par {j+1}" if j < 5 else
                    (f"Par p-{n_params-j-1}" if j < n_params-1 else "Par p")
                    for j in range(n_params)]

    J  = rng.standard_normal((n_obs, n_params)) * 0.3
    dp = rng.standard_normal(n_params) * 0.4
    r  = rng.standard_normal(n_obs) * 0.3

    C = -r[:, None] * J * dp[None, :]
    return {
        "jacobian":   pd.DataFrame(J, index=obs_labels, columns=param_labels),
        "residuals":  pd.DataFrame({
            "name": obs_labels, "group": obs_groups,
            "residual": r, "weighted_res": r,
        }),
        "parameters": pd.DataFrame({
            "name": param_labels, "group": param_groups,
            "value": rng.uniform(0.1, 10, n_params),
        }),
        "_dp": dp,
        "obs_groups":  list(dict.fromkeys(obs_groups)),
        "param_groups": list(dict.fromkeys(param_groups)),
    }

PEST_DATA = _mock_data()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  COMPUTE CONTRIBUTION MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def _compute_C(data: dict):
    J  = data["jacobian"].values
    r  = data["residuals"]["weighted_res"].values
    dp = data.get("_dp", np.ones(J.shape[1]))
    C  = -r[:, None] * J * dp[None, :]
    return C, data["jacobian"].index.tolist(), data["jacobian"].columns.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _heatmap_fig(C, obs_labels, param_labels, annotate=False):
    from matplotlib.colors import TwoSlopeNorm
    vmax = max(np.abs(C).max(), 1e-6)

    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.82, 0.18],
        row_heights=[0.85, 0.15],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=C, x=param_labels, y=obs_labels,
        colorscale="RdBu_r",
        zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(
            title=dict(text="C[i,j]", font=dict(size=9)),
            thickness=12, len=0.7,
            x=1.01, tickfont=dict(size=8),
        ),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>C = %{z:.4f}<extra></extra>",
        xgap=1, ygap=1,
    ), row=1, col=1)

    # Row sums
    bar_colors_r = ["#c0774a" if v >= 0 else "#4a90d9" for v in row_sums]
    fig.add_trace(go.Bar(
        x=row_sums, y=obs_labels,
        orientation="h",
        marker_color=bar_colors_r,
        marker_line_width=0,
        showlegend=False,
        hovertemplate="%{y}<br>Σ = %{x:.4f}<extra></extra>",
    ), row=1, col=2)
    fig.add_vline(x=0, line=dict(color="#aaa", width=0.8), row=1, col=2)

    # Col sums
    bar_colors_c = ["#c0774a" if v >= 0 else "#4a90d9" for v in col_sums]
    fig.add_trace(go.Bar(
        x=param_labels, y=col_sums,
        marker_color=bar_colors_c,
        marker_line_width=0,
        showlegend=False,
        hovertemplate="%{x}<br>Σ = %{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color="#aaa", width=0.8), row=2, col=1)

    n_obs, n_params = C.shape
    bl = base_layout(dict(l=80, r=20, t=20, b=60))
    bl["font"] = dict(family="DM Sans, sans-serif", size=10, color="#555")
    fig.update_layout(
        **bl,
        height=max(380, n_obs * 34 + 100),
        showlegend=False,
    )
    fig.update_xaxes(tickfont=dict(size=9), showgrid=False, zeroline=False)
    fig.update_yaxes(tickfont=dict(size=9), showgrid=False, zeroline=False)
    fig.update_xaxes(tickangle=-35, row=1, col=1)
    fig.update_xaxes(title_text="Σ C[i,·]", row=1, col=2, title_font=dict(size=9))
    fig.update_yaxes(title_text="Σ C[·,j]", row=2, col=1, title_font=dict(size=9))
    return fig


def _top_contributors_table(C, obs_labels, param_labels, top_n=10):
    rows = []
    for i, obs in enumerate(obs_labels):
        for j, par in enumerate(param_labels):
            rows.append({"obs": obs, "param": par, "value": C[i, j]})
    df = (pd.DataFrame(rows)
          .reindex(pd.Series([r["value"] for r in rows]).abs()
                   .sort_values(ascending=False).index)
          .head(top_n)
          .reset_index(drop=True))

    hstyle = {"background": "#f4f5f7", "padding": "6px 12px",
               "fontSize": "11px", "fontWeight": "600", "color": "#555",
               "borderBottom": "1px solid #e8e9ec", "textAlign": "right"}
    cstyle = {"padding": "5px 12px", "fontSize": "11px",
               "fontFamily": "DM Mono, monospace", "textAlign": "right",
               "borderBottom": "1px solid #f0f1f3"}
    lstyle = {**cstyle, "textAlign": "left", "fontFamily": "DM Sans, sans-serif"}

    def chip(val):
        color = "#c0774a" if val >= 0 else "#4a90d9"
        return html.Span(f"{val:+.4f}", style={"color": color, "fontWeight": "600"})

    header = html.Tr([html.Th(c, style=hstyle)
                      for c in ["Rank", "Observation", "Parameter", "C[i,j]"]])
    body = [html.Tr([
        html.Td(f"#{i+1}", style={**cstyle, "color": "#999"}),
        html.Td(row["obs"],   style=lstyle),
        html.Td(row["param"], style=lstyle),
        html.Td(chip(row["value"]), style=cstyle),
    ]) for i, row in df.iterrows()]

    return html.Div(style={"overflowX": "auto"},
                    children=[html.Table(
                        style={"width": "100%", "borderCollapse": "collapse"},
                        children=[html.Thead(header), html.Tbody(body)])])


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

def layout(data: dict = None) -> dmc.Stack:
    d = data or PEST_DATA
    C, obs_labels, param_labels = _compute_C(d)
    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)
    total_pos = C[C > 0].sum()
    total_neg = abs(C[C < 0].sum())
    best_param = param_labels[int(np.argmax(col_sums))]
    worst_param = param_labels[int(np.argmin(col_sums))]

    return dmc.Stack(gap="lg", p="lg", children=[
        page_header(
            "Contribution Matrix",
            "C[i,j] = −rᵢ · J[i,j] · Δθⱼ  —  contribution of parameter j to reducing residual i",
        ),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base": 2, "sm": 4}, spacing="md", children=[
            kpi_card("Parameters",   str(len(param_labels)), icon="tabler:adjustments-horizontal", color="teal"),
            kpi_card("Observations", str(len(obs_labels)),   icon="tabler:list-numbers",           color="blue"),
            kpi_card("Best param",   best_param,             icon="tabler:arrow-up",               color="green"),
            kpi_card("Worst param",  worst_param,            icon="tabler:arrow-down",             color="red"),
        ]),
        dmc.Paper(shadow="xs", p="md", radius="md", withBorder=True,
                  style={"borderColor": "#eee"},
                  children=[
                      section_header(
                          "Contribution Heatmap",
                          "Red = reduces residual (good)  |  Blue = increases residual (undesired)  |  "
                          "White ≈ no effect",
                      ),
                      dcc.Graph(
                          id="contrib-heatmap",
                          figure=_heatmap_fig(C, obs_labels, param_labels),
                          config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]},
                      ),
                  ]),
        dmc.Paper(shadow="xs", p="md", radius="md", withBorder=True,
                  style={"borderColor": "#eee"},
                  children=[
                      section_header("Top Contributors",
                                     "Observation–parameter pairs with the largest |C[i,j]|"),
                      html.Div(id="contrib-table",
                               children=_top_contributors_table(C, obs_labels, param_labels)),
                  ]),
        insight_alert([
            f"Total positive contribution (misfit reduction): {total_pos:.4f}",
            f"Total negative contribution (misfit increase):  {total_neg:.4f}",
            f"Net: {total_pos - total_neg:+.4f}. "
            + ("Mostly beneficial update." if total_pos > total_neg
               else "Update is causing more harm than good — review parameter constraints."),
            f"Best parameter: {best_param} (largest column sum).",
            f"Worst parameter: {worst_param} (most negative column sum) — "
            "consider tighter bounds or regularisation.",
        ]),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def register_callbacks(app) -> None:
    pass  # Static page — no interactive filters needed here


# ──────────────────────────────────────────────────────────────────────────────
# 6.  STANDALONE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = dash.Dash(__name__, external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap"
    ])
    app.layout = dmc.MantineProvider(
        theme={"fontFamily": "DM Sans, sans-serif", "primaryColor": "teal"},
        children=dmc.Container(layout(), size="xl", py="lg"),
    )
    register_callbacks(app)
    app.run(debug=True)
