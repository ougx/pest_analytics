# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:46:12 2026

@author: hydro
"""

"""
PEST Calibration Diagnostics — Objective Function Page
=======================================================
Drop-in Dash page module.  Mount it inside your AppShell like:

    from objective_function import layout, register_callbacks
    register_callbacks(app)

The page expects a data dict from your pest_reader.  A mock is provided
so the file runs standalone:

    python objective_function.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, callback
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MOCK DATA  (replace with your pest_reader output)
# ──────────────────────────────────────────────────────────────────────────────

def _mock_pest_data():
    """Generate realistic-looking PEST optimisation history."""
    rng = np.random.default_rng(42)
    n_iter = 14

    # Φ curves — exponential decay + noise
    t = np.arange(n_iter)
    phi_total = 12_000 * np.exp(-0.35 * t) + rng.normal(0, 80, n_iter)
    phi_meas  = phi_total * 0.72
    phi_reg   = phi_total * 0.28

    # Lambda — large then drops
    lambda_vals = 100 * np.exp(-0.4 * t) + rng.uniform(0.1, 2, n_iter)

    # Φ by observation group (last iteration)
    obs_groups   = ["Heads", "Flows", "Conc.", "Levels", "Temps"]
    phi_by_group = np.array([480, 210, 340, 95, 79])

    # Φ per group per iteration (stacked area)
    group_phi_iter = np.outer(
        phi_by_group / phi_by_group.sum(), phi_meas
    ) + rng.normal(0, 5, (len(obs_groups), n_iter))
    group_phi_iter = np.clip(group_phi_iter, 0, None)

    # Relative improvement per iteration
    rel_improve = np.diff(phi_total) / phi_total[:-1] * 100

    return {
        "n_iter": n_iter,
        "iterations": t + 1,
        "phi_total": phi_total,
        "phi_meas":  phi_meas,
        "phi_reg":   phi_reg,
        "lambda_vals": lambda_vals,
        "obs_groups":   obs_groups,
        "phi_by_group": phi_by_group,
        "group_phi_iter": group_phi_iter,
        "rel_improve": rel_improve,
        "converged": True,
        "phimlim": 850.0,
    }

PEST_DATA = _mock_pest_data()


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PLOTLY THEME HELPERS
# ──────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "total":   "#2f7a5e",
    "meas":    "#4a90d9",
    "reg":     "#c0774a",
    "lambda":  "#9b7ec8",
    "groups":  ["#4a90d9", "#2f7a5e", "#c0774a", "#e8a838", "#8ab3cc"],
    "improve": "#2f7a5e",
    "warn":    "#d94a4a",
    "phimlim": "#e8a838",
}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=11, color="#555"),
    margin=dict(l=50, r=20, t=30, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        font=dict(size=10),
        bgcolor="rgba(0,0,0,0)",
    ),

    hoverlabel=dict(bgcolor="white", bordercolor="#ddd", font_size=11),
)

def _ax(overrides=None):
    ax = dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.05)",
        zeroline=False, linecolor="rgba(0,0,0,0.12)",
        tickfont=dict(size=10),
    )
    if overrides:
        ax.update(overrides)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _phi_convergence_chart(data, log_scale: bool = False):
    iters = data["iterations"]
    fig = go.Figure()

    # Shaded region under total Φ
    fig.add_trace(go.Scatter(
        x=np.concatenate([iters, iters[::-1]]),
        y=np.concatenate([data["phi_total"], np.zeros(len(iters))]),
        fill="toself",
        fillcolor="rgba(47,122,94,0.07)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Measurement Φ
    fig.add_trace(go.Scatter(
        x=iters, y=data["phi_meas"],
        name="Measurement Φ",
        line=dict(color=PALETTE["meas"], width=2),
        mode="lines+markers",
        marker=dict(size=5, symbol="circle"),
    ))

    # Regularisation Φ
    fig.add_trace(go.Scatter(
        x=iters, y=data["phi_reg"],
        name="Regularisation Φ",
        line=dict(color=PALETTE["reg"], width=2, dash="dash"),
        mode="lines+markers",
        marker=dict(size=5, symbol="diamond"),
    ))

    # Total Φ
    fig.add_trace(go.Scatter(
        x=iters, y=data["phi_total"],
        name="Total Φ",
        line=dict(color=PALETTE["total"], width=2.5),
        mode="lines+markers",
        marker=dict(size=6),
    ))

    # PHIMLIM target line
    fig.add_hline(
        y=data["phimlim"],
        line=dict(color=PALETTE["phimlim"], width=1.5, dash="dot"),
        annotation_text="PHIMLIM",
        annotation_position="right",
        annotation_font=dict(size=9, color=PALETTE["phimlim"]),
    )

    fig.update_layout(
        **BASE_LAYOUT,
        yaxis_type="log" if log_scale else "linear",
        yaxis_title="Φ value",
        xaxis=_ax({"dtick": 1, "title": "Iteration"}),
    )
    return fig


def _phi_by_group_chart(data):
    fig = go.Figure()
    groups = data["obs_groups"]
    phi    = data["phi_by_group"]
    pct    = phi / phi.sum() * 100

    for i, (grp, val, p, col) in enumerate(
        zip(groups, phi, pct, PALETTE["groups"])
    ):
        fig.add_trace(go.Bar(
            name=grp,
            x=[grp],
            y=[val],
            marker_color=col,
            text=[f"{p:.1f}%"],
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate=f"<b>{grp}</b><br>Φ = %{{y:.1f}}<br>Share = {p:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        showlegend=False,
        yaxis_title="Φ contribution",
        xaxis_title="Observation group",
        barmode="group",
        bargap=0.25,
    )
    return fig


def _stacked_group_iter_chart(data):
    fig = go.Figure()
    iters  = data["iterations"]
    groups = data["obs_groups"]

    for i, (grp, col) in enumerate(zip(groups, PALETTE["groups"])):
        fig.add_trace(go.Scatter(
            x=iters,
            y=data["group_phi_iter"][i],
            name=grp,
            stackgroup="one",
            opacity=0.7,
            line=dict(color=col, width=1),
            hovertemplate=f"<b>{grp}</b><br>Iter %{{x}}<br>Φ = %{{y:.1f}}<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        yaxis_title="Measurement Φ",
        xaxis=_ax({"dtick": 1, "title": "Iteration"}),
    )
    return fig


def _lambda_chart(data):
    iters = data["iterations"]
    lam   = data["lambda_vals"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=lam,
        name="λ (Marquardt)",
        line=dict(color=PALETTE["lambda"], width=2),
        mode="lines+markers",
        marker=dict(size=5, symbol="square"),
        fill="tozeroy",
        fillcolor="rgba(155,126,200,0.08)",
        hovertemplate="Iter %{x}<br>λ = %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        yaxis_title="λ (log scale)",
        yaxis_type="log",
        xaxis=_ax({"dtick": 1, "title": "Iteration"}),
        showlegend=False,
    )
    return fig


def _relative_improvement_chart(data):
    iters = data["iterations"][1:]     # one fewer point
    improv = data["rel_improve"]

    colors = [
        PALETTE["improve"] if v < 0 else PALETTE["warn"]
        for v in improv
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=iters,
        y=improv,
        marker_color=colors,
        hovertemplate="Iter %{x}<br>ΔΦ/Φ = %{y:.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color="#aaa", width=0.8))

    fig.update_layout(
        **BASE_LAYOUT,
        yaxis_title="Relative Φ change (%)",
        xaxis=_ax({"dtick": 1, "title": "Iteration"}),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4.  KPI CARD COMPONENT
# ──────────────────────────────────────────────────────────────────────────────

def kpi_card(label, value, unit="", icon=None, color="teal", delta=None):
    """A compact metric card using dmc.Paper."""
    delta_el = []
    if delta is not None:
        sign  = "▲" if delta > 0 else "▼"
        clr   = "red" if delta > 0 else "teal"
        delta_el = [
            dmc.Text(f"{sign} {abs(delta):.1f}%", size="xs", c=clr, mt=2)
        ]

    return dmc.Paper(
        shadow="xs",
        p="md",
        radius="md",
        withBorder=True,
        style={"borderColor": "#eee"},
        children=[
            dmc.Group(
                justify="space-between",
                align="flex-start",
                children=[
                    dmc.Stack(gap=2, children=[
                        dmc.Text(label, size="xs", c="dimmed", tt="uppercase",
                                 fw=500, style={"letterSpacing": "0.05em"}),
                        dmc.Group(align="baseline", gap=4, children=[
                            dmc.Text(value, size="xl", fw=700,
                                     style={"fontFamily": "DM Mono, monospace",
                                            "fontSize": "1.6rem"}),
                            dmc.Text(unit, size="sm", c="dimmed"),
                        ]),
                        *delta_el,
                    ]),
                    dmc.ThemeIcon(
                        DashIconify(icon=icon, width=18) if icon else html.Span(),
                        size="lg", radius="md", variant="light", color=color,
                    ) if icon else html.Span(),
                ],
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5.  SECTION HEADER HELPER
# ──────────────────────────────────────────────────────────────────────────────

def section_header(title, subtitle=None):
    return dmc.Stack(gap=0, mb="xs", children=[
        dmc.Text(title, fw=600, size="sm",
                 style={"color": "#333", "letterSpacing": "0.02em"}),
        dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else html.Span(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 6.  PAGE LAYOUT
# ──────────────────────────────────────────────────────────────────────────────

def layout(data: dict = None):
    """
    Return the full page layout.
    Pass your parsed pest_reader dict as `data`.
    Falls back to mock data if None.
    """
    d = data or PEST_DATA

    # ── derived scalars ──────────────────────────────────────────
    phi_final   = d["phi_total"][-1]
    phi_initial = d["phi_total"][0]
    reduction   = (phi_initial - phi_final) / phi_initial * 100
    pct_meas    = d["phi_meas"][-1] / d["phi_total"][-1] * 100
    pct_reg     = d["phi_reg"][-1]  / d["phi_total"][-1] * 100
    lam_final   = d["lambda_vals"][-1]
    converged   = d.get("converged", False)

    status_badge = dmc.Badge(
        "Converged" if converged else "Not converged",
        color="teal" if converged else "red",
        variant="light",
        size="sm",
        radius="sm",
    )

    return dmc.Stack(
        gap="lg",
        p="lg",
        children=[

            # ── Page title row ────────────────────────────────────
            dmc.Group(
                justify="space-between",
                align="flex-end",
                children=[
                    dmc.Stack(gap=2, children=[
                        dmc.Title("Objective Function", order=3,
                                  style={"color": "#1a1a1a"}),
                        dmc.Text(
                            "Φ breakdown, convergence history, and Marquardt λ",
                            size="sm", c="dimmed",
                        ),
                    ]),
                    dmc.Group(gap="sm", children=[
                        status_badge,
                        dmc.Select(
                            id="phi-scale-toggle",
                            data=[
                                {"value": "linear", "label": "Linear scale"},
                                {"value": "log",    "label": "Log scale"},
                            ],
                            value="linear",
                            size="xs",
                            w=130,
                            styles={"input": {"fontSize": "11px"}},
                        ),
                    ]),
                ],
            ),

            dmc.Divider(labelPosition="left"),

            # ── KPI row ───────────────────────────────────────────
            dmc.SimpleGrid(
                cols={"base": 2, "sm": 4},
                spacing="md",
                children=[
                    kpi_card(
                        "Final Total Φ",
                        f"{phi_final:,.0f}",
                        icon="tabler:target",
                        color="teal",
                        delta=-(reduction),   # negative = reduction (good)
                    ),
                    kpi_card(
                        "Measurement Φ",
                        f"{d['phi_meas'][-1]:,.0f}",
                        unit=f"({pct_meas:.0f}%)",
                        icon="tabler:chart-scatter",
                        color="blue",
                    ),
                    kpi_card(
                        "Regularisation Φ",
                        f"{d['phi_reg'][-1]:,.0f}",
                        unit=f"({pct_reg:.0f}%)",
                        icon="tabler:adjustments",
                        color="orange",
                    ),
                    kpi_card(
                        "Iterations",
                        str(d["n_iter"]),
                        icon="tabler:refresh",
                        color="grape",
                    ),
                ],
            ),

            # ── Φ convergence + Lambda (side by side) ─────────────
            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        span={"base": 12, "md": 8},
                        children=[
                            dmc.Paper(
                                shadow="xs", p="md", radius="md", withBorder=True,
                                style={"borderColor": "#eee"},
                                children=[
                                    section_header(
                                        "Φ Convergence",
                                        "Total, measurement and regularisation Φ across iterations",
                                    ),
                                    dcc.Graph(
                                        id="phi-convergence-chart",
                                        figure=_phi_convergence_chart(d),
                                        config={"displayModeBar": False},
                                        style={"height": "280px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.GridCol(
                        span={"base": 12, "md": 4},
                        children=[
                            dmc.Paper(
                                shadow="xs", p="md", radius="md", withBorder=True,
                                style={"borderColor": "#eee"},
                                children=[
                                    section_header(
                                        "Marquardt λ",
                                        "Lambda value per iteration (log scale)",
                                    ),
                                    dcc.Graph(
                                        figure=_lambda_chart(d),
                                        config={"displayModeBar": False},
                                        style={"height": "280px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ── Relative improvement bar ──────────────────────────
            dmc.Paper(
                shadow="xs", p="md", radius="md", withBorder=True,
                style={"borderColor": "#eee"},
                children=[
                    section_header(
                        "Relative Φ Improvement per Iteration",
                        "ΔΦ/Φ (%) — negative bars are good; positive bars signal a failed step",
                    ),
                    dcc.Graph(
                        figure=_relative_improvement_chart(d),
                        config={"displayModeBar": False},
                        style={"height": "180px"},
                    ),
                ],
            ),

            # ── Φ by group + stacked group iter ───────────────────
            dmc.Grid(
                gutter="md",
                children=[
                    dmc.GridCol(
                        span={"base": 12, "md": 5},
                        children=[
                            dmc.Paper(
                                shadow="xs", p="md", radius="md", withBorder=True,
                                style={"borderColor": "#eee"},
                                children=[
                                    section_header(
                                        "Φ by Observation Group",
                                        "Final iteration — which groups dominate the objective",
                                    ),
                                    dcc.Graph(
                                        figure=_phi_by_group_chart(d),
                                        config={"displayModeBar": False},
                                        style={"height": "260px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dmc.GridCol(
                        span={"base": 12, "md": 7},
                        children=[
                            dmc.Paper(
                                shadow="xs", p="md", radius="md", withBorder=True,
                                style={"borderColor": "#eee"},
                                children=[
                                    section_header(
                                        "Measurement Φ by Group — All Iterations",
                                        "Stacked area showing how each group's contribution evolves",
                                    ),
                                    dcc.Graph(
                                        figure=_stacked_group_iter_chart(d),
                                        config={"displayModeBar": False},
                                        style={"height": "260px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),

            # ── Insight callout ───────────────────────────────────
            dmc.Alert(
                title="Calibration notes",
                color="teal",
                variant="light",
                radius="md",
                icon=DashIconify(icon="tabler:bulb", width=18),
                children=[
                    dmc.List(
                        size="sm",
                        spacing="xs",
                        children=[
                            dmc.ListItem(
                                f"Total Φ reduced by {reduction:.1f}% "
                                f"from {phi_initial:,.0f} → {phi_final:,.0f} over {d['n_iter']} iterations."
                            ),
                            dmc.ListItem(
                                f"Measurement Φ accounts for {pct_meas:.0f}% of final Φ; "
                                f"regularisation accounts for {pct_reg:.0f}%."
                            ),
                            dmc.ListItem(
                                f"Final λ = {lam_final:.3f}. "
                                + ("Low λ indicates gradient-dominated update — good convergence behaviour."
                                   if lam_final < 1 else
                                   "Elevated λ suggests the optimiser is still struggling — consider more iterations.")
                            ),
                            dmc.ListItem(
                                "Φ target (PHIMLIM) is shown as a dotted line on the convergence chart. "
                                + ("Target reached ✓" if phi_final <= d["phimlim"] else
                                   f"Φ is still {phi_final - d['phimlim']:,.0f} above PHIMLIM.")
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CALLBACKS
# ──────────────────────────────────────────────────────────────────────────────

def register_callbacks(app):
    @app.callback(
        Output("phi-convergence-chart", "figure"),
        Input("phi-scale-toggle", "value"),
    )
    def update_phi_scale(scale):
        return _phi_convergence_chart(PEST_DATA, log_scale=(scale == "log"))


# ──────────────────────────────────────────────────────────────────────────────
# 8.  STANDALONE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap"
        ],
    )
    app.layout = dmc.MantineProvider(
        theme={
            "fontFamily": "DM Sans, sans-serif",
            "primaryColor": "teal",
            "components": {
                "Paper": {"defaultProps": {"shadow": "xs"}},
            },
        },
        children=dmc.Container(
            layout(),
            size="xl",
            py="lg",
        ),
    )
    register_callbacks(app)
    app.run(debug=True)