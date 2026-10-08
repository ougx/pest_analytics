"""
components/shared.py
====================
Shared theme tokens, chart helpers, and reusable UI components
used across all diagnostic pages.
"""
from __future__ import annotations
import numpy as np
from dash import html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "total":   "#2f7a5e",
    "meas":    "#4a90d9",
    "reg":     "#c0774a",
    "lambda":  "#9b7ec8",
    "warn":    "#d94a4a",
    "phimlim": "#e8a838",
}
PARAM_GROUP_COLORS = ["#4a90d9","#2f7a5e","#c0774a","#9b7ec8","#e8a838",
                      "#d94a4a","#8ab3cc","#5bab8a","#a78bda","#60b8b0"]
OBS_GROUP_COLORS   = ["#4a90d9","#2f7a5e","#c0774a","#e8a838","#9b7ec8","#d94a4a"]

# ── Plotly base layout ────────────────────────────────────────────────────────
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
GRID = dict(
    showgrid=True, gridcolor="rgba(0,0,0,0.05)",
    zeroline=False, linecolor="rgba(0,0,0,0.12)", tickfont=dict(size=10),
)

def base_layout(extra_margin: dict | None = None) -> dict:
    """Return BASE_LAYOUT with optional margin override (avoids kwarg conflict)."""
    b = {k: v for k, v in BASE_LAYOUT.items() if k != "margin"}
    b["margin"] = extra_margin or BASE_LAYOUT["margin"]
    return b

# ── Colour helpers ────────────────────────────────────────────────────────────
def pgroup_colors(groups: list[str]) -> dict[str, str]:
    return {g: PARAM_GROUP_COLORS[i % len(PARAM_GROUP_COLORS)]
            for i, g in enumerate(groups)}

def ogroup_colors(groups: list[str]) -> dict[str, str]:
    return {g: OBS_GROUP_COLORS[i % len(OBS_GROUP_COLORS)]
            for i, g in enumerate(groups)}

def desaturate(hex_color: str, factor: float = 0.5) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    L = 0.299*r + 0.587*g + 0.114*b
    return "#{:02x}{:02x}{:02x}".format(
        int((r + factor*(L-r))*255),
        int((g + factor*(L-g))*255),
        int((b + factor*(L-b))*255),
    )

def darken(hex_color: str, factor: float = 0.6) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(r*factor), int(g*factor), int(b*factor))

# ── Reusable UI components ────────────────────────────────────────────────────
def section_header(title: str, subtitle: str | None = None):
    return dmc.Stack(gap=0, mb="xs", children=[
        dmc.Text(title, fw=600, size="sm",
                 style={"color": "#333", "letterSpacing": "0.02em"}),
        dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else html.Span(),
    ])

def kpi_card(label: str, value: str, unit: str = "",
             icon: str | None = None, color: str = "teal",
             delta: float | None = None):
    delta_el = []
    if delta is not None:
        sign = "▲" if delta > 0 else "▼"
        clr  = "red" if delta > 0 else "teal"
        delta_el = [dmc.Text(f"{sign} {abs(delta):.1f}%", size="xs", c=clr, mt=2)]
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
                *delta_el,
            ]),
            dmc.ThemeIcon(
                DashIconify(icon=icon, width=18),
                size="lg", radius="md", variant="light", color=color,
            ) if icon else html.Span(),
        ])],
    )

def page_header(title: str, subtitle: str, right_controls=None):
    return dmc.Group(
        justify="space-between", align="flex-end", mb="xs",
        children=[
            dmc.Stack(gap=2, children=[
                dmc.Title(title, order=3, style={"color": "#1a1a1a"}),
                dmc.Text(subtitle, size="sm", c="dimmed"),
            ]),
            dmc.Group(gap="sm", children=right_controls or []),
        ],
    )

def insight_alert(items: list[str], title: str = "Diagnostic notes"):
    return dmc.Alert(
        title=title, color="teal", variant="light", radius="md",
        icon=DashIconify(icon="tabler:bulb", width=18),
        children=dmc.List(
            size="sm", spacing="xs",
            children=[dmc.ListItem(i) for i in items],
        ),
    )
