"""Shared sidebar navigation component."""
from dash import html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

NAV_ITEMS = [
    ("/",              "Objective Function", "tabler:target"),
    ("/residuals",     "Residuals",          "tabler:chart-scatter"),
    ("/contribution",  "Contribution",       "tabler:grid-dots"),
    ("/sensitivity",   "Sensitivity",        "tabler:antenna"),
    ("/identifiability","Identifiability",   "tabler:fingerprint"),
    ("/regularisation","Regularisation",     "tabler:adjustments"),
    ("/convergence",   "Convergence",        "tabler:trending-down"),
    ("/data-worth",    "Data Worth",         "tabler:database"),
    ("/uncertainty",   "Uncertainty",        "tabler:chart-area"),
]

def sidebar(pathname="/"):
    return dmc.Stack(
        gap=4, p="sm",
        style={"width":"200px","minHeight":"100vh",
               "background":"#f8f9fb","borderRight":"1px solid #eee"},
        children=[
            dmc.Text("PEST Diagnostics", fw=700, size="sm",
                     style={"color":"#1a1a1a","letterSpacing":"0.02em",
                            "padding":"12px 8px 4px"}),
            dmc.Divider(mb=4),
            *[dmc.NavLink(
                label=label,
                href=href,
                leftSection=DashIconify(icon=icon, width=16),
                active=pathname==href,
                style={"borderRadius":"8px","fontSize":"13px"},
                styles={"root":{"padding":"7px 10px"},
                        "label":{"fontWeight":"500" if pathname==href else "400"}},
            ) for href, label, icon in NAV_ITEMS],
        ]
    )
