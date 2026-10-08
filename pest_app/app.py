"""
PEST Calibration Diagnostic App
================================
Entry point.  Run with:  python app.py

File structure
--------------
pest_app/
├── app.py                   ← this file
├── data/
│   └── pest_reader.py       ← PEST file parser
├── pages/
│   ├── objective_function.py
│   ├── residuals.py
│   ├── contribution.py
│   ├── sensitivity.py
│   ├── identifiability.py
│   ├── regularisation.py
│   ├── convergence.py
│   ├── data_worth.py
│   └── uncertainty.py
├── components/
│   └── sidebar.py
└── assets/
    └── custom.css

Usage with real PEST data
-------------------------
    from data.pest_reader import read_pest_run
    PEST_DATA = read_pest_run("/path/to/your/pest_run")

Then pass PEST_DATA into each page's layout() call inside the routing callback.
"""

import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# ── Page imports ──────────────────────────────────────────────────────────────
from pages.objective_function import layout as obj_layout,   register_callbacks as obj_cb
from pages.residuals           import layout as res_layout,   register_callbacks as res_cb
from pages.contribution        import layout as con_layout,   register_callbacks as con_cb
from pages.sensitivity         import layout as sen_layout,   register_callbacks as sen_cb
from pages.identifiability     import layout as ide_layout,   register_callbacks as ide_cb
from pages.regularisation      import layout as reg_layout,   register_callbacks as reg_cb
from pages.convergence         import layout as conv_layout,  register_callbacks as conv_cb
from pages.data_worth          import layout as dw_layout,    register_callbacks as dw_cb
from pages.uncertainty         import layout as unc_layout,   register_callbacks as unc_cb
from components.sidebar        import sidebar, NAV_ITEMS

# ── Optional: load real PEST data ─────────────────────────────────────────────
# Uncomment and set your path to use real files instead of mock data:
#
# from data.pest_reader import read_pest_run
# PEST_DATA = read_pest_run("/path/to/your/pest_run_basename")
#
PEST_DATA = None   # None → each page falls back to its own mock data


# ── App init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700"
        "&family=DM+Mono&display=swap",
    ],
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}],
)
app.title = "PEST Calibration Diagnostics"
server = app.server   # for gunicorn/deployment


# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dmc.MantineProvider(
    theme={
        "fontFamily": "DM Sans, sans-serif",
        "primaryColor": "teal",
        "components": {
            "Paper":  {"defaultProps": {"shadow":"xs"}},
            "Title":  {"defaultProps": {"c":"#1a1a1a"}},
        },
    },
    children=html.Div([
        dcc.Location(id="url", refresh=False),

        # Top bar
        html.Div(
            style={"background":"#1a1a2e","padding":"0 24px",
                   "display":"flex","alignItems":"center","height":"52px",
                   "borderBottom":"1px solid #2a2a4a","position":"sticky",
                   "top":0,"zIndex":1000},
            children=[
                DashIconify(icon="tabler:chart-bar", width=22,
                             style={"color":"#4fc3f7","marginRight":"10px"}),
                html.Span("PEST Calibration Diagnostics",
                          style={"color":"white","fontWeight":"700","fontSize":"15px",
                                 "letterSpacing":"0.02em","fontFamily":"DM Sans, sans-serif"}),
                html.Div(style={"flex":1}),
                dmc.Badge("v1.0", color="teal", variant="light", size="sm"),
            ],
        ),

        # Body: sidebar + main content
        html.Div(
            style={"display":"flex","minHeight":"calc(100vh - 52px)"},
            children=[
                # Sidebar
                html.Div(id="sidebar-container",
                         style={"flexShrink":0}),

                # Main content
                html.Div(
                    id="page-content",
                    style={"flex":1,"overflowY":"auto","background":"#ffffff",
                           "minHeight":"calc(100vh - 52px)"},
                ),
            ],
        ),
    ]),
)


# ── Routing callback ──────────────────────────────────────────────────────────
@app.callback(
    Output("page-content",       "children"),
    Output("sidebar-container",  "children"),
    Input("url", "pathname"),
)
def route(pathname):
    nav = sidebar(pathname or "/")
    d = PEST_DATA   # None → each page uses its own mock data

    routes = {
        "/":               lambda: obj_layout(data=d),
        "/residuals":      lambda: res_layout(data=d),
        "/contribution":   lambda: con_layout(data=d),
        "/sensitivity":    lambda: sen_layout(data=d),
        "/identifiability":lambda: ide_layout(data=d),
        "/regularisation": lambda: reg_layout(data=d),
        "/convergence":    lambda: conv_layout(data=d),
        "/data-worth":     lambda: dw_layout(data=d),
        "/uncertainty":    lambda: unc_layout(data=d),
    }

    fn = routes.get(pathname, routes["/"])
    return fn(), nav


# ── Register all callbacks ────────────────────────────────────────────────────
for cb in [obj_cb, res_cb, con_cb, sen_cb, ide_cb, reg_cb, conv_cb, dw_cb, unc_cb]:
    cb(app)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
