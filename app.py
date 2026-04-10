"""
This app creates a simple sidebar layout using inline style arguments and the
dbc.Nav component.

dcc.Location is used to track the current location, and a callback uses the
current location to render the appropriate page content. The active prop of
each NavLink is set automatically according to the current pathname. To use
this feature you must install dash-bootstrap-components >= 0.11.0.

For more details on building multi-page Dash applications, check out the Dash
documentation: https://dash.plotly.com/urls
"""

import dash
import dash_mantine_components as dmc
from dash import Input, Output, dcc, html, callback, State
from module import phiupdate, parupdate, parsen, parcor, obssen, parident, uploadpest, obssim, obsinfl


stylesheets = [
    "https://unpkg.com/@mantine/dates@7/styles.css",
    "https://unpkg.com/@mantine/code-highlight@7/styles.css",
    "https://unpkg.com/@mantine/charts@7/styles.css",
    "https://unpkg.com/@mantine/carousel@7/styles.css",
    "https://unpkg.com/@mantine/notifications@7/styles.css",
    "https://unpkg.com/@mantine/nprogress@7/styles.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
]

scripts = [
    "https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.10.8/dayjs.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.10.8/locale/ru.min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/dayjs/1.10.8/locale/fr.min.js",
    "https://www.googletagmanager.com/gtag/js?id=G-4PJELX1C4W",
    "https://media.ethicalads.io/media/client/ethicalads.min.js",
    "https://unpkg.com/hotkeys-js/dist/hotkeys.min.js",
]

app = dash.Dash(
    __name__,
    title="SSPA|PEST_HP_ANALYTICS",
    # use_pages=True,
    external_scripts=scripts,
    external_stylesheets=stylesheets,
    suppress_callback_exceptions=True
)

sspa_logo = dmc.Anchor(dmc.Image(
    radius="md",
    w=506,
    src="https://sspa.com/wp-content/uploads/2021/05/SSPA_LogoTagline_blue_864x95.png",),
    href="https://www.sspa.com",
)

apps = {
    "uploadpest":"Upload Data",
    "phiupdate":"Phi Evolution",
    "parupdate":"Parameter Evolution",
    "parcor":"Parameter Correlation",
    "parsen":"Parameter Sensitivity",
    "parident":"Parameter Identifiability",
    "obssen":"Observation Sensitivity",
    "obssim":"Observation Similarity",
    "obsinfl":"Observation Influence",
}


layout = dmc.AppShell(
    [
        dmc.AppShellHeader(dmc.Group([sspa_logo, dmc.Title('PEST_ANALYTICS', order=2)]), px=80, py=10),
        # dmc.AppShellNavbar(
        #     id="navbar",
        #     children=[
        #         dmc.NavLink(label="Home", href="/"),
        #         dmc.NavLink(label="Parameter Update", href="/parupdate"),
        #         dmc.NavLink(label="App2", href="/app2"),
        #     ],
        #     p="md",
        # ),
        dmc.AppShellMain([
            html.H4('This is Status', id="home-status", style={"padding-left": 205}),
            dmc.Tabs(
                [
                    dmc.TabsList([dmc.TabsTab(v, value=k, size="large", style={"display": "block"}) for k, v in apps.items()] ),] + [
                    dmc.TabsPanel(eval(k+".layout"), value=k, id="tab-"+k, style={"padding-left":20}) for k, v in apps.items()
                ],
                # color="red", # default is blue
                orientation="vertical", # or "vertical"
                variant="default", # or "outline" or "pills"
                value="uploadpest", # default value on first render
                id="main-tabs",
            ),
        ]),
    ],
    header={
        "height": {"base": 60, "md": 70, "lg": 80},
    },
    # navbar={
    #     "width": {"base": 200, "md": 300, "lg": 400},
    #     "breakpoint": "sm",
    #     "collapsed": {"mobile": True},
    # },
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(layout)


# @callback(
#     Output("appshell", "navbar"),
#     Input("burger", "opened"),
#     State("appshell", "navbar"),
# )
# def toggle_navbar(opened, navbar):
#     navbar["collapsed"] = {"mobile": not opened}
#     return navbar



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
