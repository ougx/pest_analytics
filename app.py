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
# dash-mantine-components >= 0.14 needs React 18; Dash 2.x defaults to React 16
dash._dash_renderer._set_react_version("18.2.0")
from dash import Input, Output, dcc, html, callback, State
from module import apps, sections


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

TITLE = "PEST_HP_ANALYTICS"

app = dash.Dash(
    __name__,
    title="SSPA|" + TITLE,
    # use_pages=True,
    external_scripts=scripts,
    external_stylesheets=stylesheets,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
)

sspa_logo = dmc.Anchor(dmc.Image(
    radius="md",
    w=506,
    src=app.get_asset_url("SSPA_LogoTagline_blue_864x95.png"),),
    href="https://www.sspa.com",
)

def nav_items():
    """tab list grouped into sections, each with a small heading"""
    items = []
    for section, pages in sections.items():
        items.append(dmc.Text(section, size="xs", fw=700, c="dimmed", tt="uppercase", mt="sm", mb=4, px="sm"))
        items += [dmc.TabsTab(v.fullname, value=k, size="large", style={"display": "block"}) for k, v in pages.items()]
    return items


layout = dmc.AppShell(
    [
        dmc.AppShellHeader(dmc.Group([sspa_logo, dmc.Title(TITLE, order=2)]), px=80, py=10),
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

            dmc.Tabs(
                [
                    dmc.TabsList(nav_items()),] + [
                    dmc.TabsPanel(v.layout, value=k, id="tab-"+k, style={"padding-left":20}) for k, v in apps.items()
                ],
                # color="red", # default is blue
                orientation="vertical", # or "vertical"
                variant="default", # or "outline" or "pills"
                value="uploadpest", # default value on first render
                id="main-tabs",
            ),
        ]),
        dmc.AppShellFooter(dmc.Text("No data loaded.", id="home-status", size="sm", fw=600), px="md", py=8),
    ],
    header={
        "height": {"base": 60, "md": 70, "lg": 80},
    },
    footer={"height": 36},
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
    import os
    # 8050 falls inside a Windows reserved port range (8013-8112, Hyper-V/WSL);
    # check with: netsh interface ipv4 show excludedportrange protocol=tcp
    port = int(os.environ.get("PORT", 8888))
    app.run(host="0.0.0.0", port=port, debug=True)
