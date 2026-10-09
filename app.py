"""
PEST_HP Analytics: a Dash + dash-mantine-components app for post-processing PEST_HP runs.
Pages are in module/ (grouped into navigation sections in module/__init__.py).

Run for development with `python app.py` (PORT env var, default 8888) or serve with
waitress (waitress.bat). The app loads no third-party scripts or stylesheets, so it
works on an offline network; only the Residual Map basemap tiles come from the internet.
"""

import dash
import dash_mantine_components as dmc
# dash-mantine-components >= 0.14 needs React 18; Dash 2.x defaults to React 16
dash._dash_renderer._set_react_version("18.2.0")
from module import apps, sections


TITLE = "PEST_HP_ANALYTICS"

app = dash.Dash(
    __name__,
    title="SSPA|" + TITLE,
    # use_pages=True,
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
