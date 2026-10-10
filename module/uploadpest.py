from dash import html, dcc, Input, Output, callback, clientside_callback, State, no_update, ALL, ctx
import dash_mantine_components as dmc
import os
from utils import parse_zip, save_upload, read_sites, check_pest_files
from geo import read_layers, guess_category, LAYER_CATEGORIES

name = __name__.removeprefix("module.") + "-"
fullname = "Upload Data"


UPLOAD_PROMPT = "Drag and drop a Zip file containing PEST files (*.pst, *.rei, *.par, *.jco)"
LAYER_PROMPT = ("Drag and drop zipped shapefiles (.shp with its .shx, .dbf and .prj) or GeoJSON files "
                "to show on the Residual Map; several files at once are fine")

upload = dcc.Upload(
    id="upload-pest",
    children=UPLOAD_PROMPT,
    style={
        "border": "1px dashed",
        "width": "100%",
        "height": "60px",
        "padding": "20px",
        "textAlign": "center",
        "color": "#228be6",
    },
    # className="mx-auto",
    # multiple=True,  # enables folder drag-and-drop
    # enable_folder_selection=True,  # changes the file picker to a folder picker
    # accept=".png",
)# dmc.Button(, variant="outline", justify="center", fullWidth=True, size="lg", color="blue", styles={"label": {"width": "100%"}},)



layout = html.Div([
    # html.H4('This is our Parameter Update page', id=name+"home-status"),
    dmc.TextInput(
        # props as configured above:
        placeholder=r"zip file path, e.g. \\file2019\Projects\MichaelOu\PEST\napa2026_04.zip",
        label="Specify Path to PEST Zipfile on SSP&A Server",
        # description="Enter your full name as it appears on your card",
        required=True,
        id='pest-path',
        persistence=True
        # value=r"C:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1744-PecoModelUpdate\06_Model\Peco_Calib_20260316.zip"
        # other props...
    ),
    html.H3("or", style={"text-align": "center"}),
    upload,
    dmc.Button("Load Data", id="btn-parsedata", my=20, size="large", fullWidth=True),
    html.Div(id="upload-message"),
    dmc.Text([
        "Specify or upload a zip file of PEST_HP (or PEST) output: case.pst, the per-iteration case.rei.N, case.par.N and "
        "case.jco.N files, and case.hp (with PEST, set REISAVEITN, PARSAVEITN and JCOSAVEITN to save them). "
        "PEST++ output (e.g. .jcb, case.N.par) is not read.",
        html.Br(),
        "Optional obs.csv [obsname, grp*, time] and par.csv [parname, grp*] in the zip file can be used to regroup observations and parameters.",
        html.Br(),
        "Phi is calculated from the .rei files; the initial Phi is calculated from the .hp file or the .rei.0 file.",
        html.Br(),
        "Optional site.csv [site, lat, lon, ...] in the zip file locates the sites (e.g. grpSite values) on the Residual Map.",
    ], size="sm", c="dimmed"),
    html.H4("Map layers (optional)", style={"margin-top": "30px"}),
    dcc.Upload(
        id="upload-layers",
        children=LAYER_PROMPT,
        multiple=True,
        accept=".zip,.geojson,.json",
        style={"border": "1px dashed", "width": "100%", "padding": "20px", "textAlign": "center", "color": "#228be6"},
    ),
    html.Div(id="layer-message", style={"margin-top": "10px"}),
    dmc.Group([
        html.Div(id="layer-list"),
        dmc.Button("Clear layers", id="btn-clear-layers", size="xs", variant="subtle", color="red"),
    ], justify="space-between", mt=10),
    html.H4("Runs to compare (optional)", style={"margin-top": "30px"}),
    dmc.Text("Other runs to compare with the loaded run (the reference) on the Compare pages. "
             "Observations, parameters and prior information are matched by name.", size="sm", c="dimmed"),
    dmc.Group([
        dmc.TextInput(id="compare-path", placeholder="zip file path of another run", style={"flex": 1}),
        dmc.Button("Add run", id="btn-add-compare", variant="outline"),
    ], mt=10),
    dcc.Upload(
        id="upload-compare",
        children="or drop the zip file of another run here",
        accept=".zip",
        style={"border": "1px dashed", "width": "100%", "padding": "14px", "textAlign": "center",
               "color": "#228be6", "margin-top": "10px"},
    ),
    html.Div(id="compare-message", style={"margin-top": "10px"}),
    dmc.Group([
        html.Div(id="compare-list"),
        dmc.Button("Clear runs", id="btn-clear-compare", size="xs", variant="subtle", color="red"),
    ], justify="space-between", mt=10),
    dcc.Store(id="compare-runs", storage_type="memory"),
    dcc.Store(id="compare-runs-saved", storage_type="memory"),
    dcc.Store(id="map-layers", storage_type="memory"),
    dcc.Store(id="map-layers-saved", storage_type="memory"),
    # pages listen to pest-data. Its small reference (a path or upload key) is also saved in the
    # browser's sessionStorage, so a page reload in this browser tab restores the loaded data.
    dcc.Store(id="pest-data", storage_type="memory"),
    dcc.Store(id="pest-saved", storage_type="memory"),
    dcc.Location(id="pest-url"),
])


@callback(
    Output("pest-data", "data"),
    Input("btn-parsedata", "n_clicks"),
    State("upload-pest", "filename"),
    State("upload-pest", "contents"),
    State("pest-path", "value"),
)
def update_pest_data(nclick, filename, contents, pestpath):
    # the uploaded file is saved on the server; the browser only keeps its key
    if not nclick:
        return no_update
    return {"filename": filename, "upload": save_upload(contents) if contents else None, "pestpath":pestpath}


# (dcc.Store's own session storage is reset to its initial value at start-up here, so the
# reference is saved and restored with sessionStorage directly)
clientside_callback(
    """function (data) {
        if (data) { window.sessionStorage.setItem("pest-data", JSON.stringify(data)); }
        return window.dash_clientside.no_update;
    }""",
    Output("pest-saved", "data"),
    Input("pest-data", "data"),
)

clientside_callback(
    """function (pathname) {
        const saved = window.sessionStorage.getItem("pest-data");
        return saved ? JSON.parse(saved) : window.dash_clientside.no_update;
    }""",
    Output("pest-data", "data", allow_duplicate=True),
    Input("pest-url", "pathname"),
    prevent_initial_call="initial_duplicate",  # runs on page load
)



# map layers: same pattern, saved separately so layers can be added without reloading the PEST data
clientside_callback(
    """function (layers) {
        if (layers !== null && layers !== undefined) {
            window.sessionStorage.setItem("map-layers", JSON.stringify(layers));
        }
        return window.dash_clientside.no_update;
    }""",
    Output("map-layers-saved", "data"),
    Input("map-layers", "data"),
)

clientside_callback(
    """function (pathname) {
        const saved = window.sessionStorage.getItem("map-layers");
        return saved ? JSON.parse(saved) : window.dash_clientside.no_update;
    }""",
    Output("map-layers", "data", allow_duplicate=True),
    Input("pest-url", "pathname"),
    prevent_initial_call="initial_duplicate",  # runs on page load
)


@callback(
    Output("map-layers", "data"),
    Output("layer-message", "children"),
    Output("upload-layers", "contents"),
    Input("upload-layers", "contents"),
    State("upload-layers", "filename"),
    State("map-layers", "data"),
    running=[(Output("upload-layers", "disabled"), True, False)],
)
def add_layers(contents, filenames, layers):
    """save uploaded layer files on the server and list their layers; the browser keeps only keys"""
    if not contents:
        return no_update, no_update, no_update
    layers = list(layers or [])
    errors = []
    for content, filename in zip(contents, filenames):
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        if ext not in ("zip", "geojson", "json"):
            errors.append(f"{filename}: not a .zip or .geojson file")
            continue
        try:
            key = save_upload(content, ext)
            found = read_layers(key, ext)
        except Exception as e:
            errors.append(f"{filename}: {e}")
            continue
        for i, layer in enumerate(found):
            name = layer["name"] if ext == "zip" else os.path.splitext(filename)[0]
            entry = {"key": key, "ext": ext, "index": i, "name": name, "kind": layer["kind"], "count": layer["count"],
                     "category": guess_category(name, layer["kind"])}
            layers = [l for l in layers if (l["key"], l["index"]) != (key, i)] + [entry]
    msg = [message(e, "red") for e in errors] or None
    return layers, msg, None   # clear the drop zone so the same file can be dropped again


@callback(
    Output("map-layers", "data", allow_duplicate=True),
    Output("layer-message", "children", allow_duplicate=True),
    Input("btn-clear-layers", "n_clicks"),
    prevent_initial_call=True,
)
def clear_layers(n_clicks):
    return [], None


@callback(
    Output("layer-list", "children"),
    Input("map-layers", "data"),
)
def show_layers(layers):
    """one row per layer with its category, which sets its map style (geo.LAYER_CATEGORIES)"""
    if not layers:
        return dmc.Text("No map layers loaded.", size="sm", c="dimmed")
    kinds = {"fill": "polygon", "line": "line", "circle": "point"}
    rows = []
    for i, l in enumerate(layers):
        described = f"{l['name']} ({l['count']} {kinds[l['kind']]}{'' if l['count'] == 1 else 's'})"
        rows.append(dmc.Group([
            dmc.Select(id={"type": "layer-category", "index": i}, data=list(LAYER_CATEGORIES),
                       value=l.get("category") or guess_category(l["name"], l["kind"]), size="xs", w=150),
            dmc.Text(described, size="sm"),
        ], gap="sm"))
    return dmc.Stack([dmc.Text("Layers (the category sets the map style):", size="sm", fw=500)] + rows, gap=4)


@callback(
    Output("map-layers", "data", allow_duplicate=True),
    Input({"type": "layer-category", "index": ALL}, "value"),
    State("map-layers", "data"),
    prevent_initial_call=True,
)
def set_layer_categories(categories, layers):
    if not layers or len(categories) != len(layers):
        return no_update
    if all(l.get("category") == c for l, c in zip(layers, categories)):
        return no_update   # nothing changed (e.g. the list was just redrawn)
    return [{**l, "category": c} for l, c in zip(layers, categories)]


# runs to compare: same pattern, a list of run references ({"name", "filename", "upload", "pestpath"})
clientside_callback(
    """function (runs) {
        if (runs !== null && runs !== undefined) {
            window.sessionStorage.setItem("compare-runs", JSON.stringify(runs));
        }
        return window.dash_clientside.no_update;
    }""",
    Output("compare-runs-saved", "data"),
    Input("compare-runs", "data"),
)

clientside_callback(
    """function (pathname) {
        const saved = window.sessionStorage.getItem("compare-runs");
        return saved ? JSON.parse(saved) : window.dash_clientside.no_update;
    }""",
    Output("compare-runs", "data", allow_duplicate=True),
    Input("pest-url", "pathname"),
    prevent_initial_call="initial_duplicate",  # runs on page load
)


@callback(
    Output("compare-runs", "data", allow_duplicate=True),
    Output("compare-message", "children"),
    Output("upload-compare", "contents"),
    Input("btn-add-compare", "n_clicks"),
    Input("upload-compare", "contents"),
    State("upload-compare", "filename"),
    State("compare-path", "value"),
    State("compare-runs", "data"),
    prevent_initial_call=True,
    running=[(Output("btn-add-compare", "loading"), True, False)],
)
def add_compare_run(n_clicks, contents, filename, path, runs):
    """check that the run can be read and add its reference to the list"""
    if ctx.triggered_id == "upload-compare":
        if not contents:
            return no_update, no_update, no_update
        ref = {"filename": filename, "upload": save_upload(contents), "pestpath": None}
    else:
        if not path:
            return no_update, message("Enter the zip file path of another run first.", "yellow"), no_update
        ref = {"filename": None, "upload": None, "pestpath": path.strip().strip('"')}
    runs = list(runs or [])
    if any((r.get("upload"), r.get("pestpath")) == (ref["upload"], ref["pestpath"]) for r in runs):
        return no_update, message("That run is already in the list.", "yellow"), None
    where = ref["pestpath"] or filename
    try:
        pst = parse_zip(ref, False, False, False)[0]
    except Exception as e:
        return no_update, message(f"Could not read {where}: {e}", "red"), None
    if pst is None:
        return no_update, message(f"File not found: {where}", "red"), None
    if pst == "":
        return no_update, message(f"No .pst file found in {where}.", "red"), None
    return runs + [{**ref, "name": pst}], message(f"Added {pst}.", "green"), None


@callback(
    Output("compare-runs", "data", allow_duplicate=True),
    Input({"type": "compare-remove", "index": ALL}, "n_clicks"),
    State("compare-runs", "data"),
    prevent_initial_call=True,
)
def remove_compare_run(n_clicks, runs):
    if not runs or not ctx.triggered_id or not any(n_clicks):
        return no_update   # the list was just redrawn
    i = ctx.triggered_id["index"]
    return [r for j, r in enumerate(runs) if j != i]


@callback(
    Output("compare-runs", "data", allow_duplicate=True),
    Output("compare-message", "children", allow_duplicate=True),
    Input("btn-clear-compare", "n_clicks"),
    prevent_initial_call=True,
)
def clear_compare_runs(n_clicks):
    return [], None


@callback(
    Output("compare-list", "children"),
    Input("compare-runs", "data"),
)
def show_compare_runs(runs):
    if not runs:
        return dmc.Text("No runs to compare.", size="sm", c="dimmed")
    rows = [dmc.Group([
        dmc.Button("Remove", id={"type": "compare-remove", "index": i}, size="compact-xs", variant="subtle", color="red"),
        dmc.Text(f"{r['name']}  ({r.get('pestpath') or r.get('filename')})", size="sm"),
    ], gap="sm") for i, r in enumerate(runs)]
    return dmc.Stack(rows, gap=4)


@callback(
    Output("upload-pest", "children"),
    Input("upload-pest", "filename"),
)
def show_upload_name(filename):
    if filename:
        return f"Selected: {filename} (click Load Data, or drop another file to replace it)"
    return UPLOAD_PROMPT


def message(text, color):
    return dmc.Alert(text, color=color, variant="light", mb=10)


@callback(
    Output("home-status", "children"),
    Output("main-tabs", "value"),
    Output("upload-message", "children"),
    Input("pest-data", "data"),
    running=[(Output("btn-parsedata", "loading"), True, False)],
)
def upload_pest(data):
    if not data:
        return ["No data loaded.", "uploadpest", None]
    name = data.get("pestpath") or data.get("filename") or "the zip file"
    try:
        pst, par, obs, pars, reis, jcos = parse_zip(data, True, True, False)
    except Exception as e:
        return ["No data loaded.", "uploadpest", message(f"Could not read {name}: {e}", "red")]
    if pst is None:
        if data.get("pestpath") and not data.get("upload"):
            return ["No data loaded.", "uploadpest", message(f"File not found: {data['pestpath']}", "red")]
        return ["No data loaded.", "uploadpest", message("Specify a path or drop a zip file first.", "yellow")]
    if pst == "":
        return ["No data loaded.", "uploadpest", message(f"No .pst file found in {name}.", "red")]
    summary = (f"Loaded {pst}: {len(obs)} observations ({int((obs.weight > 0).sum())} weighted), "
               f"{len(par)} parameters, {len(pars)} iterations.")
    messages = [message(summary, "green")]
    try:
        sites = read_sites(data)
        if sites is not None:
            messages[0] = message(summary + f" site.csv: {len(sites)} sites with coordinates.", "green")
    except ValueError as e:
        messages.append(message(f"site.csv was not used: {e}", "yellow"))
    messages += [message(w, "yellow") for w in check_pest_files(data)]
    return [f"Loaded {pst}", "phiupdate", messages]
