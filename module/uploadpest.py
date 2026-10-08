import dash
from dash import html, dcc, Input, Output, callback, clientside_callback, State, no_update
import dash_mantine_components as dmc
from utils import parse_zip, save_upload

name = __name__.lstrip("module.") + "-"
fullname = "Upload Data"


UPLOAD_PROMPT = "Drag and drop a Zip file containing PEST files (*.pst, *.rei, *.par, *.jco)"

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
        "Specify or upload a zip file containing PEST files (*.pst, *.rei, *.par, *.jco, *.hp).",
        html.Br(),
        "Optional obs.csv [obsname, grp*, time] and par.csv [parname, grp*] in the zip file can be used to regroup observations and parameters.",
        html.Br(),
        "Phi is calculated from the .rei files; the initial Phi is calculated from the .hp file or the .rei.0 file.",
    ], size="sm", c="dimmed"),
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
    return [f"Loaded {pst}", "phiupdate", message(summary, "green")]
