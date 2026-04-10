import dash
from dash import html, dcc, Input, Output, callback, State
import dash_mantine_components as dmc
from utils import parse_zip

name = __name__.lstrip("module.") + "-"


upload = dcc.Upload(
    id="upload-pest",
    children="Drag and drop a Zip file containing PEST files (*.pst, *.rei, *.par, *.jco)",
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
    upload,
    dcc.Store(id="pest-data", storage_type="memory"),
])


@callback(
    Output("pest-data", "data"),
    Input("upload-pest", "filename"),
    Input("upload-pest", "contents"),
)
def update_pest_data(filename, contents):
    if filename and contents:
        # print({"filename": filename, "contents": contents})
        return {"filename": filename, "contents": contents}
    return {}


@callback(
    Output("home-status", "children"),
    Output("main-tabs", "value"),
    Input("pest-data", "data"),
)
def upload_pest(data):
    if data is None or data == {}:
        return [html.Div(["No file uploaded. Please upload a zip file containing PEST files.",
             html.Br(),
             "Optional obs.csv[obsname,obsgp] and par.csv[parname,pargp] in the zip file can be used to reassign observation and parameter groups.",
             html.Br(),
             "Phi calculated from `.rei` files. Initial Phi value is calculated from `.hp` file or `.rei.0` file."
        ],), "uploadpest"]
    else:
        filename = data.get("filename")
        contents = data.get("contents")
        pst, par, obs, pars, jcos, reis = parse_zip(contents, False, False, False)
        return [f"{filename} Uploaded. Analyzing {pst}...", "phiupdate"]
