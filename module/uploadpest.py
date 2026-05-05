import dash
from dash import html, dcc, Input, Output, callback, State
import dash_mantine_components as dmc
from utils import parse_zip

name = __name__.lstrip("module.") + "-"
fullname = "Upload Data"


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
    dcc.Store(id="pest-data", storage_type="memory"),
])


@callback(
    Output("pest-data", "data"),
    Input("btn-parsedata", "n_clicks"),
    State("upload-pest", "filename"),
    State("upload-pest", "contents"),
    State("pest-path", "value"),
    prevent_initial_callback=False,
)
def update_pest_data(nclick, filename, contents, pestpath):
    if nclick:
        return {"filename": filename, "contents": contents, "pestpath":pestpath}



@callback(
    Output("home-status", "children"),
    Output("main-tabs", "value"),
    Input("pest-data", "data"),
)
def upload_pest(data):
    if data:
        pst, par, obs, pars, jcos, reis = parse_zip(data, False, False, False)
        if pst:
            return [f"Zipfile successfully parsed. Analyzing {pst}...", "phiupdate"]
    return [html.Div(["No file uploaded or specified. Please specify or upload a zip file containing PEST files.",
        html.Br(),
        "Optional obs.csv[obsname,obsgp] and par.csv[parname,pargp] in the zip file can be used to reassign observation and parameter groups.",
        html.Br(),
        "Phi calculated from `.rei` files. Initial Phi value is calculated from `.hp` file or `.rei.0` file."
    ],), "uploadpest"]
