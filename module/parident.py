from dash import html, dcc, Input, Output, State, callback, no_update
import dash_mantine_components as dmc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from utils import parse_zip, isiterable
from layout import update_niter, about

name = __name__.lstrip("module.") + "-"
fullname = "Parameter Identifiability"

md = dcc.Markdown('''

    Doherty and Hunt (2009) define the identifiability of a parameter as the square of the cosine
    between a vector pointing in the direction of the parameter and the projection of that vector
    onto the calibration solution space. This projection is zero where a parameter lies entirely
    within the null space; it is one when a parameter lies entirely within the solution space. If the
    identifiability of a parameter is zero, then its uncertainty is not reduced through the
    calibration process.

    The identifiability $f_j$ of parameter $j$ can be defined as:

    $$
                   f_j = (VV^T)_{jj}
    $$
    where $V$ is a matrix whose columns are orthogonal unit vectors that span the calibration solution space;
    the subscript $jj$ indicates the jth diagonal element of the matrix.
    $V$ can be obtained through singular value decomposition of the weighted parameter sensitivity matrix.

    **Reference:**

    Doherty, J., Hunt, R.J., 2009. Two statistics for evaluating parameter identifiability and error reduction. Journal of Hydrology 366, 119–127. https://doi.org/10.1016/j.jhydrol.2008.12.018

    ''',
    mathjax=True)

# However if its identifiability is unity, this does not mean that its
#    estimation takes place without any potential for error. What it does indicate is that all of its
#    potential for error is inherited from noise in the measurement dataset.

layout = html.Div([
    html.H3(fullname),
    about(md),
    dmc.Group([
        dmc.Select(id=name+"iter", label="Iteration:", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"svdtrunc", label="SVD Truncation:", data=["1e-4", "1e-5", "5e-6", "1e-6", "5e-7", "1e-7", "5e-8", "1e-8", "1e-9"], value="5e-7", style={"width": "150px"}),
        dmc.Select(id=name+"layout", label="Layout:", data=["Horizontal", "Vertical"], value="Vertical", style={"width": "150px"}),
        dmc.Button("Plot Identifiability", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    dcc.Loading(dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, )), style={"margin-bottom":"90px"})),
])




@callback(
    Output(name+"iter", "data"),
    Input( "pest-data", "data"),
)
def upload_pest(data):
    return update_niter(data, False, False, True)


@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"svdtrunc", "value"),
    State(name+"layout", "value"),
)
def plot(clicks, data, iiter, svdtrunc, layout):
    if clicks is None or clicks == 0:
        return px.scatter(title="Upload data and click 'Plot Identifiability' to see plot")
    if data is None or data == {}:
        return px.scatter(title="No data uploaded. Please upload data and click 'Plot Identifiability' to see plot")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, True)
    if obs is None:
        return px.scatter(title="No .pst file found in zip file.")
    if not jcos:
        return px.scatter(title="No .jco file found in zip file.")

    csplimit = 1e-10
    nobs_useful = (obs.weight > 0).sum()

    jco0 = jcos[int(iiter)]
    mask = jco0.index.isin(obs.obsname)
    jco = jco0.loc[mask, ]
    WJ = jco * obs.set_index("obsname").loc[jco.index, "weight"].values[:, None]
    csp = (WJ**2).sum() ** 0.5 / nobs_useful
    mask = WJ.columns.isin(csp[csp>csplimit].index)
    JJ = WJ.loc[:,mask]


    U, s, Vt = np.linalg.svd(JJ, full_matrices=False)
    V = Vt.T
    sr = s / s[0]
    V1 = V[:,sr>=float(svdtrunc)]

    V1 = pd.DataFrame(V1, index=JJ.columns, columns=['eig'+str(i+1) for i in range(V1.shape[1])])
    f = np.diag(V1 @ V1.T)
    f = pd.Series(f, index=JJ.columns)
    fig = px.bar(
        x=f.index if layout=="Horizontal" else f,
        y=f if layout=="Horizontal" else f.index,
    )

    if layout == "Horizontal":
        fig.update_layout(
            height=700,
            xaxis_title="Parameter",
            yaxis_title="Parameter Identifiability",
        )
    else:
        fig.update_layout(
            height=len(f) * 18,
            yaxis_title="Parameter",
            xaxis_title="Parameter Identifiability",
        )
        fig.update_yaxes(autorange="reversed")
    return fig
