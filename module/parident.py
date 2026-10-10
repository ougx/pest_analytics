from dash import html, dcc, Input, Output, State, callback
import dash_mantine_components as dmc
import plotly.express as px
from utils import parse_zip
import numpy as np
import plotly.graph_objects as go
from pst import get_identifiability, eigenvalue_ratios
from plotting import message_figure
from layout import update_weights, update_jco_iter, about, graph

name = __name__.removeprefix("module.") + "-"
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
    The solution space keeps the singular vectors whose eigenvalue of $J^TQJ$ (the squared singular value of $Q^{1/2}J$)
    is at least the selected fraction of the largest one, like PEST's `EIGTHRESH` (default 5e-7).
    The eigenvalue spectrum above the bars shows where the cutoff falls: a clear drop in the spectrum is a natural
    place for it; a cutoff in a smooth part of the spectrum makes identifiability sensitive to its exact value.

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
        dmc.Select(id=name+"iter", label="Jacobian Iteration:", description=".jco.N, at iteration N-1 parameters", data=["1"], value="1", style={"width": "150px"}),
        dmc.Select(id=name+"svdtrunc", label="Eigenvalue Ratio (EIGTHRESH):", data=["1e-4", "1e-5", "5e-6", "1e-6", "5e-7", "1e-7", "5e-8", "1e-8", "1e-9"], value="5e-7", style={"width": "150px"}),
        dmc.Select(id=name+"layout", label="Layout:", data=["Horizontal", "Vertical"], value="Vertical", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        dmc.Button("Plot Identifiability", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    graph(name+"spectrum", message_figure(), "Eigenvalue Spectrum"),
    graph(name+"plot", message_figure("Load data, then click 'Plot Identifiability'."), fullname + "", style={"margin-bottom":"90px"}),
])




@callback(
    Output(name+"iter", "data"),
    Output(name+"iter", "value"),
    Input( "pest-data", "data"),
)
def fill_iterations(data):
    return update_jco_iter(data)


@callback(
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Input( "pest-data", "data"),
)
def upload_weight_columns(data):
    return update_weights(data, "weight")


@callback(
    Output(name+"plot", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"svdtrunc", "value"),
    State(name+"layout", "value"),
    State( name+"weight", "value"),
)
def plot(clicks, data, iiter, svdtrunc, layout, weight="weight"):
    if clicks is None or clicks == 0:
        return message_figure("Load data, then click 'Plot Identifiability'.")
    if data is None or data == {}:
        return message_figure("No data loaded. Load data on the Upload Data page, then click 'Plot Identifiability'.")
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, True)
    if obs is None:
        return message_figure("No .pst file found in zip file.")
    if not jcos:
        return message_figure("No .jco file found in zip file.")

    f, nkept, nvec = get_identifiability(obs, jcos[int(iiter)], float(svdtrunc), weight_col=weight)
    fig = px.bar(
        x=f.index if layout=="Horizontal" else f,
        y=f if layout=="Horizontal" else f.index,
    )

    fig.update_layout(title=f"Solution space: {nkept} of {nvec} singular vectors "
                            f"(eigenvalue ratio >= {float(svdtrunc):g})")
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


@callback(
    Output(name+"spectrum", "figure"),
    Input( name+"update-plot", "n_clicks"),
    State( "pest-data", "data"),
    State( name+"iter", "value"),
    State( name+"svdtrunc", "value"),
    State( name+"weight", "value"),
)
def plot_spectrum(clicks, data, iiter, svdtrunc, weight="weight"):
    if not clicks or not data:
        return message_figure()
    pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, True)
    if obs is None or not jcos:
        return message_figure()
    return spectrum_figure(eigenvalue_ratios(obs, jcos[int(iiter)], weight_col=weight), float(svdtrunc), int(iiter))


def spectrum_figure(ratios, eigthresh, iiter):
    kept = ratios >= eigthresh
    fig = go.Figure()
    for mask, label, color in ((kept, "Solution space", "#1f77b4"), (~kept, "Null space", "#bbbbbb")):
        fig.add_trace(go.Scatter(x=ratios.index[mask], y=ratios[mask], mode="markers", name=f"{label} ({int(mask.sum())})",
                                 marker=dict(color=color, size=7),
                                 hovertemplate="vector %{x}: eigenvalue ratio %{y:.3g}<extra></extra>"))
    fig.add_hline(y=eigthresh, line=dict(color="#d62728", dash="dash"),
                  annotation_text=f"EIGTHRESH {eigthresh:g}", annotation_position="top right")
    low = min(ratios.min(), eigthresh) / 10
    fig.update_layout(
        title=f"Eigenvalues of JᵀQJ relative to the largest (.jco.{iiter})", height=380,
        xaxis_title="Singular vector", yaxis=dict(title="Eigenvalue ratio (s/s₁)²", type="log",
                                                    range=[np.log10(low), 0.3]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"),
    )
    return fig
