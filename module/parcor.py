import dash
from dash import html, dcc
import dash_mantine_components as dmc

name = __name__.lstrip("module.") + "-"
fullname = "Parameter Correlation"

md = dcc.Markdown('''

    The parameter correlation matrix is one of the most practical diagnostics in calibration
    by telling you which parameters your data can (and cannot) distinguish.

    Starting from the covariance matrix $C = (J^TQJ)^{-1}$, where $J$ is the Jacobian matrix; and Q is the weighting matrix.

    The correlation matrix $R$ is a normalized form of the covariance matrix $C$, with elements defined as:

    $$
                   p_{ij} = \\frac{C_{ij}}{\sqrt{C_{ii}C_{jj}}}
    $$

    The correlation coefficient $p_{ij}$ ranges between -1 and 1. Diagonal elements $p_{ii}$ are always equal to 1,
    while off-diagonal elements $p_{ij}, i \\neq j$ quantify the degree of linear dependence between parameters $i$ and $j$.

    High absolute values of $p_{ij}$ indicate strong correlation,
    meaning the parameters have similar effects on model outputs and are difficult to estimate independently.
    This can lead to an ill-conditioned or nearly singular system, which may cause instability or failure in parameter updates during calibration.
    To address this issue, highly correlated parameters can be constrained or tied together, reducing redundancy.
    Alternatively, regularization techniques, such as introducing preferred parameter values, can be applied to stabilize the inversion.
    ''',
    mathjax=True)


layout = html.Div([
    html.H5(fullname),
    md,
    dmc.Group([
        dmc.Select(id=name+"stype", label="Sensitivity Type:", data=["L1 Norm", "L2 Norm"], value="L2 Norm", style={"width": "150px"}),
        dmc.Select(id=name+"otype", label="Observation Type:",
                   data=["Observations", "Obs Groups"], value="Observations", style={"width": "150px"}),
        dmc.Select(id=name+"npar", label="Number of Parameters:", data=["5", "10", "20", "30", "40", "50", "100"], value="50", style={"width": "150px"}),
        dmc.Button("Plot Parameter", id=name+"update-plot", color="blue"),
    ], mt=10, align="flex-end"),
    dmc.MultiSelect(
        label="Select parameter to analyze sensitivity:",
        id=name+"parselect",
        value=[],
        data=data,
        mb=10,
    ),
    dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, ))),
])
