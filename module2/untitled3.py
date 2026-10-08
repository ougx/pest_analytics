# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:46:36 2026

@author: hydro
"""

import numpy as np
import dash
from dash import dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Example data (replace with your own C matrix) ---
n_obs = 10
n_par = 8
np.random.seed(0)

C = 0.1 * np.random.randn(n_obs, n_par)

row_sum = C.sum(axis=1)
col_sum = C.sum(axis=0)

# --- Create subplot layout ---
fig = make_subplots(
    rows=2, cols=2,
    column_widths=[0.8, 0.2],
    row_heights=[0.85, 0.15],
    specs=[
        [{"type": "heatmap"}, {"type": "bar"}],
        [{"type": "bar"}, None]
    ],
    horizontal_spacing=0.05,
    vertical_spacing=0.08
)

# --- Main heatmap ---
heatmap = go.Heatmap(
    z=C,
    colorscale="RdBu",
    zmid=0,
    colorbar=dict(title="C_ij"),
)

fig.add_trace(heatmap, row=1, col=1)

# --- Row sums (per observation) ---
fig.add_trace(
    go.Bar(
        x=row_sum,
        y=[f"Obs {i+1}" for i in range(n_obs)],
        orientation='h',
        marker=dict(color="gray")
    ),
    row=1, col=2
)

# --- Column sums (per parameter) ---
fig.add_trace(
    go.Bar(
        x=[f"Par {j+1}" for j in range(n_par)],
        y=col_sum,
        marker=dict(color="gray")
    ),
    row=2, col=1
)

# --- Layout styling ---
fig.update_layout(
    title="Observation–Parameter Contribution Matrix",
    height=700,
    showlegend=False,
)

fig.update_xaxes(title_text="Parameters (j)", row=1, col=1)
fig.update_yaxes(title_text="Observations (i)", row=1, col=1)

fig.update_xaxes(title_text="Contribution", row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=2)

fig.update_yaxes(title_text="Column Sum", row=2, col=1)

# --- Dash app ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Observation–Parameter Contribution Dashboard"),

    dcc.Graph(
        figure=fig,
        style={"height": "750px"}
    ),

    html.Div([
        html.H4("How to interpret"),
        html.Ul([
            html.Li("Red: reduces residual"),
            html.Li("Blue: increases residual"),
            html.Li("Rows: which parameters affect each observation"),
            html.Li("Columns: total contribution of each parameter"),
        ])
    ], style={"width": "60%", "margin": "auto"})
])

if __name__ == "__main__":
    app.run(debug=True)