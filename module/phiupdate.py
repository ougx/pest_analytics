import dash
from dash import dcc, html, callback, Input, Output, no_update
import dash_mantine_components as dmc
from utils import parse_zip, isiterable
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from layout import update_groups, update_weights

# dash.register_page(__name__, path='/')

name = __name__.lstrip("module.") + "-"
fullname = "Phi Evolution"


md = dcc.Markdown('''

    This page investigates Phi update.

    ''',
    mathjax=True)

layout = dmc.Stack(
    [
        html.H3(fullname),
        md,
        dmc.Group([
        dmc.Select(id=name+"grp"   , label="Grouped by:" , data=["obsgrp"], value="obsgrp", style={"width": "150px"}),
        dmc.Select(id=name+"weight", label="Weighted by:", data=["weight"], value="weight", style={"width": "150px"}),
        ], mt=10, align="flex-end"),
        dcc.Loading(dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, )), style={"margin-bottom":"90px"})),
    ]
)


@callback(
    Output(name+"grp"   , "data"),
    Output(name+"grp"   , "value"),
    Output(name+"weight", "data"),
    Output(name+"weight", "value"),
    Input( "pest-data", "data"),
)
def upload_pest(data):
    return update_groups(data, "grpType") + update_weights(data, "weight")


@callback(
    Output(name+"plot", "figure"),
    Input("pest-data", "data"),
    Input(name+"grp", "value"),
    Input(name+"weight", "value"),
)
def upload_pest(data, grp, weight):
    if data:
        pst, par, obs, pars, reis, jcos = parse_zip(data, True, True, False)
        if pst and reis:
            phi = {}
            for i, df in reis.items():
                #print(df)
                df = df[df[weight]>0].copy()
                if not pd.api.types.is_numeric_dtype(df.residual) or (df.residual.abs() > 1e20).any():
                    return go.Figure(layout=dict(
                        title=f"Unreasonable residuals identified in Iteration {i}. Please check your {pst}.rei.{i} file",
                        height=650,)
                    )
                df["phi"] = (df[weight] * df.residual)**2
                phi[i] = df.groupby(grp).phi.sum()
            phi = pd.DataFrame(phi)
            phi.columns.name = "Iteration"
            phi = phi.stack().rename("phi").reset_index()
            phi_ter = phi.groupby("Iteration").phi.sum().reset_index()
            phi_ter_no_regul = phi[~phi[grp].str.startswith("regul")].groupby("Iteration").phi.sum().reset_index()
            phi_ter["%"] = (phi_ter_no_regul.phi / phi_ter_no_regul.phi.iloc[0] * 100).round(1).astype(str) + "%"
            # print(reis)
            # print(phi)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for g, df in phi.groupby(grp):
                df["pct"] = (df.phi / df.Iteration.map(phi_ter_no_regul.set_index("Iteration").phi) * 100).round(1)
                fig.add_trace(go.Bar(
                    x="Iter" + df.Iteration.astype(str),
                    y=df.phi,
                    hovertext=[t+"% of Total Phi" for t in df[["pct"]].to_string(index=False, header=False).split("\n")],
                    name=g,
                    legendgroup=g,
                ))
            fig.update_layout(barmode='stack')
            # fig = px.bar(
            #     phi,
            #     x="Iteration",
            #     y="phi",
            #     color="group",

            #     height=650,)
            # add %
            fig.add_trace(go.Scatter(
                x="Iter" + phi_ter.Iteration.astype(str),
                y=phi_ter.phi*1.01,
                mode="text",
                text=phi_ter["%"],
                textposition="top center",
                textfont=dict(
                    color="red"
                ),
                showlegend=False,
            ))

            vars = {0:0}
            for i, df in pars.items():
                if i == 0: continue
                v = ((df.parval - par.parval0.values) / par.parval0.values)
                vars[i] = v[par.partrans!="fixed"].var()

            variance = pd.Series(vars).sort_index().rename("Parameter Variance")
            fig.add_trace(go.Scatter(
                x="Iter" + variance.index.astype(str),
                y=variance,
                mode="lines+markers",
                line=dict(color='black', width=2),
                name="Parameter Variance",
                showlegend=True,
            ), secondary_y=True)

            fig.update_layout(
                title="Phi Update by Group; % not included regul group; Parameter variance = var((ParVal - Parval0) / Parval0)",
                height=650,
            )
            fig.update_xaxes(title="Iteration", )
            fig.update_yaxes(title="Phi", )
            fig.update_yaxes(title="Parameter Variance", showgrid=False, secondary_y=True)
            return fig

    return go.Figure(layout=dict(
            title="Upload data to see Phi Update plot.",
            height=650,))
