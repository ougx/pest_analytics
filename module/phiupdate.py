import dash
from dash import dcc, html, callback, Input, Output
import dash_mantine_components as dmc
from utils import parse_zip, isiterable
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

# dash.register_page(__name__, path='/')

name = __name__.lstrip("module.") + "-"

layout = dmc.Stack(
    [
        dcc.Graph(id=name+"plot", figure=go.Figure(layout=dict(height=650, ))), #
    ]
)

@callback(
    Output(name+"plot", "figure"),
    Input("pest-data", "data"),
)
def upload_pest(data):
    if data is None or data == {}:
        return go.Figure(layout=dict(
                title="Upload data to see Phi Update plot.",
                height=650,))
    else:
        filename = data.get("filename")
        contents = data.get("contents")
        pst, par, obs, pars, reis, jcos = parse_zip(contents, True, True, False)
        phi = {}
        for i, df in reis.items():
            #print(df)
            df["phi"] = (df.weight * df.residual)**2
            phi[i] = df.groupby("group").phi.sum()
        phi = pd.DataFrame(phi)
        phi.columns.name = "Iteration"
        phi = phi.stack().rename("phi").reset_index()
        # print(reis)
        # print(phi)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for g, df in phi.groupby("group"):
            fig.add_trace(go.Bar(
                x=df.Iteration,
                y=df.phi,
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
        phi_ter = phi.groupby("Iteration").phi.sum().reset_index()
        phi_ter_no_regul = phi[~phi.group.str.startswith("regul")].groupby("Iteration").phi.sum().reset_index()
        phi_ter["%"] = (phi_ter_no_regul.phi / phi_ter_no_regul.phi.iloc[0] * 100).round(1).astype(str) + "%"
        fig.add_trace(go.Scatter(
            x=phi_ter.Iteration,
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
            x=variance.index,
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
