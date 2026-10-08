"""PEST Calibration Diagnostics — Data Worth Page"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

def _mock_data():
    rng=np.random.default_rng(13)
    obs_groups={"Heads":15,"Flows":8,"Conc.":10,"Levels":7}
    rows=[]
    for g,n in obs_groups.items():
        for i in range(n):
            name=f"{g[:2].lower()}{i:02d}"
            leverage=rng.beta(2,5)
            influence=rng.beta(1.5,4)*leverage*2
            worth_pred=rng.exponential(0.3)
            rows.append(dict(name=name,group=g,leverage=leverage,
                             influence=influence,worth_pred=worth_pred))
    df=pd.DataFrame(rows)
    # Simulate "remove one" impact on prediction uncertainty
    pred_unc_base=1.0
    remove_impact=rng.exponential(0.15,len(df))
    df["remove_impact"]=remove_impact
    return dict(residuals=df, obs_groups=list(obs_groups.keys()),
                _data_worth_df=df, _pred_unc_base=pred_unc_base)

PEST_DATA=_mock_data()
COLORS=["#4a90d9","#2f7a5e","#c0774a","#e8a838","#9b7ec8"]
BASE=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans, sans-serif",size=11,color="#555"),
          margin=dict(l=54,r=20,t=36,b=44),
          hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))
GRID=dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,tickfont=dict(size=10))

def _leverage_fig(df,groups):
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(groups)}
    fig=go.Figure()
    for g in groups:
        sub=df[df["group"]==g]
        fig.add_trace(go.Bar(name=g,x=sub["name"],y=sub["leverage"],
            marker_color=pgc[g],marker_line_width=0,opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>Group: {g}<br>Leverage=%{{y:.3f}}<extra></extra>"))
    hii_mean=df["leverage"].mean()
    fig.add_hline(y=hii_mean,line=dict(color="#555",width=1,dash="dot"),
        annotation_text=f"Mean={hii_mean:.3f}",annotation_position="right",
        annotation_font=dict(size=8))
    p=len(df.columns); n=len(df)
    fig.add_hline(y=2*p/n,line=dict(color="#d94a4a",width=1,dash="dash"),
        annotation_text="2p/n threshold",annotation_position="right",
        annotation_font=dict(size=8,color="#d94a4a"))
    fig.update_layout(**BASE,barmode="group",
        xaxis=dict(tickfont=dict(size=8),tickangle=-60,showticklabels=True,showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Leverage (hᵢᵢ)"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _influence_scatter(df,groups):
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(groups)}
    fig=go.Figure()
    for g in groups:
        sub=df[df["group"]==g]
        fig.add_trace(go.Scatter(x=sub["leverage"],y=sub["influence"],
            name=g,mode="markers",
            marker=dict(color=pgc[g],size=8,opacity=0.78,line=dict(width=0.5,color="white")),
            customdata=sub["name"].values,
            hovertemplate=f"<b>%{{customdata}}</b><br>Group:{g}<br>Leverage=%{{x:.3f}}<br>Influence=%{{y:.3f}}<extra></extra>"))
    # Threshold for high leverage
    fig.add_vline(x=df["leverage"].quantile(0.75),line=dict(color="#e8a838",width=1,dash="dot"),
        annotation_text="Q75 leverage",annotation_position="top",annotation_font=dict(size=8))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Leverage"),
        yaxis=dict(**GRID,title="Influence (Cook's D analog)"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _remove_impact_fig(df,groups,n=15):
    top=df.nlargest(n,"remove_impact")
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(groups)}
    fig=go.Figure(go.Bar(
        x=top["remove_impact"].values[::-1],
        y=[f"{r.name} ({r.group})" for _,r in top.iloc[::-1].iterrows()],
        orientation="h",
        marker_color=[pgc.get(r.group,"#aaa") for _,r in top.iloc[::-1].iterrows()],
        marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{y}</b><br>Prediction unc. increase=%{x:.3f}<extra></extra>"))
    b2={**BASE,"margin":dict(l=130,r=20,t=36,b=44)}
    fig.update_layout(**b2,xaxis=dict(**GRID,title="Increase in prediction uncertainty if removed"),
        yaxis=dict(tickfont=dict(size=9),showgrid=False,zeroline=False),showlegend=False)
    return fig

def _group_worth_fig(df,groups):
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(groups)}
    group_worth=df.groupby("group")["worth_pred"].sum().reindex(groups)
    group_n=df.groupby("group").size().reindex(groups)
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Total worth",x=groups,y=group_worth.values,
        marker_color=[pgc[g] for g in groups],marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Total worth=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(name="Per obs.",x=groups,y=(group_worth/group_n).values,
        mode="markers",marker=dict(color="#333",size=10,symbol="diamond"),
        yaxis="y2",hovertemplate="<b>%{x}</b><br>Per-obs worth=%{y:.3f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(tickfont=dict(size=10),showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Total predictive worth"),
        yaxis2=dict(title="Per-observation worth",overlaying="y",side="right",
                    showgrid=False,zeroline=False,tickfont=dict(size=9)),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _sec(t,s=None):
    return dmc.Stack(gap=0,mb="xs",children=[
        dmc.Text(t,fw=600,size="sm",style={"color":"#333"}),
        dmc.Text(s,size="xs",c="dimmed") if s else html.Span()])

def _kpi(label,value,icon=None,color="teal"):
    return dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},
        children=[dmc.Group(justify="space-between",align="flex-start",children=[
            dmc.Stack(gap=2,children=[
                dmc.Text(label,size="xs",c="dimmed",tt="uppercase",fw=500),
                dmc.Text(value,fw=700,style={"fontFamily":"DM Mono, monospace","fontSize":"1.5rem"})]),
            dmc.ThemeIcon(DashIconify(icon=icon,width=18),size="lg",radius="md",variant="light",color=color) if icon else html.Span()])])

def layout(data=None):
    d=data or PEST_DATA
    df=d.get("_data_worth_df",pd.DataFrame())
    groups=d.get("obs_groups",[])
    if df.empty:
        return dmc.Alert("Data worth information not available.",color="orange",title="Data missing")
    n_high_lev=int((df["leverage"]>df["leverage"].quantile(0.75)).sum())
    n_high_inf=int((df["influence"]>df["influence"].quantile(0.9)).sum())
    most_worth=df.loc[df["worth_pred"].idxmax(),"name"]
    most_impact=df.loc[df["remove_impact"].idxmax(),"name"]
    return dmc.Stack(gap="lg",p="lg",children=[
        dmc.Group(justify="space-between",align="flex-end",children=[
            dmc.Stack(gap=2,children=[
                dmc.Title("Data Worth",order=3,style={"color":"#1a1a1a"}),
                dmc.Text("Observation leverage, influence, and contribution to prediction uncertainty",size="sm",c="dimmed")])]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4},spacing="md",children=[
            _kpi("Total observations",str(len(df)),"tabler:list-numbers","teal"),
            _kpi("High leverage",str(n_high_lev),"tabler:arrow-up-right","orange"),
            _kpi("High influence",str(n_high_inf),"tabler:flame","red"),
            _kpi("Most valuable",most_worth,"tabler:star","grape")]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Observation Leverage","hᵢᵢ = influence on own fitted value. Red dashed = 2p/n threshold for high leverage"),
            dcc.Graph(figure=_leverage_fig(df,groups),config={"displayModeBar":False},style={"height":"250px"})]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":5},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Leverage vs Influence","High leverage + high influence = most critical observations"),
                    dcc.Graph(figure=_influence_scatter(df,groups),config={"displayModeBar":False},style={"height":"260px"})])),
            dmc.GridCol(span={"base":12,"md":7},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Top-15 Observations by Predictive Worth","Removing these observations would increase prediction uncertainty the most"),
                    dcc.Graph(figure=_remove_impact_fig(df,groups),config={"displayModeBar":False},style={"height":"260px"})]))]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Predictive Worth by Observation Group",
                 "Total and per-observation worth for each group. Identifies redundant vs essential data collection"),
            dcc.Graph(figure=_group_worth_fig(df,groups),config={"displayModeBar":False},style={"height":"220px"})]),
        dmc.Alert(title="Data worth notes",color="teal",variant="light",radius="md",
            icon=DashIconify(icon="tabler:bulb",width=18),
            children=dmc.List(size="sm",spacing="xs",children=[
                dmc.ListItem(f"{n_high_lev} high-leverage observations — check for errors or outliers."),
                dmc.ListItem(f"Most prediction-critical observation: {most_impact}. Prioritise quality control for this measurement."),
                dmc.ListItem(f"Most informative observation overall: {most_worth}."),
                dmc.ListItem("Groups with low per-observation worth may be redundant — consider reducing future monitoring.")]))])

def register_callbacks(app): pass
