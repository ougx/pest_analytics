"""PEST Calibration Diagnostics — Uncertainty Page"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

def _mock_data():
    rng=np.random.default_rng(17)
    pg={"Hyd. cond.":["K_hk1","K_hk2","K_hk3","K_hk4"],
        "Storage":["Sy1","Ss1","Sy2"],
        "Recharge":["Rch_z1","Rch_z2","Rch_z3"],
        "Boundary":["GHB_k","RIV_k","DRN_k"]}
    params=[p for ps in pg.values() for p in ps]
    n_p=len(params)
    J=rng.normal(0,1,(60,n_p)); J[:,0]*=4; J[:,1]*=3
    W=np.diag(rng.uniform(0.5,2,60))
    try:
        JtWJ=J.T@W@J
        cov=np.linalg.inv(JtWJ+0.01*np.eye(n_p))
    except: cov=np.eye(n_p)*0.1
    std_dev=np.sqrt(np.abs(np.diag(cov)))
    prior_std=rng.uniform(0.5,2,n_p)
    reduction=np.clip(1-std_dev/prior_std,0,1)*100
    # Prediction uncertainty
    pred_names=[f"GW_head_{i+1}" for i in range(8)]
    pred_mean=rng.uniform(10,50,8)
    pred_std=rng.exponential(2,8)
    pred_prior_std=pred_std+rng.uniform(1,5,8)
    # 95% confidence bands for a time series prediction
    t_pred=np.linspace(0,10,50)
    sim_trend=5*np.exp(-0.3*t_pred)+2
    unc_band=pred_std[0]*np.exp(-0.1*t_pred)+0.5
    return dict(
        covariance=pd.DataFrame(cov,index=params,columns=params),
        param_groups=list(pg.keys()),
        _param_names=params,
        _param_group_map={p:g for g,ps in pg.items() for p in ps},
        _std_dev=std_dev, _prior_std=prior_std,
        _unc_reduction=reduction,
        _pred_names=pred_names,_pred_mean=pred_mean,
        _pred_std=pred_std,_pred_prior_std=pred_prior_std,
        _t_pred=t_pred,_sim_trend=sim_trend,_unc_band=unc_band)

PEST_DATA=_mock_data()
COLORS=["#4a90d9","#2f7a5e","#c0774a","#9b7ec8","#e8a838","#d94a4a"]
BASE=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans, sans-serif",size=11,color="#555"),
          margin=dict(l=54,r=20,t=36,b=44),
          hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))
GRID=dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,tickfont=dict(size=10))

def _std_comparison_fig(names,prior_std,post_std,pgmap,pgroups):
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(pgroups)}
    order=np.argsort(post_std)[::-1]
    ns=[names[i] for i in order]; ps=prior_std[order]; qs=post_std[order]
    gs=[pgmap.get(n,"?") for n in ns]
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Prior std",x=ns,y=ps,
        marker_color="rgba(180,180,180,0.5)",marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Prior std=%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Posterior std",x=ns,y=qs,
        marker_color=[pgc.get(g,"#aaa") for g in gs],marker_line_width=0,opacity=0.9,
        hovertemplate="<b>%{x}</b><br>Posterior std=%{y:.4f}<extra></extra>"))
    fig.update_layout(**BASE,barmode="overlay",
        xaxis=dict(tickangle=-40,tickfont=dict(size=9),showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Standard deviation"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _unc_reduction_fig(names,reduction,pgmap,pgroups):
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(pgroups)}
    order=np.argsort(reduction)[::-1]
    ns=[names[i] for i in order]; rs=reduction[order]
    gs=[pgmap.get(n,"?") for n in ns]
    fig=go.Figure(go.Bar(x=ns,y=rs,
        marker_color=[pgc.get(g,"#aaa") for g in gs],marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Uncertainty reduction=%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50,line=dict(color="#e8a838",width=1,dash="dot"),
        annotation_text="50%",annotation_position="right",annotation_font=dict(size=8))
    fig.update_layout(**BASE,
        xaxis=dict(tickangle=-40,tickfont=dict(size=9),showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Uncertainty reduction (%)",range=[0,105]),showlegend=False)
    return fig

def _pred_unc_fig(pred_names,pred_mean,pred_std,pred_prior_std):
    fig=go.Figure()
    x=list(range(len(pred_names)))
    fig.add_trace(go.Scatter(x=pred_names,y=pred_mean+1.96*pred_prior_std,
        fill=None,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred_names,y=pred_mean-1.96*pred_prior_std,
        fill="tonexty",mode="lines",line=dict(width=0),
        fillcolor="rgba(180,180,180,0.25)",name="Prior 95% CI",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred_names,y=pred_mean+1.96*pred_std,
        fill=None,mode="lines",line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred_names,y=pred_mean-1.96*pred_std,
        fill="tonexty",mode="lines",line=dict(width=0),
        fillcolor="rgba(74,144,217,0.25)",name="Posterior 95% CI",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=pred_names,y=pred_mean,mode="markers+lines",
        name="Predicted value",line=dict(color="#4a90d9",width=2),
        marker=dict(size=8,color="#4a90d9"),
        hovertemplate="<b>%{x}</b><br>Predicted=%{y:.2f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(tickfont=dict(size=9),showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Predicted value"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _time_series_unc_fig(t,sim,band):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=sim+1.96*band,fill=None,mode="lines",
        line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t,y=sim-1.96*band,fill="tonexty",mode="lines",
        line=dict(width=0),fillcolor="rgba(74,144,217,0.2)",name="95% CI",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t,y=sim+band,fill=None,mode="lines",
        line=dict(width=0),showlegend=False,hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t,y=sim-band,fill="tonexty",mode="lines",
        line=dict(width=0),fillcolor="rgba(74,144,217,0.15)",name="68% CI",hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t,y=sim,mode="lines",name="Best estimate",
        line=dict(color="#4a90d9",width=2),
        hovertemplate="t=%{x:.2f}<br>Value=%{y:.3f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Time"),
        yaxis=dict(**GRID,title="Predicted value"),
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
    names=d.get("_param_names",[]); post_std=d.get("_std_dev",np.array([]))
    prior_std=d.get("_prior_std",np.array([])); reduction=d.get("_unc_reduction",np.array([]))
    pred_names=d.get("_pred_names",[]); pred_mean=d.get("_pred_mean",np.array([]))
    pred_std=d.get("_pred_std",np.array([])); pred_prior_std=d.get("_pred_prior_std",np.array([]))
    t_pred=d.get("_t_pred",np.array([])); sim=d.get("_sim_trend",np.array([]))
    band=d.get("_unc_band",np.array([])); pgmap=d.get("_param_group_map",{})
    pgroups=d.get("param_groups",[])
    mean_red=float(reduction.mean()) if len(reduction) else 0
    max_red_param=names[int(np.argmax(reduction))] if len(names) else "N/A"
    min_red_param=names[int(np.argmin(reduction))] if len(names) else "N/A"
    return dmc.Stack(gap="lg",p="lg",children=[
        dmc.Group(justify="space-between",align="flex-end",children=[
            dmc.Stack(gap=2,children=[
                dmc.Title("Uncertainty",order=3,style={"color":"#1a1a1a"}),
                dmc.Text("FOSM parameter uncertainty, prediction confidence intervals, and uncertainty reduction",size="sm",c="dimmed")])]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4},spacing="md",children=[
            _kpi("Parameters",str(len(names)),"tabler:adjustments","teal"),
            _kpi("Mean unc. reduction",f"{mean_red:.1f}%","tabler:trending-down","green"),
            _kpi("Best constrained",max_red_param,"tabler:check","blue"),
            _kpi("Least constrained",min_red_param,"tabler:alert-triangle","orange")]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Prior vs Posterior Standard Deviation","Grey=prior, coloured=posterior. Overlap indicates poor data constraint"),
                    dcc.Graph(figure=_std_comparison_fig(names,prior_std,post_std,pgmap,pgroups),
                              config={"displayModeBar":False},style={"height":"280px"})])),
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Uncertainty Reduction by Parameter","% reduction in std dev from prior to posterior"),
                    dcc.Graph(figure=_unc_reduction_fig(names,reduction,pgmap,pgroups),
                              config={"displayModeBar":False},style={"height":"280px"})]))]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Prediction Uncertainty — Point Predictions",
                 "95% confidence intervals for each target prediction. Grey=prior CI, blue=posterior CI"),
            dcc.Graph(figure=_pred_unc_fig(pred_names,pred_mean,pred_std,pred_prior_std),
                      config={"displayModeBar":False},style={"height":"260px"})]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Prediction Uncertainty — Time Series",
                 "68% and 95% confidence bands around the best-estimate prediction trajectory"),
            dcc.Graph(figure=_time_series_unc_fig(t_pred,sim,band),
                      config={"displayModeBar":False},style={"height":"260px"})]),
        dmc.Alert(title="Uncertainty notes",color="teal",variant="light",radius="md",
            icon=DashIconify(icon="tabler:bulb",width=18),
            children=dmc.List(size="sm",spacing="xs",children=[
                dmc.ListItem(f"Mean uncertainty reduction across all parameters: {mean_red:.1f}%."),
                dmc.ListItem(f"Best constrained: {max_red_param}. Least constrained: {min_red_param} — consider adding targeted observations."),
                dmc.ListItem("Note: these are linear (FOSM) uncertainty estimates. Run PEST++ FOSM or Monte Carlo for non-linear cases."),
                dmc.ListItem("Wide posterior CI relative to prior = data is not constraining that parameter — check sensitivity.")]))])

def register_callbacks(app): pass
