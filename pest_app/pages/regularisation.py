"""PEST Calibration Diagnostics — Regularisation Page"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

def _mock_data():
    rng = np.random.default_rng(5)
    n = 14
    t = np.arange(n)
    phi_meas = 8000*np.exp(-0.3*t)+rng.normal(0,60,n)
    phi_reg  = 4000*np.exp(-0.1*t)+rng.normal(0,40,n)
    phi_total= phi_meas+phi_reg
    reg_factor = 1.0/(1+0.5*t)+rng.uniform(0,0.05,n)
    pg = {"Hyd. cond.":["K_hk1","K_hk2","K_hk3"],
          "Storage":["Sy1","Ss1"],
          "Recharge":["Rch_z1","Rch_z2"]}
    params=[p for ps in pg.values() for p in ps]
    prior = rng.uniform(0.5,5,len(params))
    post  = prior + rng.normal(0,0.5,len(params))
    reg_pull = np.abs(post-prior)/prior*100
    return dict(phi_history=pd.DataFrame({"iteration":t+1,"phi_total":phi_total,
                    "phi_meas":phi_meas,"phi_reg":phi_reg,"lambda":np.ones(n)}),
                n_iter=n, iterations=t+1,
                phi_total=phi_total,phi_meas=phi_meas,phi_reg=phi_reg,
                phimlim=850.0,
                _reg_factor=reg_factor,
                _param_names=params,
                _param_group_map={p:g for g,ps in pg.items() for p in ps},
                _prior_values=prior,_post_values=post,_reg_pull=reg_pull)

PEST_DATA=_mock_data()
BASE=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans, sans-serif",size=11,color="#555"),
          margin=dict(l=54,r=20,t=36,b=44),
          hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))
GRID=dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,tickfont=dict(size=10))

def _tradeoff_fig(phi_meas,phi_reg):
    fig=go.Figure()
    n=len(phi_meas)
    colors=[f"rgba(74,144,217,{0.3+0.7*i/max(n-1,1)})" for i in range(n)]
    fig.add_trace(go.Scatter(x=phi_meas,y=phi_reg,mode="lines+markers",
        marker=dict(color=list(range(n)),colorscale="Blues",size=8,
                    colorbar=dict(title=dict(text="Iter",font=dict(size=9)),thickness=10,tickfont=dict(size=8))),
        line=dict(color="#4a90d9",width=1.5),
        hovertemplate="Iter %{text}<br>Φ_meas=%{x:.1f}<br>Φ_reg=%{y:.1f}<extra></extra>",
        text=[str(i+1) for i in range(n)]))
    fig.add_trace(go.Scatter(x=[phi_meas[-1]],y=[phi_reg[-1]],mode="markers",
        marker=dict(color="#d94a4a",size=12,symbol="star"),name="Final",
        hovertemplate="Final<br>Φ_meas=%{x:.1f}<br>Φ_reg=%{y:.1f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Measurement Φ"),
        yaxis=dict(**GRID,title="Regularisation Φ"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _phi_breakdown_fig(iters,phi_meas,phi_reg,phimlim):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=iters,y=phi_meas,name="Measurement Φ",
        mode="lines+markers",line=dict(color="#4a90d9",width=2),marker=dict(size=5),
        stackgroup="one",fillcolor="rgba(74,144,217,0.3)"))
    fig.add_trace(go.Scatter(x=iters,y=phi_reg,name="Regularisation Φ",
        mode="lines+markers",line=dict(color="#c0774a",width=2),marker=dict(size=5),
        stackgroup="one",fillcolor="rgba(192,119,74,0.3)"))
    if phimlim:
        fig.add_hline(y=phimlim,line=dict(color="#e8a838",width=1.5,dash="dot"),
            annotation_text="PHIMLIM",annotation_position="right",
            annotation_font=dict(size=9,color="#e8a838"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="Φ"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _reg_factor_fig(iters,reg_factor):
    fig=go.Figure(go.Scatter(x=iters,y=reg_factor,mode="lines+markers",
        line=dict(color="#9b7ec8",width=2),marker=dict(size=5),
        fill="tozeroy",fillcolor="rgba(155,126,200,0.1)",
        hovertemplate="Iter %{x}<br>Factor=%{y:.4f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="Regularisation weight factor"),showlegend=False)
    return fig

def _reg_pull_fig(names,pull,pgmap):
    order=np.argsort(pull)[::-1]
    COLORS=["#4a90d9","#2f7a5e","#c0774a","#9b7ec8","#e8a838"]
    unique_g=list(dict.fromkeys([pgmap.get(names[i],"?") for i in order]))
    pgc={g:COLORS[i%len(COLORS)] for i,g in enumerate(unique_g)}
    fig=go.Figure(go.Bar(
        x=[names[i] for i in order], y=[pull[i] for i in order],
        marker_color=[pgc.get(pgmap.get(names[i],"?"),"#aaa") for i in order],
        marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Reg. pull = %{y:.1f}%<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(tickangle=-40,tickfont=dict(size=9),showgrid=False,zeroline=False),
        yaxis=dict(**GRID,title="Regularisation pull (% of prior)"),showlegend=False)
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
    phi_meas=d.get("phi_meas",np.array([]))
    phi_reg=d.get("phi_reg",np.array([]))
    iters=d.get("iterations",np.array([]))
    phimlim=d.get("phimlim",None)
    reg_factor=d.get("_reg_factor",np.ones(len(iters)))
    names=d.get("_param_names",[])
    pgmap=d.get("_param_group_map",{})
    pull=d.get("_reg_pull",np.zeros(len(names)))
    final_ratio=float(phi_reg[-1]/max(phi_meas[-1],1e-9)) if len(phi_reg) else 0
    pct_reg=float(phi_reg[-1]/(phi_reg[-1]+phi_meas[-1]+1e-9)*100) if len(phi_reg) else 0
    most_pulled=names[int(np.argmax(pull))] if len(names) else "N/A"
    return dmc.Stack(gap="lg",p="lg",children=[
        dmc.Group(justify="space-between",align="flex-end",children=[
            dmc.Stack(gap=2,children=[
                dmc.Title("Regularisation",order=3,style={"color":"#1a1a1a"}),
                dmc.Text("Φ_meas vs Φ_reg trade-off, regularisation weight factor, and per-parameter pull",size="sm",c="dimmed")])]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4},spacing="md",children=[
            _kpi("Final Φ_reg",f"{phi_reg[-1]:,.0f}" if len(phi_reg) else "—","tabler:adjustments","grape"),
            _kpi("Reg. share",f"{pct_reg:.1f}%","tabler:percentage","orange"),
            _kpi("Φ_reg/Φ_meas",f"{final_ratio:.3f}","tabler:divide","blue"),
            _kpi("Most pulled",most_pulled,"tabler:arrow-big-right","red")]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Φ_meas vs Φ_reg Trade-off Curve",
                         "L-curve: early iterations top-right; convergence toward bottom-left; star = final"),
                    dcc.Graph(figure=_tradeoff_fig(phi_meas,phi_reg),
                              config={"displayModeBar":False},style={"height":"280px"})])),
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Φ Breakdown — Stacked by Component",
                         "Stacked area showing measurement vs regularisation Φ across iterations"),
                    dcc.Graph(figure=_phi_breakdown_fig(iters,phi_meas,phi_reg,phimlim),
                              config={"displayModeBar":False},style={"height":"280px"})]))]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":5},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Regularisation Weight Factor","PHIREG scaling factor across iterations"),
                    dcc.Graph(figure=_reg_factor_fig(iters,reg_factor),
                              config={"displayModeBar":False},style={"height":"240px"})])),
            dmc.GridCol(span={"base":12,"md":7},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Per-parameter Regularisation Pull",
                         "How far each parameter was pulled from its prior (%) by regularisation"),
                    dcc.Graph(figure=_reg_pull_fig(names,pull,pgmap),
                              config={"displayModeBar":False},style={"height":"240px"})]))]),
        dmc.Alert(title="Regularisation notes",color="teal",variant="light",radius="md",
            icon=DashIconify(icon="tabler:bulb",width=18),
            children=dmc.List(size="sm",spacing="xs",children=[
                dmc.ListItem(f"Regularisation accounts for {pct_reg:.1f}% of final total Φ."),
                dmc.ListItem("If Φ_reg/Φ_meas >> 1, regularisation is dominating — consider relaxing PHIMLIM or reducing reg. weight."),
                dmc.ListItem(f"Most regularised parameter: {most_pulled}. Verify its prior value and uncertainty."),
                dmc.ListItem("A well-balanced calibration sits near the L-curve corner on the trade-off plot.")]))])

def register_callbacks(app): pass
