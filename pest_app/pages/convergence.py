"""PEST Calibration Diagnostics — Convergence Page"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

def _mock_data():
    rng=np.random.default_rng(9)
    n=14; t=np.arange(n)
    phi=12000*np.exp(-0.35*t)+rng.normal(0,80,n)
    lam=100*np.exp(-0.4*t)+rng.uniform(0.1,2,n)
    grad=np.exp(-0.3*t)*rng.uniform(0.8,1.2,n)
    rel_improve=np.diff(phi)/phi[:-1]*100
    param_change=rng.exponential(0.5,(n,8))*np.exp(-0.2*t[:,None])
    param_names=["K_hk1","K_hk2","Sy1","Rch_z1","Rch_z2","GHB_k","RIV_k","alphaL"]
    model_calls=np.cumsum(rng.integers(8,15,n))
    return dict(n_iter=n,iterations=t+1,phi_total=phi,lambda_vals=lam,
                rel_improve=rel_improve,_grad_norm=grad,
                _param_change=pd.DataFrame(param_change,columns=param_names),
                _model_calls=model_calls,converged=True,phimlim=850.0)

PEST_DATA=_mock_data()
BASE=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Sans, sans-serif",size=11,color="#555"),
          margin=dict(l=54,r=20,t=36,b=44),
          hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))
GRID=dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,tickfont=dict(size=10))

def _param_change_fig(df):
    fig=go.Figure()
    COLORS=["#4a90d9","#2f7a5e","#c0774a","#9b7ec8","#e8a838","#d94a4a","#8ab3cc","#5bab8a"]
    for i,col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=list(range(1,len(df)+1)),y=df[col],
            name=col,mode="lines+markers",
            line=dict(color=COLORS[i%len(COLORS)],width=1.5),marker=dict(size=4),
            hovertemplate=f"{col}<br>Iter %{{x}}<br>|Δθ|=%{{y:.4f}}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="|Δθ| (log)",type="log"),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=9),bgcolor="rgba(0,0,0,0)"))
    return fig

def _lambda_grad_fig(iters,lam,grad):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=iters,y=lam,name="Lambda (λ)",
        mode="lines+markers",line=dict(color="#9b7ec8",width=2),marker=dict(size=5),
        hovertemplate="Iter %{x}<br>λ=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=iters,y=grad,name="Gradient norm",
        mode="lines+markers",line=dict(color="#c0774a",width=2,dash="dash"),marker=dict(size=5),
        yaxis="y2",hovertemplate="Iter %{x}<br>‖∇Φ‖=%{y:.4f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="Lambda (log)",type="log"),
        yaxis2=dict(title="Gradient norm (log)",type="log",overlaying="y",side="right",
                    showgrid=False,zeroline=False,tickfont=dict(size=9)),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _model_calls_fig(iters,calls):
    per_iter=np.diff(np.concatenate([[0],calls]))
    fig=go.Figure(go.Bar(x=iters,y=per_iter,
        marker_color="#4a90d9",marker_line_width=0,opacity=0.82,
        hovertemplate="Iter %{x}<br>Model calls=%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=iters,y=calls,name="Cumulative",
        mode="lines+markers",line=dict(color="#d94a4a",width=1.5),marker=dict(size=4),
        yaxis="y2",hovertemplate="Iter %{x}<br>Total calls=%{y}<extra></extra>"))
    fig.update_layout(**BASE,barmode="overlay",
        xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="Calls per iteration"),
        yaxis2=dict(title="Cumulative calls",overlaying="y",side="right",
                    showgrid=False,zeroline=False,tickfont=dict(size=9)),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _rel_improve_fig(iters,rel):
    fig=go.Figure(go.Bar(x=iters[1:],y=rel,
        marker_color=["#2f7a5e" if v<0 else "#d94a4a" for v in rel],
        marker_line_width=0,opacity=0.85,
        hovertemplate="Iter %{x}<br>ΔΦ/Φ=%{y:.2f}%<extra></extra>"))
    fig.add_hline(y=0,line=dict(color="#555",width=0.8))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Iteration",dtick=1),
        yaxis=dict(**GRID,title="Relative Φ change (%)"),showlegend=False)
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
    iters=d.get("iterations",np.array([])); phi=d.get("phi_total",np.array([]))
    lam=d.get("lambda_vals",np.array([])); rel=d.get("rel_improve",np.array([]))
    grad=d.get("_grad_norm",np.ones(len(iters))); param_ch=d.get("_param_change",pd.DataFrame())
    calls=d.get("_model_calls",np.array([])); converged=d.get("converged",False)
    n_failed=int((rel>0).sum()) if len(rel) else 0
    total_calls=int(calls[-1]) if len(calls) else 0
    final_lam=float(lam[-1]) if len(lam) else 0
    return dmc.Stack(gap="lg",p="lg",children=[
        dmc.Group(justify="space-between",align="flex-end",children=[
            dmc.Stack(gap=2,children=[
                dmc.Title("Convergence",order=3,style={"color":"#1a1a1a"}),
                dmc.Text("Lambda, gradient norm, parameter change trajectories, and model call efficiency",size="sm",c="dimmed")]),
            dmc.Badge("Converged" if converged else "Not converged",
                      color="teal" if converged else "red",variant="light",size="sm")]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4},spacing="md",children=[
            _kpi("Total iterations",str(d.get("n_iter",0)),"tabler:refresh","teal"),
            _kpi("Failed steps",str(n_failed),"tabler:alert-triangle","red" if n_failed>2 else "orange"),
            _kpi("Final lambda",f"{final_lam:.3f}","tabler:sigma","grape"),
            _kpi("Model calls",str(total_calls),"tabler:cpu","blue")]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Parameter Change Magnitude","‖Δθⱼ‖ per iteration for each parameter (log scale). Flattening = convergence"),
            dcc.Graph(figure=_param_change_fig(param_ch) if not param_ch.empty else go.Figure(),
                      config={"displayModeBar":False},style={"height":"280px"})]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Lambda & Gradient Norm","λ (left) and ‖∇Φ‖ (right) — both should decrease monotonically"),
                    dcc.Graph(figure=_lambda_grad_fig(iters,lam,grad),
                              config={"displayModeBar":False},style={"height":"250px"})])),
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Relative Φ Improvement","Green = improvement, Red = failed step (Φ increased)"),
                    dcc.Graph(figure=_rel_improve_fig(iters,rel),
                              config={"displayModeBar":False},style={"height":"250px"})]))]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Model Call Efficiency","Calls per iteration (bars) and cumulative total (line)"),
            dcc.Graph(figure=_model_calls_fig(iters,calls) if len(calls) else go.Figure(),
                      config={"displayModeBar":False},style={"height":"220px"})]),
        dmc.Alert(title="Convergence notes",color="teal",variant="light",radius="md",
            icon=DashIconify(icon="tabler:bulb",width=18),
            children=dmc.List(size="sm",spacing="xs",children=[
                dmc.ListItem(f"{n_failed} failed step(s) detected (Φ increased). Check for model instability or parameter bounds."),
                dmc.ListItem(f"Final λ = {final_lam:.3f}. "+("Good gradient-dominated convergence." if final_lam<1 else "Elevated λ — optimiser still struggling.")),
                dmc.ListItem(f"Total model calls: {total_calls}. Aim to minimise via NOPTMAX and PHIREDSTP tuning.")]))])

def register_callbacks(app): pass
