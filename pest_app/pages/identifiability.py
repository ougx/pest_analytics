"""PEST Calibration Diagnostics — Identifiability Page"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_mantine_components as dmc
from dash_iconify import DashIconify

def _mock_data():
    rng = np.random.default_rng(11)
    pg = {"Hyd. cond.":["K_hk1","K_hk2","K_hk3","K_hk4"],
          "Storage":["Sy1","Ss1","Sy2"],
          "Recharge":["Rch_z1","Rch_z2","Rch_z3"],
          "Boundary":["GHB_k","RIV_k","DRN_k"]}
    params = [p for ps in pg.values() for p in ps]
    n_p, n_o = len(params), 40
    J = rng.normal(0,1,(n_o,n_p))
    J[:,0]*=5; J[:,1]*=4; J[:,7]*=3
    J[:,-1]*=0.2; J[:,-2]*=0.15
    J[:,3] = J[:,2]*0.95 + rng.normal(0,0.05,n_o)
    W = np.diag(rng.uniform(0.5,2.0,n_o))
    JtWJ = J.T@W@J
    U,S,Vt = np.linalg.svd(J,full_matrices=False)
    ident = np.array([float(np.sum(Vt[:,i]**2)) for i in range(n_p)])
    try: cov = np.linalg.inv(JtWJ+1e-6*np.eye(n_p))
    except: cov = np.eye(n_p)
    std = np.sqrt(np.abs(np.diag(cov)))
    corr = cov/np.outer(std+1e-12,std+1e-12)
    corr = np.clip(corr,-1,1)
    return dict(param_groups=list(pg.keys()),
                _param_names=params,
                _param_group_map={p:g for g,ps in pg.items() for p in ps},
                _ident_scores=ident, _singular_values=S, _vt=Vt,
                covariance=pd.DataFrame(cov,index=params,columns=params),
                _correlation=pd.DataFrame(corr,index=params,columns=params))

PEST_DATA = _mock_data()
COLORS = ["#4a90d9","#2f7a5e","#c0774a","#9b7ec8","#e8a838","#d94a4a"]
BASE = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans, sans-serif",size=11,color="#555"),
            margin=dict(l=54,r=20,t=36,b=44),
            hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))
GRID = dict(showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False,tickfont=dict(size=10))

def _ident_bar(names,scores,pgmap,pgroups):
    pgc = {g:COLORS[i%len(COLORS)] for i,g in enumerate(pgroups)}
    order = np.argsort(scores)[::-1]
    ss,ns = scores[order],[names[i] for i in order]
    gs = [pgmap.get(n,"?") for n in ns]
    fig = go.Figure(go.Bar(x=ns,y=ss,
        marker_color=[pgc.get(g,"#aaa") for g in gs],marker_line_width=0,opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Identifiability = %{y:.3f}<extra></extra>"))
    for thr,col,lbl in [(0.7,"#2f7a5e","Good (0.7)"),(0.4,"#e8a838","Marginal (0.4)"),(0.1,"#d94a4a","Poor (0.1)")]:
        fig.add_hline(y=thr,line=dict(color=col,width=1,dash="dot"),
            annotation_text=lbl,annotation_position="right",annotation_font=dict(size=8,color=col))
    fig.update_layout(**BASE,xaxis=dict(**GRID,tickangle=-40,title="Parameter"),
        yaxis=dict(**GRID,title="Identifiability score",range=[0,1.05]),showlegend=False)
    return fig

def _svd_fig(S):
    S_norm = S/S[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1,len(S)+1)),y=S,name="Singular values",
        mode="lines+markers",line=dict(color="#4a90d9",width=2),marker=dict(size=5),
        hovertemplate="SV #%{x}<br>Value = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(range(1,len(S)+1)),y=S_norm,name="Normalised",
        mode="lines+markers",line=dict(color="#c0774a",width=2,dash="dash"),marker=dict(size=5),
        yaxis="y2",hovertemplate="SV #%{x}<br>Norm = %{y:.4f}<extra></extra>"))
    fig.update_layout(**BASE,xaxis=dict(**GRID,title="Singular value index",dtick=1),
        yaxis=dict(**GRID,title="Singular value"),
        yaxis2=dict(title="Normalised",overlaying="y",side="right",
                    showgrid=False,zeroline=False,tickfont=dict(size=9),range=[0,1.1]),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _corr_fig(corr):
    names = corr.columns.tolist()
    z = corr.values.copy(); np.fill_diagonal(z,np.nan)
    fig = go.Figure(go.Heatmap(z=z,x=names,y=names,colorscale="RdBu_r",zmid=0,zmin=-1,zmax=1,
        colorbar=dict(title=dict(text="ρ",font=dict(size=9)),thickness=12,len=0.8,tickfont=dict(size=8)),
        xgap=1,ygap=1,hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>ρ = %{z:.3f}<extra></extra>"))
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            if not np.isnan(z[i,j]) and abs(z[i,j])>0.95:
                fig.add_annotation(x=names[j],y=names[i],text="!",
                    font=dict(size=10,color="black"),showarrow=False)
    b2 = dict(**BASE); b2["margin"]=dict(l=70,r=20,t=36,b=100)
    fig.update_layout(**b2,
        xaxis=dict(tickfont=dict(size=9),tickangle=-40,showgrid=False,zeroline=False),
        yaxis=dict(tickfont=dict(size=9),showgrid=False,zeroline=False,autorange="reversed"))
    return fig

def _null_fig(Vt,names,n=3):
    fig = go.Figure()
    cols = ["#4a90d9","#c0774a","#9b7ec8"]
    for k in range(min(n,Vt.shape[0])):
        fig.add_trace(go.Bar(name=f"Null vec {k+1}",x=names,y=Vt[-(k+1),:],
            marker_color=cols[k],marker_line_width=0,opacity=0.75,
            hovertemplate=f"Null vec {k+1}<br><b>%{{x}}</b><br>%{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=0,line=dict(color="#888",width=0.8))
    fig.update_layout(**BASE,barmode="group",
        xaxis=dict(tickangle=-40,tickfont=dict(size=9),showgrid=False,zeroline=False),
        yaxis=dict(title="Component",showgrid=True,gridcolor="rgba(0,0,0,0.05)",zeroline=False),
        legend=dict(orientation="h",y=1.05,x=0,font=dict(size=10),bgcolor="rgba(0,0,0,0)"))
    return fig

def _sec(title,sub=None):
    return dmc.Stack(gap=0,mb="xs",children=[
        dmc.Text(title,fw=600,size="sm",style={"color":"#333"}),
        dmc.Text(sub,size="xs",c="dimmed") if sub else html.Span()])

def _kpi(label,value,icon=None,color="teal"):
    return dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},
        children=[dmc.Group(justify="space-between",align="flex-start",children=[
            dmc.Stack(gap=2,children=[
                dmc.Text(label,size="xs",c="dimmed",tt="uppercase",fw=500),
                dmc.Text(value,size="xl",fw=700,style={"fontFamily":"DM Mono, monospace","fontSize":"1.5rem"})]),
            dmc.ThemeIcon(DashIconify(icon=icon,width=18),size="lg",radius="md",variant="light",color=color) if icon else html.Span()])])

def layout(data=None):
    d = data or PEST_DATA
    names=d.get("_param_names",[]); scores=d.get("_ident_scores",np.array([]))
    S_vals=d.get("_singular_values",np.array([])); Vt=d.get("_vt",np.zeros((1,1)))
    corr=d.get("_correlation",pd.DataFrame()); pgmap=d.get("_param_group_map",{})
    pgroups=d.get("param_groups",[])
    n_good=int((scores>=0.7).sum()) if len(scores) else 0
    n_poor=int((scores<0.1).sum()) if len(scores) else 0
    corr_warn=0
    if not corr.empty:
        c=corr.values.copy(); np.fill_diagonal(c,0)
        corr_warn=int((np.abs(c)>0.95).sum()//2)
    cond=float(S_vals[0]/S_vals[-1]) if len(S_vals)>1 and S_vals[-1]>0 else 0
    return dmc.Stack(gap="lg",p="lg",children=[
        dmc.Group(justify="space-between",align="flex-end",children=[
            dmc.Stack(gap=2,children=[
                dmc.Title("Identifiability & Uniqueness",order=3,style={"color":"#1a1a1a"}),
                dmc.Text("SVD-based identifiability, null-space analysis, and parameter correlation matrix",size="sm",c="dimmed")])]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4},spacing="md",children=[
            _kpi("Well-identified",str(n_good),"tabler:check","teal"),
            _kpi("Poorly identified",str(n_poor),"tabler:alert-triangle","red"),
            _kpi("High correlations",str(corr_warn),"tabler:link","orange"),
            _kpi("Condition number",f"{cond:.1f}","tabler:calculator","grape")]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Parameter Identifiability Scores",
                 "0=not identifiable, 1=fully resolved. Green ≥0.7, Amber ≥0.4, Red <0.1"),
            dcc.Graph(figure=_ident_bar(names,scores,pgmap,pgroups),
                      config={"displayModeBar":False},style={"height":"300px"})]),
        dmc.Grid(gutter="md",children=[
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Singular Value Spectrum","Rapid drop = near-singular system"),
                    dcc.Graph(figure=_svd_fig(S_vals),config={"displayModeBar":False},style={"height":"260px"})])),
            dmc.GridCol(span={"base":12,"md":6},children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _sec("Null-Space Directions","Parameter combinations the data cannot constrain"),
                    dcc.Graph(figure=_null_fig(Vt,names),config={"displayModeBar":False},style={"height":"260px"})]))]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _sec("Parameter Correlation Matrix","! = |ρ|>0.95, cannot be estimated independently"),
            dcc.Graph(figure=_corr_fig(corr) if not corr.empty else go.Figure(),
                      config={"displayModeBar":False},style={"height":"360px"})]),
        dmc.Alert(title="Identifiability notes",color="teal",variant="light",radius="md",
            icon=DashIconify(icon="tabler:bulb",width=18),
            children=dmc.List(size="sm",spacing="xs",children=[
                dmc.ListItem(f"{n_poor} parameter(s) poorly identifiable — fix to prior or add regularisation."),
                dmc.ListItem(f"{corr_warn} highly correlated pair(s) (|ρ|>0.95) — cannot be estimated independently."),
                dmc.ListItem(f"Condition number = {cond:.1f}. "+("Well-conditioned." if cond<100 else "Ill-conditioned — SVD truncation recommended."))]))])

def register_callbacks(app): pass
