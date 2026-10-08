"""
PEST Calibration Diagnostics — Contribution Matrix Page
========================================================
C[i,j] = -r_i * J[i,j] * Δθ_j
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify

# ── Mock data ─────────────────────────────────────────────────────────────────
def _mock_data() -> dict:
    rng = np.random.default_rng(42)
    obs_groups = ["Heads", "Flows", "Conc.", "Levels"]
    param_groups = {"Hyd. cond.": ["K_hk1","K_hk2","K_hk3"],
                    "Storage":    ["Sy1","Ss1"],
                    "Recharge":   ["Rch_z1","Rch_z2","Rch_z3"],
                    "Boundary":   ["GHB_k","RIV_k"]}
    all_params = [p for ps in param_groups.values() for p in ps]
    n_obs, n_par = 20, len(all_params)
    obs_names = [f"{g[:2].lower()}{i:02d}" for g in obs_groups for i in range(n_obs//len(obs_groups))]
    J  = rng.normal(0, 0.5, (n_obs, n_par))
    dp = rng.normal(0, 0.3, n_par)
    r  = rng.normal(0, 0.2, n_obs)
    C  = -r[:, None] * J * dp[None, :]
    obs_group_map = {n: g for g in obs_groups for n in obs_names if n[:2]==g[:2].lower()}
    return dict(
        jacobian=pd.DataFrame(J, index=obs_names, columns=all_params),
        parameters=pd.DataFrame([{"name":p,"group":g,"value":float(rng.uniform(0.1,10))}
                                  for g,ps in param_groups.items() for p in ps]),
        residuals=pd.DataFrame({"name":obs_names,
                                 "group":[obs_group_map[n] for n in obs_names],
                                 "residual":r,"weighted_res":r}),
        _contribution_matrix=pd.DataFrame(C, index=obs_names, columns=all_params),
        _dp=dp, obs_groups=obs_groups,
        param_groups=list(param_groups.keys()),
        _obs_group_map=obs_group_map,
    )

PEST_DATA = _mock_data()

BASE_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=11, color="#555"),
    margin=dict(l=54,r=20,t=36,b=44),
    hoverlabel=dict(bgcolor="white",bordercolor="#ddd",font_size=11))

def _compute_C(data):
    if "_contribution_matrix" in data:
        return data["_contribution_matrix"]
    J  = data.get("jacobian")
    res = data.get("residuals")
    dp = data.get("_dp")
    if J is None or res is None or dp is None:
        return None
    r = res.set_index("name")["residual"].reindex(J.index).fillna(0).values
    return pd.DataFrame(-r[:,None]*J.values*dp[None,:], index=J.index, columns=J.columns)

def _heatmap_fig(C: pd.DataFrame) -> go.Figure:
    vmax = max(np.abs(C.values).max(), 1e-6)
    fig = go.Figure(go.Heatmap(
        z=C.values, x=C.columns.tolist(), y=C.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
        colorbar=dict(title=dict(text="C[i,j]",font=dict(size=9)),
                      thickness=12,len=0.8,tickfont=dict(size=8)),
        xgap=1, ygap=1,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>C = %{z:.4f}<extra></extra>",
    ))
    row_sums = C.sum(axis=1).values
    fig.add_trace(go.Bar(
        x=row_sums, y=C.index.tolist(), orientation="h",
        marker_color=["#d94a4a" if v<0 else "#2f7a5e" for v in row_sums],
        marker_line_width=0, opacity=0.8, name="Row sum",
        xaxis="x2", yaxis="y",
        hovertemplate="Row sum = %{x:.4f}<extra></extra>",
    ))
    base2 = {k:v for k,v in BASE_LAYOUT.items() if k!="margin"}
    base2["legend"] = dict(orientation="h",y=1.05,x=0,font=dict(size=9),bgcolor="rgba(0,0,0,0)")
    fig.update_layout(
        **base2,
        xaxis=dict(side="bottom", tickfont=dict(size=9), tickangle=-40,
                   showgrid=False, zeroline=False, domain=[0,0.78]),
        xaxis2=dict(title="Σⱼ C[i,j]", tickfont=dict(size=9),
                    showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                    zeroline=True, zerolinecolor="#ccc", domain=[0.82,1.0]),
        yaxis=dict(tickfont=dict(size=9), showgrid=False, zeroline=False),
        barmode="overlay",
        margin=dict(l=70,r=20,t=36,b=100),
    )
    return fig

def _col_sum_fig(C: pd.DataFrame) -> go.Figure:
    col_sums = C.sum(axis=0)
    fig = go.Figure(go.Bar(
        x=col_sums.index.tolist(), y=col_sums.values,
        marker_color=["#d94a4a" if v<0 else "#2f7a5e" for v in col_sums.values],
        marker_line_width=0, opacity=0.82,
        hovertemplate="<b>%{x}</b><br>Σᵢ C[i,j] = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="#888", width=0.8))
    fig.update_layout(**BASE_LAYOUT,
        xaxis=dict(tickangle=-40, tickfont=dict(size=9), showgrid=False, zeroline=False),
        yaxis=dict(title="Σᵢ C[i,j]", showgrid=True, gridcolor="rgba(0,0,0,0.05)", zeroline=False),
        showlegend=False)
    return fig

def _top_contributors_fig(C: pd.DataFrame, n=10) -> go.Figure:
    flat = C.stack().reset_index()
    flat.columns = ["obs","param","value"]
    flat["abs"] = flat["value"].abs()
    top = flat.nlargest(n, "abs")
    labels = [f"{r.obs}×{r.param}" for _, r in top.iterrows()]
    fig = go.Figure(go.Bar(
        x=top["value"].values[::-1], y=labels[::-1], orientation="h",
        marker_color=["#d94a4a" if v<0 else "#2f7a5e" for v in top["value"].values[::-1]],
        marker_line_width=0, opacity=0.82,
        hovertemplate="<b>%{y}</b><br>C = %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#888", width=0.8))
    base3={k:v for k,v in BASE_LAYOUT.items() if k!="margin"}
    fig.update_layout(**base3,
        xaxis=dict(title="Contribution C[i,j]", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=False),
        yaxis=dict(tickfont=dict(size=9), showgrid=False, zeroline=False),
        margin=dict(l=120,r=20,t=36,b=44), showlegend=False)
    return fig

def _section(title, subtitle=None):
    return dmc.Stack(gap=0, mb="xs", children=[
        dmc.Text(title, fw=600, size="sm", style={"color":"#333","letterSpacing":"0.02em"}),
        dmc.Text(subtitle, size="xs", c="dimmed") if subtitle else html.Span(),
    ])

def _kpi(label, value, icon=None, color="teal"):
    return dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},
        children=[dmc.Group(justify="space-between",align="flex-start",children=[
            dmc.Stack(gap=2,children=[
                dmc.Text(label,size="xs",c="dimmed",tt="uppercase",fw=500,style={"letterSpacing":"0.05em"}),
                dmc.Text(value,size="xl",fw=700,style={"fontFamily":"DM Mono, monospace","fontSize":"1.5rem"}),
            ]),
            dmc.ThemeIcon(DashIconify(icon=icon,width=18),size="lg",radius="md",
                variant="light",color=color) if icon else html.Span(),
        ])])

def layout(data: dict = None) -> dmc.Stack:
    d = data or PEST_DATA
    C = _compute_C(d)
    if C is None:
        return dmc.Alert("Jacobian, residuals, or parameter changes not available.",
                         color="orange", title="Data missing")
    col_sums = C.sum(axis=0)
    row_sums = C.sum(axis=1)
    net = C.values.sum()
    n_helping  = int((col_sums > 0).sum())
    n_hurting  = int((col_sums < 0).sum())
    max_cell   = C.stack().abs().idxmax()

    return dmc.Stack(gap="lg", p="lg", children=[
        dmc.Group(justify="space-between", align="flex-end", children=[
            dmc.Stack(gap=2, children=[
                dmc.Title("Contribution Matrix", order=3, style={"color":"#1a1a1a"}),
                dmc.Text("C[i,j] = −rᵢ · J[i,j] · Δθⱼ  —  contribution of parameter j to reducing residual i",
                         size="sm", c="dimmed"),
            ]),
        ]),
        dmc.Divider(),
        dmc.SimpleGrid(cols={"base":2,"sm":4}, spacing="md", children=[
            _kpi("Net contribution", f"{net:+.3f}", icon="tabler:sum", color="teal"),
            _kpi("Helping params",   str(n_helping), icon="tabler:arrow-up", color="green"),
            _kpi("Hurting params",   str(n_hurting), icon="tabler:arrow-down", color="red"),
            _kpi("Max cell",f"{max_cell[0]}×{max_cell[1]}",icon="tabler:flame",color="orange"),
        ]),
        dmc.Paper(shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
            _section("Contribution Heatmap + Row Sums",
                     "Red = reduces residual (good); Blue = increases residual (bad). Right panel = Σⱼ C[i,j] per observation"),
            dcc.Graph(id="con-heatmap", figure=_heatmap_fig(C),
                      config={"displayModeBar":False}, style={"height":"480px"}),
        ]),
        dmc.Grid(gutter="md", children=[
            dmc.GridCol(span={"base":12,"md":7}, children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _section("Column Sums — Parameter Net Contribution",
                             "Σᵢ C[i,j] per parameter — net effect across all observations"),
                    dcc.Graph(id="con-colsum", figure=_col_sum_fig(C),
                              config={"displayModeBar":False}, style={"height":"240px"}),
                ])),
            dmc.GridCol(span={"base":12,"md":5}, children=dmc.Paper(
                shadow="xs",p="md",radius="md",withBorder=True,style={"borderColor":"#eee"},children=[
                    _section("Top Individual Contributions",
                             "Largest |C[i,j]| cells — the (obs, param) pairs driving the update"),
                    dcc.Graph(id="con-top", figure=_top_contributors_fig(C),
                              config={"displayModeBar":False}, style={"height":"240px"}),
                ])),
        ]),
        dmc.Alert(title="How to read", color="blue", variant="light", radius="md",
            icon=DashIconify(icon="tabler:info-circle", width=18),
            children=dmc.List(size="sm", spacing="xs", children=[
                dmc.ListItem("Red cells: parameter j's update is helping reduce residual i."),
                dmc.ListItem("Blue cells: parameter j's update is making residual i worse."),
                dmc.ListItem("Column sums < 0: that parameter is net-hurting — consider its bounds or prior."),
                dmc.ListItem("Row sums ≈ 0: that observation is being pulled equally in both directions — check weighting."),
            ])),
    ])

def register_callbacks(app): pass
if __name__ == "__main__":
    app = dash.Dash(__name__)
    app.layout = dmc.MantineProvider(theme={"primaryColor":"teal"},
        children=dmc.Container(layout(), size="xl", py="lg"))
    app.run(debug=True)
