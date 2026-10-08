import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def message_figure(text=None, height=220):
    """
    Placeholder shown instead of a plot (before plotting, or when data is missing):
    a plain centred message without axes. With no text, an empty placeholder.
    """
    fig = go.Figure()
    fig.update_layout(
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    if text:
        fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color="#868e96"))
    return fig

# colors of iteration series ("Iteration Colors" selector)
palettes = {
    "Jet": px.colors.sequential.Jet,
    "Plotly": px.colors.qualitative.Plotly,
    "Set1": px.colors.qualitative.Set1,
    "Dark24": px.colors.qualitative.Dark24,
    "Alphabet": px.colors.qualitative.Alphabet,
}

# continuous color scales for matrices and signed values ("Color Scale" selector)
colorscales = ["RdBu", "BrBG", "RdYlBu", "Jet", "Viridis", "Plotly3"]

stat_names = "RMSE R2 MeanAbsResid MeanResid MinResid MaxResid".split()


def residual_stats(ts):
    """
    Fit statistics (in stat_names order) of the targets in ts with weight > 0;
    zero-weight targets are excluded whether or not they are plotted.
    """
    ts = ts[ts["weight"] > 0]
    r = ts["residual"]
    if len(ts) == 0:
        return [np.nan] * len(stat_names)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.corrcoef(ts["measured"], ts["modelled"])[0, 1] ** 2 if len(ts) > 1 else np.nan
    return [np.sqrt((r**2).mean()), r2, r.abs().mean(), r.mean(), r.min(), r.max()]
