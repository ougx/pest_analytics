import base64
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _array(v):
    """Plain values from figure JSON, decoding Plotly's {"dtype", "bdata"} typed arrays."""
    if isinstance(v, dict) and "bdata" in v:
        a = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"]))
        if v.get("shape"):
            a = a.reshape([int(n) for n in str(v["shape"]).split(",")])
        return a
    return v


def figure_to_frame(fig):
    """
    The data behind a figure (as sent back by dcc.Graph) as a table for CSV download:
    a heatmap as its matrix (overlay markers ignored), a table as its cells, a Sankey as
    its links, and other traces as one row per point. Returns None for a figure without data.
    """
    traces = (fig or {}).get("data") or []
    heatmaps = [t for t in traces if t.get("type") == "heatmap"]
    if heatmaps:
        t = heatmaps[0]
        z = np.asarray(_array(t.get("z")), dtype=float)
        index = _array(t.get("y")) if t.get("y") is not None else range(z.shape[0])
        columns = _array(t.get("x")) if t.get("x") is not None else range(z.shape[1])
        return pd.DataFrame(z, index=list(index), columns=list(columns))
    frames = []
    for t in traces:
        kind = t.get("type", "scatter")
        if kind == "table":
            header = [str(h) for h in _array(t["header"]["values"])]
            cells = [list(_array(c)) for c in _array(t["cells"]["values"])]
            frames.append(pd.DataFrame(dict(zip(header, cells))))
        elif kind == "sankey":
            labels = list(_array(t["node"]["label"]))
            link = t["link"]
            df = pd.DataFrame(dict(
                source=[labels[i] for i in _array(link["source"])],
                target=[labels[i] for i in _array(link["target"])],
                value=list(_array(link["value"])),
            ))
            if link.get("customdata") is not None:
                df["details"] = [str(c).replace("<br>", "; ") for c in _array(link["customdata"])]
            frames.append(df)
        else:
            cols = {k: _array(t[k]) for k in ("x", "y", "z", "text", "customdata")
                    if t.get(k) is not None and not isinstance(t.get(k), str)}
            if not cols:
                continue
            n = max(len(v) for v in cols.values())
            df = pd.DataFrame({"trace": [t.get("name", kind)] * n})
            for k, v in cols.items():
                v = np.asarray(v, dtype=object) if not isinstance(v, np.ndarray) else v
                if v.ndim == 2:   # e.g. customdata with several columns
                    for j in range(v.shape[1]):
                        df[f"{k}{j}"] = v[:, j]
                elif len(v) == n:
                    df[k] = v
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


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
