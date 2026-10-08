import numpy as np
import plotly.express as px

palettes = {
    "Jet": px.colors.sequential.Jet,
    "Plotly": px.colors.qualitative.Plotly,
    "Set1": px.colors.qualitative.Set1,
    "Dark24": px.colors.qualitative.Dark24,
    "Alphabet": px.colors.qualitative.Alphabet,
}

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
