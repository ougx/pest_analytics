"""common functions for updating layout components
"""
from dash import no_update
import dash_mantine_components as dmc
from utils import parse_zip


def about(md, label="About this analysis"):
    """collapsed panel holding a page's explanation, so the controls and plots come first"""
    return dmc.Accordion(
        dmc.AccordionItem([
            dmc.AccordionControl(label),
            dmc.AccordionPanel(md),
        ], value="about"),
        value=None,
        variant="contained",
        chevronPosition="left",
        mb=10,
    )

def update_groups(data, default_value="obsgrp"):
    """add observation group columns"""
    if data is not None:
        pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, False)
        if pst:
            return [
                ["obsgrp"] + [c for c in obs.columns if c.lower().startswith("grp")],
                default_value if default_value in obs.columns else "obsgrp"
            ]
    return [no_update, no_update]

def update_weights(data, default_value="weight"):
    """add observation weight columns"""
    if data is not None:
        pst, par, obs, pars, reis, jcos = parse_zip(data, False, False, False)
        if pst:
            weight_cols = [c for c in obs.columns if c.lower().startswith("weight")]
            return [
                weight_cols,
                default_value if default_value in obs.columns else weight_cols[0]
            ]
    return [no_update, no_update]

def update_obs(data, otype, grp, filter_zeroweight=True):
    """add observations or observation groups to a multiselect"""
    if data is not None:
        pst, par, obs, pars, jcos, reis = parse_zip(data, False, False, False)
        if obs is not None:
            if filter_zeroweight:
                obs = obs[obs.weight>0]
            if otype == "Observations":
                obss = []
                for gp, df in obs.groupby("obsgrp"):
                    obss.append({"group": gp, "items": df.obsname.tolist()})
            else:
                obss = [gp for gp in obs[grp].unique().astype(str) if not gp.startswith("regul")]
            # print("obss", obss)
            return obss
    return no_update


def update_jco_iter(data, need_rei=False):
    """
    Jacobian iteration selector: [options, value] with the last iteration selected.
    need_rei: only offer iterations with the matching residuals (.rei.(N-1) or .rei.N).
    """
    if data is not None:
        pst, par, obs, pars, reis, jcos = parse_zip(data, False, need_rei, True)
        if pst and jcos:
            iters = [str(i) for i in sorted(jcos) if not need_rei or (i - 1) in reis or i in reis]
            if iters:
                return [iters, iters[-1]]
    return [no_update, no_update]


def update_niter(data, loadpar=False, loadrei=False, loadjco=False, ):
    """update iteration selector"""
    if data is not None:
        pst, par, obs, pars, reis, jcos = parse_zip(data, loadpar, loadrei, loadjco)
        if pst:
            n = min(len(pars) if loadpar else 100, len(reis) if loadrei else 100, len(jcos) if loadjco else 100)
            return [str(i+1) for i in range(n)]
        return no_update
