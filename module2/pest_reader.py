"""
pest_reader.py
==============
Parses standard PEST / PEST++ output files into a single unified dict
that all diagnostic pages consume.

Supported files
---------------
  .rec   — run record  (Φ history, lambda, iterations)
  .rei   — residuals   (observed, simulated, weighted residuals)
  .par   — parameter values (final + per-iteration if .ipar files present)
  .sen   — sensitivity (composite scaled + one-percent scaled)
  .jco   — Jacobian matrix (binary or ASCII)
  .cov   — parameter covariance matrix

Usage
-----
    from pest_reader import read_pest_run

    data = read_pest_run("path/to/pest_run")   # pass base name, no extension
    # data is a dict — pass directly to any diagnostic page layout()

Dependencies
------------
    pip install numpy pandas pyemu
    pyemu is optional — only needed for .jcb (binary Jacobian)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import warnings as _warnings
import numpy as np
import pandas as pd

# Suppress pyemu/flopy optional-dependency noise
_warnings.filterwarnings("ignore", message=".*flopy.*", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def read_pest_run(base_path: str | Path) -> dict:
    """
    Read all available PEST output files for a given run.

    Parameters
    ----------
    base_path : str or Path
        Base path of the PEST run, without extension.
        E.g.  "/models/mymodel/calib"
        The function will look for calib.rec, calib.rei, calib.par, etc.

    Returns
    -------
    dict with keys (all optional — absent if file missing):
        phi_history      : pd.DataFrame  columns: iteration, phi_total, phi_meas, phi_reg, lambda
        residuals        : pd.DataFrame  columns: name, group, obs, sim, residual, weight, weighted_res
        parameters       : pd.DataFrame  columns: name, group, value, lower, upper  (final iteration)
        param_history    : pd.DataFrame  index=iteration, columns=param names  (if .ipar files found)
        sensitivity      : pd.DataFrame  columns: name, group, css, oss_mean
        jacobian         : pd.DataFrame  index=obs_names, columns=param_names
        covariance       : pd.DataFrame  index=param_names, columns=param_names
        obs_groups       : list[str]
        param_groups     : list[str]
        n_iter           : int
        iterations       : np.ndarray
        phi_total        : np.ndarray
        phi_meas         : np.ndarray
        phi_reg          : np.ndarray
        lambda_vals      : np.ndarray
        rel_improve      : np.ndarray
        phi_by_group     : np.ndarray   (same order as obs_groups)
        group_phi_iter   : np.ndarray   shape (n_groups, n_iter)
        converged        : bool
        phimlim          : float | None
        meta             : dict          version, pest_mode, etc.
    """
    base = Path(base_path)
    data: dict = {}

    # Parse each file type, silently skip if not found
    _try(data, _parse_rec,  base.with_suffix(".rec"))
    _try(data, _parse_rei,  base.with_suffix(".rei"))
    _try(data, _parse_par,  base.with_suffix(".par"))
    _try(data, _parse_sen,  base.with_suffix(".sen"))
    _try(data, _parse_jco,  base.with_suffix(".jco"))
    _try(data, _parse_cov,  base.with_suffix(".cov"))

    # Try binary Jacobian if ASCII not found
    if "jacobian" not in data:
        _try(data, _parse_jcb, base.with_suffix(".jcb"))

    # Derive per-iteration parameter history from .ipar files
    _try(data, _parse_ipar_series, base)

    # Derive convenience arrays from phi_history DataFrame
    _derive_phi_arrays(data)

    # Derive group-level Φ arrays from residuals
    _derive_group_phi(data)

    return data


# ──────────────────────────────────────────────────────────────────────────────
# HELPER — silently handle missing / malformed files
# ──────────────────────────────────────────────────────────────────────────────

def _try(data: dict, fn, *args):
    try:
        fn(data, *args)
    except FileNotFoundError:
        pass
    except Exception as exc:
        fname = args[0] if args else "?"
        warnings.warn(f"pest_reader: could not parse {fname}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# .rec — Run record
# ──────────────────────────────────────────────────────────────────────────────
#
# PEST run records are free-form text.  The key patterns are:
#
#   Model calls so far =  ...   (marks start of new iteration block)
#   Sum of squared weighted residuals (ie phi) =  ...
#   Contribution to phi from observation group "xxx" =  ...
#   Lambda =  ...
#   Current lambda =  ...
#   CONVERGED   /   FAILED TO CONVERGE
#   PHIMLIM =  ...

_RE_ITER      = re.compile(r"Model calls so far\s*=\s*(\d+)", re.I)
_RE_PHI       = re.compile(r"Sum of squared weighted residuals.*?=\s*([\d.eE+\-]+)", re.I)
_RE_PHI_MEAS  = re.compile(r"Contribution to phi from observation group.*?=\s*([\d.eE+\-]+)", re.I)
_RE_LAMBDA    = re.compile(r"(?:Current\s+)?[Ll]ambda\s*=\s*([\d.eE+\-]+)")
_RE_PHIMLIM   = re.compile(r"PHIMLIM\s*=\s*([\d.eE+\-]+)", re.I)
_RE_CONVERGED = re.compile(r"\bCONVERGED\b", re.I)
_RE_FAILED    = re.compile(r"FAILED TO CONVERGE", re.I)
_RE_REG       = re.compile(r"Contribution to phi from regularisation\s*=\s*([\d.eE+\-]+)", re.I)
_RE_GRP_PHI   = re.compile(
    r'Contribution to phi from observation group\s+"([^"]+)"\s*=\s*([\d.eE+\-]+)', re.I
)
_RE_VERSION   = re.compile(r"PEST\+?\+?\s+Version\s+([\d.]+)", re.I)
_RE_MODE      = re.compile(r"PEST is running in ([\w\s]+) mode", re.I)


def _parse_rec(data: dict, path: Path) -> None:
    text = path.read_text(errors="replace")

    rows        = []          # one row per iteration
    phimlim     = None
    converged   = False
    meta        = {}

    # ── metadata ──────────────────────────────────────────────────
    m = _RE_VERSION.search(text)
    if m:
        meta["version"] = m.group(1)
    m = _RE_MODE.search(text)
    if m:
        meta["pest_mode"] = m.group(1).strip()

    m = _RE_PHIMLIM.search(text)
    if m:
        phimlim = float(m.group(1))

    if _RE_CONVERGED.search(text):
        converged = True

    # ── split into iteration blocks by "Model calls so far" ───────
    blocks = _RE_ITER.split(text)
    # blocks = [preamble, model_calls_1, block_1, model_calls_2, block_2 …]
    # We only want odd-indexed blocks (the text after each model-calls number)
    iter_num = 0
    for i in range(1, len(blocks), 2):
        model_calls = int(blocks[i])
        block = blocks[i + 1] if i + 1 < len(blocks) else ""

        iter_num += 1

        phi_total = np.nan
        phi_meas  = np.nan
        phi_reg   = np.nan
        lam       = np.nan
        group_phi: dict[str, float] = {}

        m = _RE_PHI.search(block)
        if m:
            phi_total = float(m.group(1))

        m = _RE_REG.search(block)
        if m:
            phi_reg = float(m.group(1))

        for gm in _RE_GRP_PHI.finditer(block):
            group_phi[gm.group(1)] = float(gm.group(2))

        # Measurement phi = sum of group contributions
        if group_phi:
            phi_meas = sum(group_phi.values())
        elif not np.isnan(phi_total) and not np.isnan(phi_reg):
            phi_meas = phi_total - phi_reg

        # Lambda — take the LAST occurrence in this block
        lam_matches = _RE_LAMBDA.findall(block)
        if lam_matches:
            lam = float(lam_matches[-1])

        rows.append({
            "iteration":   iter_num,
            "model_calls": model_calls,
            "phi_total":   phi_total,
            "phi_meas":    phi_meas,
            "phi_reg":     phi_reg,
            "lambda":      lam,
            "group_phi":   group_phi,
        })

    if not rows:
        raise ValueError("No iteration blocks found in .rec file")

    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "group_phi"}
        for r in rows
    ])
    df.set_index("iteration", inplace=True)

    data["phi_history"] = df
    data["converged"]   = converged
    data["phimlim"]     = phimlim
    data["meta"]        = meta

    # Per-group phi at last iteration
    if rows and rows[-1]["group_phi"]:
        gp = rows[-1]["group_phi"]
        data["_group_phi_last"] = gp        # used later in _derive_group_phi

    # Build group_phi_iter matrix (n_groups × n_iter)
    # Collect all group names seen across all iterations
    all_groups = []
    for r in rows:
        for g in r["group_phi"]:
            if g not in all_groups:
                all_groups.append(g)

    if all_groups:
        matrix = np.full((len(all_groups), len(rows)), np.nan)
        for col_i, r in enumerate(rows):
            for row_i, g in enumerate(all_groups):
                if g in r["group_phi"]:
                    matrix[row_i, col_i] = r["group_phi"][g]
        data["_group_phi_iter_raw"] = matrix
        data["_all_groups_rec"]     = all_groups


# ──────────────────────────────────────────────────────────────────────────────
# .rei — Residuals
# ──────────────────────────────────────────────────────────────────────────────
#
# Standard format (space-delimited, header on line 2):
#
#   Name        Group    Measured     Modelled   Residual  Weight
#   obs1        heads    10.23        10.15       0.08      1.0
#   ...

def _parse_rei(data: dict, path: Path) -> None:
    rows = []
    header_found = False

    with path.open(errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue

            # Header line detection
            low = line.lower()
            if "name" in low and ("group" in low or "measured" in low):
                header_found = True
                continue

            if not header_found:
                continue

            parts = line.split()
            # Expect at least 6 columns: name group measured modelled residual weight
            if len(parts) < 5:
                continue

            try:
                name     = parts[0]
                group    = parts[1]
                measured = float(parts[2])
                modelled = float(parts[3])
                residual = float(parts[4])
                weight   = float(parts[5]) if len(parts) > 5 else 1.0
            except (ValueError, IndexError):
                continue

            rows.append({
                "name":         name,
                "group":        group,
                "obs":          measured,
                "sim":          modelled,
                "residual":     residual,
                "weight":       weight,
                "weighted_res": residual * weight,
            })

    if rows:
        df = pd.DataFrame(rows)
        data["residuals"]   = df
        data["obs_groups"]  = sorted(df["group"].unique().tolist())


# ──────────────────────────────────────────────────────────────────────────────
# .par — Parameter values (final iteration)
# ──────────────────────────────────────────────────────────────────────────────
#
# Format:
#   single header line
#   name   value   scale  offset
#
# PEST++ uses slightly different format; we handle both.

def _parse_par(data: dict, path: Path) -> None:
    rows = []
    header_skipped = False

    with path.open(errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_skipped:
                header_skipped = True
                continue

            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rows.append({
                    "name":   parts[0],
                    "value":  float(parts[1]),
                    "scale":  float(parts[2]) if len(parts) > 2 else 1.0,
                    "offset": float(parts[3]) if len(parts) > 3 else 0.0,
                })
            except ValueError:
                continue

    if rows:
        data["parameters"] = pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# .sen — Sensitivity file
# ──────────────────────────────────────────────────────────────────────────────
#
# Two sections:
#   1. Composite scaled sensitivities (CSS) — one value per parameter
#   2. One-percent scaled sensitivities (OSS) — matrix obs × param
#
# The file starts with a header block, then:
#   "COMPOSITE SCALED SENSITIVITIES"
#   name   css
#   ...
#   "ONE-PERCENT SCALED SENSITIVITIES"  (optional)
#   param1   param2   ...
#   obs1     val      val
#   ...

def _parse_sen(data: dict, path: Path) -> None:
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    css_rows = []
    oss_header = []
    oss_rows   = []
    mode = None

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        if "composite scaled sensitivit" in low:
            mode = "css"
            continue
        if "one-percent scaled sensitivit" in low:
            mode = "oss_header"
            continue

        if not stripped or stripped.startswith("-"):
            continue

        if mode == "css":
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    css_rows.append({"name": parts[0], "css": float(parts[1])})
                except ValueError:
                    pass

        elif mode == "oss_header":
            oss_header = stripped.split()
            mode = "oss"

        elif mode == "oss":
            parts = stripped.split()
            if len(parts) == len(oss_header) + 1:
                obs_name = parts[0]
                try:
                    vals = [float(v) for v in parts[1:]]
                    oss_rows.append({"obs": obs_name, **dict(zip(oss_header, vals))})
                except ValueError:
                    pass

    if css_rows:
        css_df = pd.DataFrame(css_rows).set_index("name")

        if oss_rows:
            oss_df = pd.DataFrame(oss_rows).set_index("obs")
            # Mean absolute OSS per parameter
            css_df["oss_mean"] = oss_df.abs().mean()
            data["oss_matrix"] = oss_df

        data["sensitivity"] = css_df.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# .jco — Jacobian matrix (ASCII)
# ──────────────────────────────────────────────────────────────────────────────
#
# Format:
#   nobs   npar   [optional flags]
#   obs1   param1   value   obs1   param2   value …  (all pairs on one line, or wrapped)
#
# PEST writes the Jacobian as a series of (obs_index, param_index, value)
# triplets, OR as a dense matrix.  The most common ASCII form is:
#
#   * parameter names
#   pname1
#   pname2
#   ...
#   * observation names
#   oname1
#   ...
#   * Jacobian matrix
#   v11  v12  v13 ...     <- row = observation
#   v21  ...

def _parse_jco(data: dict, path: Path) -> None:
    text = path.read_text(errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    param_names = []
    obs_names   = []
    matrix_rows = []
    section = None

    for line in lines:
        low = line.lower()
        if low.startswith("* parameter"):
            section = "params"
            continue
        elif low.startswith("* observation"):
            section = "obs"
            continue
        elif low.startswith("* jacobian") or low.startswith("* matrix"):
            section = "matrix"
            continue

        if section == "params":
            param_names.append(line.split()[0])
        elif section == "obs":
            obs_names.append(line.split()[0])
        elif section == "matrix":
            try:
                matrix_rows.append([float(v) for v in line.split()])
            except ValueError:
                pass

    if matrix_rows and param_names and obs_names:
        mat = np.array(matrix_rows)
        if mat.shape == (len(obs_names), len(param_names)):
            data["jacobian"] = pd.DataFrame(
                mat, index=obs_names, columns=param_names
            )
            if "param_groups" not in data:
                data["param_groups"] = param_names


# ──────────────────────────────────────────────────────────────────────────────
# .jcb — Binary Jacobian (via pyemu)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_jcb(data: dict, path: Path) -> None:
    try:
        import pyemu
    except ImportError:
        warnings.warn("pest_reader: pyemu not installed — cannot read .jcb binary Jacobian")
        return

    jco = pyemu.Jco.from_binary(str(path))
    df = jco.to_dataframe()
    data["jacobian"]    = df
    data["param_groups"] = df.columns.tolist()


# ──────────────────────────────────────────────────────────────────────────────
# .cov — Parameter covariance matrix
# ──────────────────────────────────────────────────────────────────────────────
#
# Format:
#   * covariance matrix
#   nparams
#   pname1   pname2   ...
#   v11  v12  ...
#   ...

def _parse_cov(data: dict, path: Path) -> None:
    text = path.read_text(errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    param_names = []
    matrix_rows = []
    section = None

    for line in lines:
        low = line.lower()
        if "covariance" in low and line.startswith("*"):
            section = "size"
            continue

        if section == "size":
            try:
                int(line)            # skip size line
                section = "header"
            except ValueError:
                pass
            continue

        if section == "header":
            param_names = line.split()
            section = "matrix"
            continue

        if section == "matrix":
            try:
                matrix_rows.append([float(v) for v in line.split()])
            except ValueError:
                pass

    if matrix_rows and param_names:
        mat = np.array(matrix_rows)
        if mat.shape[0] == mat.shape[1] == len(param_names):
            data["covariance"] = pd.DataFrame(
                mat, index=param_names, columns=param_names
            )


# ──────────────────────────────────────────────────────────────────────────────
# .ipar — Per-iteration parameter values
# ──────────────────────────────────────────────────────────────────────────────
#
# PEST writes  run.1.par, run.2.par …  (or run.ipar.001 etc.)
# We glob for them and stack into a DataFrame.

def _parse_ipar_series(data: dict, base: Path) -> None:
    # Try both naming conventions
    patterns = [
        sorted(base.parent.glob(f"{base.name}.*.par")),
        sorted(base.parent.glob(f"{base.name}.ipar.*")),
    ]

    files = next((p for p in patterns if p), [])
    if not files:
        return

    frames = {}
    for f in files:
        # Extract iteration number from filename
        m = re.search(r"\.(\d+)(?:\.par)?$", f.name)
        if not m:
            continue
        it = int(m.group(1))
        tmp: dict = {}
        _parse_par(tmp, f)
        if "parameters" in tmp:
            frames[it] = tmp["parameters"].set_index("name")["value"]

    if frames:
        df = pd.DataFrame(frames).T
        df.index.name = "iteration"
        df.sort_index(inplace=True)
        data["param_history"] = df


# ──────────────────────────────────────────────────────────────────────────────
# DERIVED ARRAYS — convenience scalars / arrays for charting
# ──────────────────────────────────────────────────────────────────────────────

def _derive_phi_arrays(data: dict) -> None:
    if "phi_history" not in data:
        return

    df = data["phi_history"]
    data["n_iter"]      = len(df)
    data["iterations"]  = df.index.to_numpy()
    data["phi_total"]   = df["phi_total"].to_numpy()
    data["phi_meas"]    = df["phi_meas"].to_numpy()
    data["phi_reg"]     = df["phi_reg"].to_numpy()
    data["lambda_vals"] = df["lambda"].to_numpy()

    phi = df["phi_total"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(phi[:-1] != 0, np.diff(phi) / phi[:-1] * 100, np.nan)
    data["rel_improve"] = rel


def _derive_group_phi(data: dict) -> None:
    """Build obs_groups, phi_by_group, group_phi_iter from parsed sources."""

    # Prefer per-group Φ from .rec (most accurate)
    if "_all_groups_rec" in data:
        data["obs_groups"]      = data.pop("_all_groups_rec")
        data["phi_by_group"]    = data.pop("_group_phi_iter_raw")[:, -1]
        data["group_phi_iter"]  = data.pop("_group_phi_iter_raw", np.array([]))
        return

    # Fall back to residuals
    if "residuals" not in data:
        return

    res = data["residuals"]
    groups = data.get("obs_groups", sorted(res["group"].unique().tolist()))
    data["obs_groups"] = groups

    phi_by_group = np.array([
        (res.loc[res["group"] == g, "weighted_res"] ** 2).sum()
        for g in groups
    ])
    data["phi_by_group"] = phi_by_group

    # Without iteration-level residual archives we can only provide the last iter
    # Broadcast to shape (n_groups, n_iter) as constant columns
    if "n_iter" in data:
        data["group_phi_iter"] = np.outer(phi_by_group, np.ones(data["n_iter"]))


# ──────────────────────────────────────────────────────────────────────────────
# CLI — quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pprint

    if len(sys.argv) < 2:
        print("Usage: python pest_reader.py <base_path>")
        print("  e.g. python pest_reader.py /models/calib/myrun")
        sys.exit(0)

    result = read_pest_run(sys.argv[1])
    summary = {
        k: (
            f"DataFrame {v.shape}" if isinstance(v, pd.DataFrame) else
            f"ndarray {v.shape}"   if isinstance(v, np.ndarray)  else
            v
        )
        for k, v in result.items()
        if not k.startswith("_")
    }
    pprint.pp(summary)
