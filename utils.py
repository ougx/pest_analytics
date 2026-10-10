import base64
import functools
import hashlib
import io
import os
import re
import tempfile
import threading
import time
import zipfile
from collections import OrderedDict
import pandas as pd
import numpy as np
from collections.abc import Iterable

def isiterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def read_df(buf, header=None, **kwargs):
    return pd.read_csv(io.StringIO(buf), sep='\\s+', header=header, **kwargs)

def read_par(f):
    """
    read par file into a dataframe (lines up to the first blank line after the header).
    """
    f.readline()
    lines = ''
    for l in f:
        l = l.decode().strip()
        if not l:
            break
        lines += l + '\n'
    return read_df(lines, names='parname parval scale offset'.split())

def read_rei(f):
    """read a residuals file: the table from its "Name Group Measured ..." header line"""
    for l in f:
        l = l.decode().lower().strip()
        if l.startswith('name '):
            break
    else:
        raise ValueError("no 'Name Group Measured Modelled ...' header line in the residuals file")
    lines = l + '\n'
    for l in f:
        l = l.decode().strip()
        if not l:
            break
        lines += l + '\n'
    return read_df(lines, header=0).rename(columns={"name":"obsname", "group":"obsgrp"})

def read_jco(fbuf, offset=0):
    NESPAR,NXROW,NVAL = np.frombuffer(fbuf, "i4", 3)
    offset = 4*3
    if NESPAR > 0:
        vals = np.frombuffer(fbuf, [('i','i4'), ('j','i4'),('val', 'f8')], abs(NVAL), offset=offset); offset += abs(NVAL) * 16
        dat = np.zeros([abs(NXROW), abs(NESPAR)])
        dat[vals['i'],vals['j']] = vals['val']
        pars = np.frombuffer(fbuf, 'S200', abs(NESPAR), offset=offset).astype(str); offset += abs(NESPAR) * 200
        obss = np.frombuffer(fbuf, 'S200', abs(NXROW), offset=offset).astype(str)
    else:
        vals = np.frombuffer(fbuf, [('idx','i4'),('val', 'f8')], abs(NVAL), offset=offset); offset += abs(NVAL) * 12
        dat = np.zeros(abs(NESPAR*NXROW))
        dat[vals['idx']-1] = vals['val']
        dat = dat.reshape([-1, abs(NXROW)]).T
        pars = np.frombuffer(fbuf, 'S12', abs(NESPAR), offset=offset).astype(str); offset += abs(NESPAR) * 12
        obss = np.frombuffer(fbuf, 'S20', abs(NXROW), offset=offset).astype(str)
    return pd.DataFrame(dat, index=[s.strip() for s in obss], columns=[s.strip() for s in pars])


class hpfile:
    def __init__(self, fbuf, ):
        offset = 0
        self.NPAR,self.NOBS,self.NUML = np.frombuffer(fbuf, 'i4', 3); offset += 4*3
        self.PVAL   = np.frombuffer(fbuf, 'f8', self.NPAR, offset=offset); offset += self.NPAR * 8
        self.REFOBS = np.frombuffer(fbuf, 'f8', self.NOBS, offset=offset); offset += self.NOBS * 8
        self.OBSN1  = np.frombuffer(fbuf, 'i4', self.NOBS, offset=offset); offset += self.NOBS * 4
        self.OBSN2  = np.frombuffer(fbuf, 'i4', self.NOBS, offset=offset); offset += self.NOBS * 4
        self.IIOBS  = np.frombuffer(fbuf, 'i4', self.NOBS, offset=offset); offset += self.NOBS * 4
        self.LL     = np.frombuffer(fbuf, 'i4', self.NUML, offset=offset)


def pst_section(f, title):
    """
    Decoded, non-blank lines of a "* title" section of a control file (f: binary file),
    up to the next section header.
    """
    for l in f:
        l = l.decode().strip()
        if l.lower().startswith("*") and l[1:].strip().lower().startswith(title):
            if "external" in l.lower():
                raise ValueError(f"'{l}': sections read from external files are not supported")
            break
    else:
        raise ValueError(f"no '* {title}' section in the control file")
    for l in f:
        l = l.decode().strip()
        if not l:
            continue
        if l.startswith("*"):
            break
        yield l


def read_par_in_pst(f):
    lines = ''
    tieds = {}
    for l in pst_section(f, "parameter data"):
        if len(l.replace(',', ' ').replace('\t', ' ').split()) < 5: # skip tied parameters
            tt = l.replace(',', ' ').replace('\t', ' ').lower().split()
            tieds[tt[0]] = tt[1]
            continue
        lines += l + '\n'
    par = read_df(lines, usecols=range(10),
                  names='parname partrans parchglim parval0 parlbnd parubnd pargrp scale offset dercom'.split())
    par['parname']  = par['parname'].str.lower()
    par['partrans'] = par['partrans'].str.lower()
    par['pargrp']    = par['pargrp'].str.lower()
    par['tiedto'] = ""
    for i,r in par.iterrows():
            if r["partrans"] == "tied":
                par.loc[i, 'tiedto'] = tieds[r["parname"]]
    return par

def read_obs_in_pst(f):
    lines = ''.join(l + '\n' for l in pst_section(f, "observation data"))
    obs = read_df(lines, usecols=range(4), names='obsname obsval weight obsgrp'.split())
    obs['obsname'] = obs['obsname'].str.lower()
    obs['obsgrp']   = obs['obsgrp'].str.lower()
    return obs

def check_data(data):
    # print("data", data)
    hasData = False
    if data and isinstance(data, dict):
        hasData = any([v for k, v in data.items()])
    return hasData


# ---------------------------------------------------------------------------
# Server-side storage and caching of parsed PEST data.
# The browser only holds {"filename", "upload", "pestpath"}: an uploaded zip is
# sent to the server once (on "Load Data"), saved under its content hash, and
# every callback reads the parsed result from an in-memory cache.
# ---------------------------------------------------------------------------

UPLOAD_DIR = os.environ.get("PEST_UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "pest_analytics_uploads"))
UPLOAD_MAX_AGE_DAYS = 7
CACHE_SIZE = int(os.environ.get("PEST_CACHE_SIZE", 8))  # parsed results kept in memory (up to 2 per run)

_cache = OrderedDict()   # (source key, with_jco) -> read_pest_zip result
_cache_lock = threading.Lock()
_load_locks = {}         # one lock per cache key, so concurrent callbacks parse a file only once


UPLOAD_EXTENSIONS = ("zip", "geojson", "json")


def save_upload(contents, ext="zip"):
    """Save an uploaded file (dcc.Upload base64 contents) on the server; return its key."""
    assert ext in UPLOAD_EXTENSIONS
    decoded = base64.b64decode(contents.split(',')[1])
    key = hashlib.sha256(decoded).hexdigest()[:32]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, f"{key}.{ext}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(decoded)
    _remove_old_uploads()
    return key


def _remove_old_uploads():
    cutoff = time.time() - UPLOAD_MAX_AGE_DAYS * 86400
    for name in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def upload_path(key, ext="zip"):
    """Path of a saved upload, or None. The key comes from the browser, so it is validated."""
    if isinstance(key, str) and re.fullmatch(r"[0-9a-f]{32}", key) and ext in UPLOAD_EXTENSIONS:
        path = os.path.join(UPLOAD_DIR, f"{key}.{ext}")
        if os.path.exists(path):
            return path
    return None


def _data_source(data):
    """Zip file to read for the loaded data: the server path if it exists, else the saved upload."""
    pestpath = data.get("pestpath")
    if pestpath and os.path.exists(pestpath):
        return pestpath
    if data.get("upload"):
        return upload_path(data["upload"])
    if data.get("contents"):
        return upload_path(save_upload(data["contents"]))
    return None


def _cached_read(path, with_jco):
    """
    Parse a zip once and keep the result in memory. A result with Jacobians also serves
    requests without them. Keys include the file's modification time and size, so an
    updated zip on the server is re-read.
    """
    st = os.stat(path)
    source = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    keys = [(source, True)] if with_jco else [(source, False), (source, True)]
    with _cache_lock:
        for k in keys:
            if k in _cache:
                _cache.move_to_end(k)
                return _cache[k]
        lock = _load_locks.setdefault(keys[0], threading.Lock())
    with lock:
        with _cache_lock:
            if keys[0] in _cache:
                return _cache[keys[0]]
        result = read_pest_zip(path, True, True, with_jco)
        with _cache_lock:
            _cache[keys[0]] = result
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
            _load_locks.pop(keys[0], None)
    return result


def clear_cache():
    with _cache_lock:
        _cache.clear()
    _read_sites_cached.cache_clear()


SITE_COLUMNS = {"latitude": "lat", "longitude": "lon", "long": "lon", "lng": "lon"}


def read_sites(data):
    """
    Site locations from site.csv in the loaded zip: a DataFrame with site (lower case, to match
    the grp* columns of obs.csv), lat, lon and any extra columns; None if there is no site.csv.
    Raises ValueError if site.csv is invalid.
    """
    if not check_data(data):
        return None
    path = _data_source(data)
    if not path:
        return None
    st = os.stat(path)
    sites = _read_sites_cached(os.path.abspath(path), st.st_mtime_ns, st.st_size)
    return None if sites is None else sites.copy()


@functools.lru_cache(maxsize=8)
def _read_sites_cached(path, mtime_ns, size):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if os.path.basename(n).lower() == "site.csv"]
        if not names:
            return None
        with z.open(names[0]) as f:
            sites = pd.read_csv(f)
    sites.columns = [SITE_COLUMNS.get(c.strip().lower(), c.strip().lower()) for c in sites.columns]
    missing = {"site", "lat", "lon"} - set(sites.columns)
    if missing:
        raise ValueError(f"site.csv needs columns site, lat, lon (missing: {', '.join(sorted(missing))})")
    sites["site"] = sites["site"].astype(str).str.strip().str.lower()
    for c in ["lat", "lon"]:
        sites[c] = pd.to_numeric(sites[c], errors="coerce")
    bad = sites[~sites.lat.between(-90, 90) | ~sites.lon.between(-180, 180)]
    if len(bad):
        raise ValueError(f"site.csv has {len(bad)} site(s) without a valid lat/lon, e.g. {bad.site.iloc[0]}"
                         " (coordinates must be latitude/longitude in decimal degrees)")
    return sites.drop_duplicates("site")


PESTPP_PATTERN = re.compile(r"(\.jcb$|\.\d+\.(par|rei|jco|obs)(\.csv)?$|\.(par|obs)\.csv$)", re.IGNORECASE)
ITERATION_PATTERN = re.compile(r"\.(rei|par|jco)\.\d+$", re.IGNORECASE)


def check_pest_files(data):
    """
    Warnings about the files in the loaded zip: PEST++ output (not read by this app) and missing
    per-iteration files. The app reads PEST / PEST_HP output: case.pst, case.rei.N, case.par.N,
    case.jco.N and case.hp.
    """
    path = _data_source(data) if check_data(data) else None
    if not path:
        return []
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    warnings = []
    pestpp = [n for n in names if PESTPP_PATTERN.search(n)]
    if pestpp:
        warnings.append(f"{len(pestpp)} file(s) look like PEST++ output (e.g. {os.path.basename(pestpp[0])}), "
                        "which this app does not read.")
    found = {m.group(1).lower() for n in names for m in [ITERATION_PATTERN.search(n)] if m}
    missing = [ext for ext in ("rei", "par", "jco") if ext not in found]
    if missing:
        warnings.append("No per-iteration " + ", ".join(f".{e}.N" for e in missing) + " files found; with PEST, set "
                        + ", ".join({"rei": "REISAVEITN", "par": "PARSAVEITN", "jco": "JCOSAVEITN"}[e] for e in missing)
                        + " in the control file to save them.")
    return warnings


def parse_zip(data, loadpar=True, loadrei=False, loadjco=False):
    """
    Return (pst, par, obs, pars, reis, jcos) for the loaded data, from the cache.
    The DataFrames are shared between callbacks: treat them as read-only.
    """
    if check_data(data):
        path = _data_source(data)
        if path:
            pst, par, obs, pars, reis, jcos = _cached_read(path, loadjco)
            if pst == "":
                return pst, par, obs, {}, {}, {}
            return (pst, par, obs,
                    dict(pars) if loadpar else {},
                    dict(reis) if loadrei else {},
                    dict(jcos) if loadjco else {})
    return None, None, None, None, None, None

def lower_group_columns(df):
    """lower-case the grp* columns of obs.csv/par.csv; numeric groups become strings"""
    for c in df.columns:
        if c.lower().startswith("grp"):
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.lower())


def merge_rei_obs(rei, obs):
    """
    Attach observation data to residuals. Prior-information (regularization) rows are not
    in the observation data, so their group and weight are taken from the .rei file and
    their grp* columns are set to the group name.
    """
    df = pd.merge(rei[["obsname", "measured", "modelled", "residual"]], obs, on="obsname",
                  how="left", suffixes=(None, "_1"))
    prior = ~rei["obsname"].isin(obs["obsname"]).values
    if prior.any() and {"obsgrp", "weight"} <= set(rei.columns):
        df.loc[prior, "obsgrp"] = rei["obsgrp"].values[prior]
        df.loc[prior, "weight"] = rei["weight"].values[prior]
        df.loc[prior, "obsval"] = rei["measured"].values[prior]
        for c in obs.columns:
            if c.lower().startswith("grp"):
                df.loc[prior, c] = rei["obsgrp"].values[prior]
    return df


def read_pest_zip(zip, loadpar=True, loadrei=False, loadjco=False):
    par = None; pa = None
    obs = None; ob = None
    pars = {}
    jcos = {}
    reis = {}
    hp = None
    pst = ""
    with zipfile.ZipFile(zip) as z:
        for name in z.namelist():
            if name.endswith('obs.csv'):
                with z.open(name) as f:
                    ob = pd.read_csv(f, )
                    ob["obsname"] = ob["obsname"].str.lower()
                    lower_group_columns(ob)
            if name.endswith('par.csv'):
                with z.open(name) as f:
                    pa = pd.read_csv(f, )
                    pa["parname"] = pa["parname"].str.lower()
                    lower_group_columns(pa)
            # e.g. case.jco.12 -> ("jco", 12); case.pst -> ("pst", 0)
            m = re.search(r"\.([A-Za-z]+)\.(\d+)$", name)
            if m:
                fileext, iiter = m.group(1).lower(), int(m.group(2))
            else:
                fileext, iiter = os.path.splitext(name)[1].lstrip(".").lower(), 0
            if fileext == "par" and loadpar:
                with z.open(name) as f:
                    df = read_par(f)
                pars[iiter] = df
            if fileext == "jco" and loadjco:
                with z.open(name) as f:
                    fbuf = f.read()
                jcos[iiter] = read_jco(fbuf)
            if fileext == "rei" and loadrei:
                with z.open(name) as f:
                    df = read_rei(f)
                reis[iiter] = df
            if fileext == "hp" and loadrei:
                with z.open(name) as f:
                    fbuf = f.read()
                hp = hpfile(fbuf)
            if fileext == "pst":
                pst = os.path.splitext(os.path.basename(name))[0]
                with z.open(name) as f:
                    par = read_par_in_pst(f)
                with z.open(name) as f:
                    obs = read_obs_in_pst(f)
    if pst == "":
        return ["", None, None, {}, {}, {}]
    if hp is not None and obs is not None:
        rr = obs.copy()
        rr["measured"] = obs["obsval"]
        rr["modelled"] = hp.REFOBS
        rr["residual"] = rr["measured"] - rr["modelled"]
        reis = {0:rr.loc[:, ["obsname", "obsgrp", "measured", "modelled", "residual", "weight"]]} | reis
    if par is not None and pa is not None:
        par = pd.merge(par, pa, on='parname', how="left", suffixes=(None, "_1"))
    if obs is not None and ob is not None:
        obs = pd.merge(obs, ob, on='obsname', how="left", suffixes=(None, "_1"))
    if reis:
        for i, df in reis.items():
            reis[i] = merge_rei_obs(df, obs)
    if pars:
        for i, df in pars.items():
            pars[i] = pd.merge(df, par.drop(["scale", "offset"], axis=1), on="parname", suffixes=(None, "_1"))
    return pst, par, obs, pars, reis, jcos
