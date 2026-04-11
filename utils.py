import base64
import io
import zipfile
import pandas as pd
import numpy as np
from collections.abc import Iterable

def isiterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def read_df(buf, header=None, **kwargs):
    return pd.read_csv(io.StringIO(buf), sep='\\s+', header=header, **kwargs)

def read_par(f):
    """
    read par file into a dataframe.
    """
    f.readline()
    lines = ''
    for l in f:
        if l.decode() == '\n':
            break
        lines += l.decode().strip() + '\n'
    return read_df(lines, names='parname parval scale offset'.split())

def read_rei(f):
    for l in f:
        l = l.decode().lower().strip()
        if l.startswith('name '):
            break
    lines = l + '\n'
    for l in f:
        if l == '\n':
            break
        lines += l.decode().strip() + '\n'
    return read_df(lines, header=0)

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


def read_par_in_pst(f):
    lines = ''
    tieds = {}
    for l in f:
        l = l.decode()
        if l.lower().lstrip()[:16] == '* parameter data':
            break
    for l in f:
        l = l.decode()
        if l.lstrip()[0] == '*':
            break
        if len(l.replace(',', ' ').replace('\t', ' ').split()) < 5: # skip tied parameters
            tt = l.replace(',', ' ').replace('\t', ' ').lstrip().lower().split()
            tieds[tt[0]] = tt[1]
            continue
        lines += l.strip() + '\n'
    par = read_df(lines, usecols=range(10),
                  names='parname partrans parchglim parval0 parlbnd parubnd pargp scale offset dercom'.split())
    par['parname']  = par['parname'].str.lower()
    par['partrans'] = par['partrans'].str.lower()
    par['pargp']    = par['pargp'].str.lower()
    par['tiedto'] = ""
    for i,r in par.iterrows():
            if r["partrans"] == "tied":
                par.loc[i, 'tiedto'] = tieds[r["parname"]]
    return par

def read_obs_in_pst(f):
        lines = ''
        for l in f:
            if l.decode().lower().lstrip()[:18] == '* observation data':
                break
        for l in f:
            l = l.decode()
            if l.lstrip()[0] == '*':
                break
            lines += l.strip() + '\n'

        obs = read_df(lines, usecols=range(4), names='obsname obsval weight group'.split())
        obs['obsname'] = obs['obsname'].str.lower()
        obs['group']   = obs['group'].str.lower()
        return obs

def parse_zip(contents, loadpar=True, loadrei=False, loadjco=False):
    if not contents:
        return None, None, None, None, None
    content_string = contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    return read_pest_zip(io.BytesIO(decoded), loadpar, loadrei, loadjco)

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
                    ob["obsgp"] = ob["obsgp"].str.lower()
                    ob.set_index("obsname")
            if name.endswith('par.csv'):
                with z.open(name) as f:
                    pa = pd.read_csv(f, )
                    pa["parname"] = pa["parname"].str.lower()
                    pa["pargp"] = pa["pargp"].str.lower()
                    pa = pa.set_index("parname")
            if name[-1] in "0123456789":
                iiter = int(name[-1])
                fileext = name[-5:-2]
            else:
                iiter = 0
                fileext = name[-3:]
            # print(name, iiter, fileext)
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
            if fileext == ".hp" and loadrei:
                with z.open(name) as f:
                    fbuf = f.read()
                hp = hpfile(fbuf)
            if fileext == "pst":
                pst = name.rstrip(".pst")
                with z.open(name) as f:
                    par = read_par_in_pst(f)
                with z.open(name) as f:
                    obs = read_obs_in_pst(f)
    if hp is not None and obs is not None:
        rr = obs.copy()
        rr["name"] = obs["obsname"]
        rr["group"] = obs["group"]
        rr["measured"] = obs["obsval"]
        rr["modelled"] = hp.REFOBS
        rr["residual"] = rr["measured"] - rr["modelled"]
        reis[0] = rr.loc[:, ["name", "group", "measured", "modelled", "residual", "weight"]]
    if par is not None and pa is not None:
        par["pargp"] = pa.loc[par["parname"], "pargp"].values
    if obs is not None and ob is not None:
        obs["group"] = ob.loc[obs["obsname"], "obsgp"].values
        for i, df in reis.items():
            df["group"] = ob.loc[df["name"], "obsgp"].values
    return pst, par, obs, pars, reis, jcos
#%%
if __name__ == "__main__":
    pst, par, obs, pars, reis, jcos = read_pest_zip(
        r"c:\Cloud\OneDrive - S.S. Papadopulos & Associates, Inc\1869-SWRCB_Napa\05_Model\NVIHM_SWRCB_NWT_LSPC\PEST\zzArchive\napa2026_03.zip",
        True, True, True
        )
    
   
    def test_parsen():
        stype = "L1 Norm"
        otype = "Observation"
        parselect = ["kdcr01"]
        
        ddf = {} # aggregated sensitivity
        sdf = {} # (raw)signed sensitivity
        nobs = 20
        for i, df0 in jcos.items():
            df = pd.merge(df0, obs, left_index=True, right_on="obsname")
            df.set_index("obsname", inplace=True)
            sdf[f"Iter {i}"] = (df["weight"].values[:, None] * df[parselect]).sum(axis=1)
            if stype == "L1 Norm":
                ddf[f"Iter {i}"] =  (df["weight"].values[:, None] * df[parselect] .abs()).sum(axis=1)
            else:
                ddf[f"Iter {i}"] = ((df["weight"].values[:, None] * df[parselect])**2).sum(axis=1)
        sdf = pd.DataFrame(sdf, )
        ddf = pd.DataFrame(ddf, )
        if otype == "Obs Groups":
            ddf = ddf.groupby(obs.set_index("obsname").loc[ddf.index, "obsgp"]).sum()
        
        if stype == "L2 Norm":
            ddf = ddf**0.5
            
        tot = (ddf / ddf.median(axis=0).values[None,:]).sum(axis=1)
        tot = tot.sort_values(ascending=False).iloc[:int(nobs)]
        ddf = ddf.loc[tot.index]
        sdf = sdf.loc[tot.index]
        sdf = pd.DataFrame(np.where(sdf>0, "triangle-up", "triangle-down"), columns=ddf.columns)
    
        pp = {"Iter 0": par.set_index("parname").loc[parselect, "parval0"]}
        for i, df in pars.items():
            pp[f"Iter {i}"] = df.set_index("parname").loc[parselect, "parval"]
    
        ddf["Iter 0"] = 0
        sdf["Iter 0"] = "circle"
        
        reis[i].set_index("name").loc[ddf.index, "modelled"].to_string(index=False, header=False).split("\n")
    
    def test_parcor():
        
        npar = 50
        iiter = 1
        highr = 0.9
        
        csplimit = 1e-10
        nobs_useful = (obs.weight > 0).sum()
    
        jco0 = jcos[int(iiter)]
        jco = jco0.loc[jco0.index.isin(obs.obsname), ]
        WJ = jco * obs.set_index("obsname").loc[jco.index, "weight"].values[:, None]
        csp = (WJ**2).sum() ** 0.5 / nobs_useful
        mask = WJ.columns.isin(csp[csp>csplimit].index)
        JJ = WJ.loc[:,mask]
        J2 = np.matmul(JJ.T, JJ)
        try:
            cov = np.linalg.inv(J2)
        except:
            cov = np.linalg.pinv(J2)
        std = np.sqrt(np.diag(cov))
        cor = cov / np.outer(std, std)
        cor = pd.DataFrame(cor, index=J2.columns, columns=J2.columns)
    
        rmax = (cor - np.identity(len(cor))).abs().max().sort_values()
        mask = cor.index.isin(rmax.index[-int(npar):])
        cor = cor.loc[mask, mask]
        
        cor_off = cor - np.identity(len(cor))
        highcor = cor_off[cor_off.abs()>float(highr)].stack().dropna().reset_index()
        
    def test_par_identifiable():
        svdtrunc = 1e-4
        csplimit = 1e-10
        nobs_useful = (obs.weight > 0).sum()
        jco0 = jcos[1].copy()
        
        jco = jco0.loc[jco0.index.isin(obs.obsname), ]
        WJ = jco * obs.set_index("obsname").loc[jco.index, "weight"].values[:, None]
        csp = (WJ**2).sum() ** 0.5 / nobs_useful
        mask = WJ.columns.isin(csp[csp>csplimit].index)
        JJ = WJ.loc[:,mask]
        
        
        U, s, Vt = np.linalg.svd(JJ, full_matrices=False)
        sr = s / s[0]
        V1 = Vt[sr>=svdtrunc]
    
        V = pd.DataFrame(V, index=JJ.columns, columns=['eig'+str(i+1) for i in range(ndimpar)])
        if parnames is not None:
            V = V.loc[V.index.isin(parnames)]
            
            
            
        U, s, Vt = np.linalg.svd(JJ, full_matrices=False)
        def check1():
            # check s: Rapid decay → ill-conditioning; Very small values → problematic directions
            condition_number = s[0] / s[-1]
            
            print("Singular values:", s)
            print("Condition number:", s[0] / s[-1])
        
        def check2():
            V = Vt.T
            #For small 𝜎, look at 𝑣𝑘:
            k = np.argmin(s)   # worst direction
            print(V[:, k]) # Which combination of parameters is poorly identifiable
            
            vk = V[:, k]            
            plt.bar(range(len(vk)), vk)
            plt.xlabel("Parameter index")
            plt.ylabel("Weight")
            plt.title(f"Poorly identifiable direction (σ={s[k]:.2e})")

            
        # Find poorly identifiable combinations
        threshold = 1e-6 * s[0]
        
        for k in range(len(s)):
            if s[k] < threshold:
                print(f"\nPoorly identifiable direction {k}:")
                print(V[:, k])

        plt.imshow(V, aspect='auto')


        plt.semilogy(s, 'o-')
        plt.xlabel("Index")
        plt.ylabel("Singular value")
        plt.title("Singular value spectrum")
        
        
        #%% 
        v1 = V[:, -1]
        v2 = V[:, -2]
        
        # Project identity basis (parameters) into this subspace
        coords = np.vstack([v1, v2]).T
        
        plt.scatter(coords[:,0], coords[:,1])
        
        for i in range(len(coords)):
            plt.text(coords[i,0], coords[i,1], f"p{i}")
        
        plt.xlabel("Worst direction")
        plt.ylabel("2nd worst direction")
        plt.title("Parameter relationships in poorly identifiable subspace")