from . import (
    uploadpest,
    phiupdate,
    tsplot,
    scatterplot,
    parupdate,
    parsen,
    parcor,
    parident,
    parunc,
    obssen,
    obssim,
    obsconf,
    resinfl,
    resmap,
)

# navigation sections, in display order
sections = {
    "Data": {
        "uploadpest":uploadpest,
    },
    "Fit": {
        "phiupdate":phiupdate,
        "scatterplot":scatterplot,
        "tsplot":tsplot,
        "resmap":resmap,
        "resinfl":resinfl,
    },
    "Parameters": {
        "parupdate":parupdate,
        "parsen":parsen,
        "parcor":parcor,
        "parident":parident,
        "parunc":parunc,
    },
    "Observations": {
        "obssen":obssen,
        "obssim":obssim,
        "obsconf":obsconf,
    },
}

apps = {k: v for pages in sections.values() for k, v in pages.items()}
