from . import (
    uploadpest,
    phiupdate,
    tsplot,
    scatterplot,
    parupdate,
    parsen,
    parcor,
    obssen,
    obssim,
    obsconf,
    resinfl,
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
        "resinfl":resinfl,
    },
    "Parameters": {
        "parupdate":parupdate,
        "parsen":parsen,
        "parcor":parcor,
    },
    "Observations": {
        "obssen":obssen,
        "obssim":obssim,
        "obsconf":obsconf,
    },
}

apps = {k: v for pages in sections.values() for k, v in pages.items()}
