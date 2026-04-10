import dash
from dash import html, dcc
import dash_mantine_components as dmc

name = __name__.lstrip("module.") + "-"

md = dcc.Markdown('''
    ## Introducing LaTeX using MathJax:

    This example uses the block delimiter:
    $$
    \\frac{1}{(\\sqrt{\\phi \\sqrt{5}}-\\phi) e^{\\frac25 \\pi}} =
    1+\\frac{e^{-2\\pi}} {1+\\frac{e^{-4\\pi}} {1+\\frac{e^{-6\\pi}}
    {1+\\frac{e^{-8\\pi}} {1+\\ldots} } } }
    $$

    This example uses the inline delimiter:
    $E^2=m^2c^4+p^2c^2$

    ### LaTeX in a Graph component:

    ''',
    mathjax=True)

layout = html.Div([
    md
])
