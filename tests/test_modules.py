"""Checks on how the page modules are put together (component ids, callback names)."""
import ast
import collections
import importlib
import pathlib
import pkgutil

import pytest

import module

PAGES = sorted(m.name for m in pkgutil.iter_modules(module.__path__))


@pytest.mark.parametrize("page", PAGES)
def test_component_prefix_is_page_name(page):
    mod = importlib.import_module(f"module.{page}")
    if hasattr(mod, "name"):
        assert mod.name == page + "-"


def test_registered_pages_have_distinct_prefixes():
    names = [mod.name for mod in module.apps.values()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("page", PAGES)
def test_no_function_defined_twice(page):
    # a second callback with the same name hides the first from tests and tracebacks
    source = pathlib.Path(module.__path__[0], page + ".py").read_text(encoding="utf-8")
    names = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    assert [n for n, c in collections.Counter(names).items() if c > 1] == []


def test_index_page_loads_nothing_from_other_sites():
    # no analytics, ad or CDN scripts: the app must work on an offline network (review 4.1)
    import re
    import app
    html = app.app.server.test_client().get("/").get_data(as_text=True)
    sources = re.findall(r'<(?:script|link)[^>]+(?:src|href)="([^"]+)"', html)
    assert sources and not [s for s in sources if s.startswith(("http:", "https:", "//"))]
    assert "googletagmanager" not in html and "ethicalads" not in html


def test_css_section_on_parameter_sensitivity(pest_data):
    from unittest.mock import patch
    with patch("module.parsen.parse_zip", return_value=pest_data):
        from module.parsen import plot_css
        fig = plot_css(1, {"pestpath": "x"}, "6", "obsgrp", "weight", "All")
    assert "68 of 132 parameters without sensitivity" in fig.layout.title.text
    bars = [t for t in fig.data if t.type == "bar"]
    assert {t.name for t in bars} == {"head", "flow"} and len(bars[0].y) == 132


def test_normal_scores_and_qq_line():
    import numpy as np
    from plotting import normal_scores, qq_line
    rng = np.random.default_rng(0)
    values = 3 + 2 * rng.standard_normal(4000)
    z, ordered, order = normal_scores(values)
    assert (np.diff(ordered) >= 0).all() and np.array_equal(values[order], ordered)
    assert abs(z.mean()) < 1e-9 and z[0] < -3 < 3 < z[-1]
    a, b = qq_line(z, ordered)
    assert abs(a - 3) < 0.1 and abs(b - 2) < 0.1


def test_residual_distribution_section(pest_data):
    from unittest.mock import patch
    with patch("module.resinfl.parse_zip", return_value=pest_data):
        from module.resinfl import plot_distribution
        fig = plot_distribution(1, {"pestpath": "x"}, "6", "obsgrp", "weight")
    weighted = pest_data[2].weight.gt(0).sum()
    assert f"n = {weighted}" in fig.layout.title.text
    assert sum(len(t.x) for t in fig.data if t.type == "scattergl") == weighted
