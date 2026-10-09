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
