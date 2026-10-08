"""Checks on how the page modules are put together (component ids)."""
import importlib
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
