import pytest


def test_warns_define_input_and_result():
    from numina.core.recipeinout import RecipeInput, RecipeResult, define_input, define_result
    from numina.core.recipes import BaseRecipe

    class MyInput(RecipeInput):
        pass

    class MyResult(RecipeResult):
        pass

    with pytest.warns(DeprecationWarning, match="define_input is deprecated"):
        decorate_input = define_input(MyInput)
    with pytest.warns(DeprecationWarning, match="define_result is deprecated"):
        decorate_result = define_result(MyResult)

    @decorate_input
    @decorate_result
    class RecipeTest(BaseRecipe):
        pass

    assert RecipeTest.RecipeInput is MyInput
    assert RecipeTest.RecipeResult is MyResult


def test_lines_catalog_data_product_type():
    from numina.types import linescatalog, product

    assert linescatalog.DataProductType is product.DataProductType


def test_warns_matplotlib_qt(monkeypatch):
    """matplotlib_qt is a deprecated alias of matplotlib_utils"""
    import importlib
    import sys

    monkeypatch.delitem(sys.modules, "numina.array.display.matplotlib_qt", raising=False)
    with pytest.warns(DeprecationWarning, match="matplotlib_qt is deprecated"):
        importlib.import_module("numina.array.display.matplotlib_qt")

    import numina.array.display.matplotlib_utils as utils
    from numina.array.display.matplotlib_qt import new_figure, parse_geometry, plt, set_window_geometry

    assert new_figure is utils.new_figure
    assert parse_geometry is utils.parse_geometry
    assert set_window_geometry is utils.set_window_geometry
    assert plt is utils.plt
