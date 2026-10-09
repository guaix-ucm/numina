import pytest


def test_warns_qc():

    with pytest.warns(DeprecationWarning):
        import numina.core.qc  # noqa: F401


def test_warns_products():

    with pytest.warns(DeprecationWarning):
        import numina.core.products  # noqa: F401


def test_warns_query_constraints():
    import numina.types.datatype as dt
    from numina.core.dataholders import Requirement
    from numina.core.query import Constraint
    from numina.types.frame import DataFrameType
    from numina.types.linescatalog import LinesCatalog

    req = Requirement(dt.PlainPythonType(ref=1), description="param", destination="param")
    with pytest.warns(DeprecationWarning, match="Requirement.query_constraints"):
        assert isinstance(req.query_constraints(), Constraint)

    with pytest.warns(DeprecationWarning, match="DataTypeBase.query_constraints"):
        assert isinstance(dt.PlainPythonType(ref=1).query_constraints(), Constraint)

    with pytest.warns(DeprecationWarning, match="DataTypeBase.query_constraints"):
        assert isinstance(DataFrameType().query_constraints(), Constraint)

    with pytest.warns(DeprecationWarning, match="DataProductMixin.query_constraints"):
        assert isinstance(LinesCatalog().query_constraints(), Constraint)

    with pytest.warns(DeprecationWarning, match="Constraint is deprecated"):
        Constraint()


def test_warns_multitype_query_on_dal():
    from numina.types.frame import DataFrameType
    from numina.types.multitype import MultiType

    with pytest.warns(DeprecationWarning, match="MultiType._query_on_dal"):
        # the subtypes have no query method, it fails after warning
        with pytest.raises(AttributeError):
            MultiType(DataFrameType)._query_on_dal("name", None, None)


def test_warns_define_requirements():
    from numina.core.recipeinout import RecipeInput, define_requirements
    from numina.core.recipes import BaseRecipe

    class MyInput(RecipeInput):
        pass

    with pytest.warns(DeprecationWarning, match="define_requirements"):

        @define_requirements(MyInput)
        class RecipeTest(BaseRecipe):
            pass

    assert RecipeTest.RecipeInput is MyInput


def test_warns_numina_tests(monkeypatch):
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "numina.tests" or name.startswith("numina.tests."):
            monkeypatch.delitem(sys.modules, name)

    with pytest.warns(DeprecationWarning, match="numina.tests is deprecated"):
        importlib.import_module("numina.tests")

    import numina.testing.plugins
    import numina.testing.testcache
    from numina.tests import testcache
    from numina.tests.plugins import pytest_configure

    assert testcache is numina.testing.testcache
    assert pytest_configure is numina.testing.plugins.pytest_configure
