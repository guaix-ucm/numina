import pytest


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


def test_warns_numina_tests(monkeypatch):
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "numina.tests" or name.startswith("numina.tests."):
            monkeypatch.delitem(sys.modules, name)

    # numina.tests also imports the deprecated modules of numina.testing
    with pytest.warns(DeprecationWarning) as record:
        importlib.import_module("numina.tests")
    assert any("numina.tests is deprecated" in str(w.message) for w in record)

    import numina.testing.plugins
    import numina.testing.testcache
    from numina.tests import testcache
    from numina.tests.plugins import pytest_configure

    assert testcache is numina.testing.testcache
    assert pytest_configure is numina.testing.plugins.pytest_configure


def test_warns_configuration_tag():
    from numina.types.product import ConfigurationTag

    with pytest.warns(DeprecationWarning, match="ConfigurationTag"):

        class MyConf(ConfigurationTag):
            pass

    with pytest.warns(DeprecationWarning, match="ConfigurationTag"):
        assert ConfigurationTag.isconfiguration()


def test_warns_generators():
    from numina.types.frame import DataFrameType
    from numina.types.product import DataProductMixin

    class MyProduct(DataProductMixin, DataFrameType):
        pass

    with pytest.warns(DeprecationWarning, match="generators"):
        assert MyProduct().generators() == []


def test_lines_catalog_data_product_type():
    from numina.types import linescatalog, product

    assert linescatalog.DataProductType is product.DataProductType


@pytest.mark.parametrize("name", ["tags_as_scalar", "tags_as_list"])
def test_warns_tags_as(name):
    import numina.core.dataholders as dh
    from numina.core.oresult import ObservationResult

    obsres = ObservationResult()
    obsres.tags = {"filter": "J"}
    with pytest.warns(DeprecationWarning, match=name):
        with getattr(dh, name)(obsres):
            pass
    assert obsres.tags == {"filter": "J"}


def test_warns_gather_info_dframe():
    import astropy.io.fits as fits

    from numina.core import DataFrame
    from numina.datamodel import DataModel

    hdu = fits.PrimaryHDU()
    hdu.header["UUID"] = "u1"
    hdulist = fits.HDUList([hdu])
    datamodel = DataModel("CLODIA")
    with pytest.warns(DeprecationWarning, match="gather_info_dframe is deprecated"):
        info = datamodel.gather_info_dframe(DataFrame(frame=hdulist))
    assert info["uuid"] == "u1"
