
import pytest

import numina.tests.simpleobj as simple


@pytest.mark.result_compare
def test_result_compare_demo1():
    from numina.types.frame import DataFrameType
    from numina.types.structured import BaseStructuredCalibration

    import numina.core

    simple_img = simple.create_simple_hdul()
    simple_stc = simple.create_simple_structured()

    class BB(numina.core.RecipeResult):
        prod1 = numina.core.Result(int, 'something1')
        prod2 = numina.core.Result(DataFrameType, 'something2')
        prod3 = numina.core.Result(BaseStructuredCalibration, 'something3')

    result = BB(prod1=1, prod2=simple_img, prod3=simple_stc)
    return result


@pytest.mark.result_compare
def test_result_compare_demo2():
    from numina.types.frame import DataFrameType
    from numina.types.structured import BaseStructuredCalibration

    import numina.core

    simple_img = simple.create_simple_hdul()
    simple_stc = simple.create_simple_structured()

    class BB(numina.core.RecipeResult):
        prod1 = numina.core.Result(int, description='something1',
                                   destination="prod4")
        prod2 = numina.core.Result(DataFrameType, 'something2')
        prod3 = numina.core.Result(BaseStructuredCalibration, 'something3')

    result = BB(prod1=1, prod2=simple_img, prod3=simple_stc)
    return result
