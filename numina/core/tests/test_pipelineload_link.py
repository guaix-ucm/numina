
import pytest

import numina.core.pipelineload as loader


@pytest.fixture(scope='module')
def drptest():
    return loader.drp_load('numina.drps.tests', 'drptest4.yaml')


def test_load_link(drptest):
    """Test the loader returns a valid config when 'values' is empty"""

    recipe = drptest.get_recipe_object("image_c")
    assert 'obresult' in recipe.query_options
    assert 'accum_in' in recipe.query_options
