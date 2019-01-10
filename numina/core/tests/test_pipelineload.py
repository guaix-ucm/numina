
import pytest

from ..pipelineload import load_confs



def test_load_confs_0():

    package = 'clodiadrp'
    node = {}

    # Error without values
    with pytest.raises(ValueError):
        load_confs(package, node)


def test_load_confs_1():
    """Test the loader returns a valid config when 'values' is empty"""
    
    package = 'clodiadrp'
    node = {
        'values': {}
    }

    result = load_confs(package, node)
    assert len(result) == 2

    result1, result2 = result

    assert len(result1) == 1
    assert 'default' in result1

    insconf = result1['default']
    assert insconf.instrument == 'EMPTY'
    assert insconf.name == 'EMPTY'