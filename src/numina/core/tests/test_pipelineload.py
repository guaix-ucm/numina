import pytest

from ..pipelineload import load_confs


def test_load_confs_path():
    """The configurations are loaded from 'path'"""

    confs, modpath = load_confs("numina", {"path": "numina.drps.tests.configs"})

    assert modpath == "numina.drps.tests.configs"
    assert "instrument-test1.json" in confs
    assert confs["instrument-test1.json"]["name"] == "TEST1"


def test_load_confs_default_path():
    """Without 'path', the configurations are in package.instrument.configs"""

    with pytest.raises(ModuleNotFoundError) as excinfo:
        load_confs("numina", {})

    assert excinfo.value.name == "numina.instrument.configs"
