import pytest

from ..pipelineload import load_confs, load_mode


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


@pytest.mark.parametrize("tagger", [None, ["KEY1"], "numina.core.taggers.get_tags_from_full_ob"])
def test_load_mode_ignores_tagger(tagger):
    """The per mode tagger of drp.yaml is ignored"""
    node = {"key": "bias", "name": "Bias", "summary": "", "description": "", "tagger": tagger}

    mode = load_mode(node)

    assert mode.key == "bias"
    assert not hasattr(mode, "tagger")
