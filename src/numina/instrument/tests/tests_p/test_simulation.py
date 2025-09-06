import pytest

from numina.keydef import extract, KeyDefinition


@pytest.mark.parametrize(
    "path, result",
    [(["PATTRI.wheel", "label"], "U"), (["PATTRI.wheel", "broken"], "B")],
)
def test_extract(pattri_header, pattri_state, path, result):
    key_def = KeyDefinition("FILTER", default="B")
    extract(pattri_header, pattri_state, path, key_def)
    assert pattri_header["FILTER"] == result
