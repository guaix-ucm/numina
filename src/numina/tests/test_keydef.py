import pytest
from astropy.io import fits

from ..keydef import KeyDefinition


def test_get_val1(pattri_header):
    key_def = KeyDefinition(key="INSMODE")
    val = key_def.get_value(pattri_header)
    assert val == "Mode_B"


@pytest.mark.parametrize("convert, value", [(str.upper, "MODE_B"), (None, "Mode_B")])
def test_get_val2(pattri_header, convert, value):
    hdu = fits.PrimaryHDU()
    hdu.header = pattri_header
    hdul = fits.HDUList([hdu])
    key_def = KeyDefinition(key="INSMODE", convert=convert)
    val = key_def.get_value(hdul)
    assert val == value


@pytest.mark.parametrize(
    "value, convert, res",
    [
        ("MODE_C", str.lower, "mode_c"),
        (None, str.lower, "mode_0"),
        ("MODE_C", None, "MODE_C"),
        (None, None, "MODE_0"),
    ],
)
def test_set_val_convert(pattri_header, value, convert, res):
    hdu = fits.PrimaryHDU()
    hdu.header = pattri_header
    hdul = fits.HDUList([hdu])
    key_def = KeyDefinition(key="INSMODE", default="MODE_0", convert=convert)
    key_def.set_value(hdul, value)

    val = key_def.get_value(hdul)
    assert val == res


def test_get_header():
    key_def = KeyDefinition(key="INSMODE", default="MODE_0")
    with pytest.raises(ValueError):
        key_def._get_header({})  # noqa
