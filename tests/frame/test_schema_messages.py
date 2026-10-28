"""The validation of images with a schema does not print"""

import importlib
import warnings

import astropy.io.fits as fits
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import numina.frame.schema
    from numina.frame.schema import validate_image


def test_deprecated():
    with pytest.warns(DeprecationWarning, match="numina.frame.schema is deprecated"):
        importlib.reload(numina.frame.schema)


def test_validate_image_does_not_print(capsys):
    hdr = fits.Header({"OBSERVER": "someone"})
    image = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    schema = {
        "extensions": {
            "primary": {"keywords": {"OBSERVER": {"required": True}}},
            "variance": {"required": False},
        }
    }
    validate_image(image, schema)
    assert capsys.readouterr().out == ""
