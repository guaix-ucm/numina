"""The validation of images with a schema does not print"""

import astropy.io.fits as fits

from numina.frame.schema import validate_image


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
