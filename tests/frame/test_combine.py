"""Combination of images of numina.frame.combine"""

import warnings

import astropy.io.fits as fits
import numpy
import pytest

import numina.array._combine as intl
import numina.frame.combine as fc
from numina.datamodel import DataModel
from numina.keydef import KeyDefinition


def image(value, uuid, bad=None, **keywords):
    hdu = fits.PrimaryHDU(numpy.full((3, 4), value, dtype="float32"))
    hdu.header["UUID"] = uuid
    for key, val in keywords.items():
        hdu.header[key] = val
    mask = numpy.zeros((3, 4), dtype="uint8")
    if bad is not None:
        mask[bad] = 1
    return fits.HDUList([hdu, fits.ImageHDU(mask, name="BPM")])


def images():
    return [image(1.0, "a", EXPTIME=1.0), image(2.0, "b", EXPTIME=2.0), image(6.0, "c", EXPTIME=6.0)]


def test_combine():
    result = fc.combine(intl.mean_method(), images())

    assert [hdu.name for hdu in result] == ["PRIMARY", "VARIANCE", "MAP"]
    assert numpy.allclose(result[0].data, 3.0)
    assert numpy.allclose(result["VARIANCE"].data, 7.0)
    assert numpy.all(result["MAP"].data == 3)
    assert result["MAP"].data.dtype == numpy.uint8
    assert result[0].header["NUM-NCOM"] == 3
    assert result[0].header["UUID"] not in {"a", "b", "c"}
    history = list(result[0].header["HISTORY"])
    assert history[0] == "Combined 3 images"
    assert history[2:] == ["Image1 a", "Image2 b", "Image3 c"]


def test_combine_options():
    imgs = images()
    imgs[0][0].header["NUM-NCOM"] = 2

    result = fc.combine(intl.median_method(), imgs, dtype="float32", include_variance=False)

    assert [hdu.name for hdu in result] == ["PRIMARY", "MAP"]
    assert result[0].data.dtype == numpy.float32
    assert numpy.allclose(result[0].data, 2.0)
    assert result[0].header["NUM-NCOM"] == 6


class OtherDataModel(DataModel):
    def get_imgid(self, img):
        return "id-" + img[0].header["UUID"]


def test_combine_datamodel():
    result = fc.combine(intl.mean_method(), images(), datamodel=OtherDataModel("TEST"))
    assert list(result[0].header["HISTORY"])[2] == "Image1 id-a"


def test_combine_method_name():
    result = fc.combine(intl.mean_method(), images(), method_name="mean")
    assert result[0].header["HISTORY"][0] == "Combined 3 images using 'mean'"


def test_combine_masks_extension():
    imgs = images()
    imgs[2]["BPM"].data[0, 0] = 1

    result = fc.combine(intl.mean_method(), imgs, masks=fc.Extension("BPM"))

    assert result[0].data[0, 0] == pytest.approx(1.5)
    assert result["MAP"].data[0, 0] == 2
    assert result[0].data[1, 1] == pytest.approx(3.0)


def test_combine_masks_sequence():
    masks = [fits.HDUList([fits.PrimaryHDU(img["BPM"].data)]) for img in images()]
    masks[0][0].data[1, 2] = 1

    result = fc.combine(intl.mean_method(), images(), masks=masks)

    assert result[0].data[1, 2] == pytest.approx(4.0)
    assert result["MAP"].data[1, 2] == 2


def test_combine_masks_errors():
    with pytest.raises(TypeError, match="mask in invalid"):
        fc.combine(intl.mean_method(), images(), masks=3)

    masks = [fits.HDUList([fits.PrimaryHDU(numpy.zeros((3, 4)))])]
    with pytest.raises(TypeError, match="len"):
        fc.combine(intl.mean_method(), images(), masks=masks)


@pytest.mark.parametrize(
    "scales",
    [
        "mean",
        "median",
        "mode",
        KeyDefinition("EXPTIME"),
        lambda img, mask, region: img[0].data[region].mean(),
    ],
)
def test_combine_scales(scales):
    result = fc.combine(intl.mean_method(), images(), scales=scales)
    assert numpy.allclose(result[0].data, 1.0)


def test_combine_keyword_default():
    result = fc.combine(intl.mean_method(), images(), zeros=KeyDefinition("OFFSET", default=0.0))
    assert numpy.allclose(result[0].data, 3.0)


def test_combine_keyword_missing():
    with pytest.raises(KeyError, match="'EXPOSED' not found in HDU 0"):
        fc.combine(intl.mean_method(), images(), scales=KeyDefinition("EXPOSED"))


def test_combine_schema_keyword_deprecated():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from numina.frame.schema import SchemaKeyword

    with pytest.warns(DeprecationWarning, match="SchemaKeyword"):
        result = fc.combine(intl.mean_method(), images(), scales=SchemaKeyword("EXPTIME"))
    assert numpy.allclose(result[0].data, 1.0)


def test_keyword_deprecated():
    with pytest.warns(DeprecationWarning, match="Keyword is deprecated"):
        keyword = fc.Keyword("EXPTIME")
    assert keyword.name == "EXPTIME"
    with pytest.raises(AttributeError):
        fc.Other


@pytest.mark.parametrize("scales", ["mean", "median", "mode"])
def test_combine_scales_masks(scales):
    result = fc.combine(intl.mean_method(), images(), masks=fc.Extension("BPM"), scales=scales)
    assert numpy.allclose(result[0].data, 1.0)


def test_combine_zeros_sequence():
    result = fc.combine(intl.mean_method(), images(), zeros=[1.0, 2.0, 6.0])
    assert numpy.allclose(result[0].data, 0.0)


def test_combine_region():
    imgs = images()
    for img in imgs:
        img[0].data[0] = 100
        img["BPM"].data[1, 0] = 1

    region = (slice(1, 3), slice(None))
    result = fc.combine(intl.mean_method(), imgs, masks=fc.Extension("BPM"), zeros="median", region=region)

    assert numpy.allclose(result[0].data[1:], 0.0)


def test_combine_weights():
    result = fc.combine(intl.mean_method(), images(), weights=[4.0, 1.0, 1.0])
    assert numpy.allclose(result[0].data, 2.0)


def test_combine_method_errors():
    with pytest.raises(KeyError, match="invalid method"):
        fc.combine(intl.mean_method(), images(), scales="max")

    with pytest.raises(TypeError, match="no callable"):
        fc.combine(intl.mean_method(), images(), zeros=3.0)
