"""Combination of images"""

import types
import uuid

import astropy.io.fits as fits
import numpy
import pytest

from numina.core import DataFrame
from numina.processing.combine import (
    basic_processing_with_combination,
    basic_processing_with_combination_frames,
    combine_imgs,
)


def create_img(value, tsutc2, extension=False):
    hdu = fits.PrimaryHDU(numpy.full((4, 4), value, dtype="float32"))
    hdu.header["TSUTC2"] = tsutc2
    hdu.header["UUID"] = str(uuid.uuid4())
    hdul = fits.HDUList([hdu])
    if extension:
        ext = fits.ImageHDU(numpy.zeros((2, 2)), name="EXTRA")
        ext.header["TSUTC2"] = tsutc2
        hdul.append(ext)
    return hdul


@pytest.mark.parametrize("extension", [False, True])
def test_combine_tsutc2_of_last_image(extension):
    imgs = [create_img(value, tsutc2, extension) for value, tsutc2 in [(1.0, 10.0), (3.0, 20.0)]]

    result = combine_imgs(imgs)

    assert result[0].header["TSUTC2"] == 20.0
    numpy.testing.assert_allclose(result[0].data, 2.0)
    assert result[0].header["NUM-NCOM"] == 2
    # the input images are not modified
    assert [img[0].header["TSUTC2"] for img in imgs] == [10.0, 20.0]
    if extension:
        assert [img["EXTRA"].header["TSUTC2"] for img in imgs] == [10.0, 20.0]
        assert result["EXTRA"].header["TSUTC2"] == 10.0


def test_combine_extensions():
    imgs = [create_img(value, 0.0, extension=True) for value in [1.0, 3.0]]

    result = combine_imgs(imgs, errors=True)

    assert [hdu.name for hdu in result] == ["PRIMARY", "EXTRA", "VARIANCE", "MAP"]
    assert result["MAP"].data.dtype == numpy.int16


def test_combine_no_images():
    with pytest.raises(ValueError):
        combine_imgs([])


class RecordFlow:
    """A flow that adds a value to the image and records the images it receives"""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, hdul):
        self.calls.append(hdul[0].data.copy())
        hdul[0].data = hdul[0].data + self.value
        return hdul


def frames_of(values):
    return [DataFrame(frame=create_img(value, 10.0)) for value in values]


@pytest.mark.parametrize("as_list", [False, True])
def test_one_flow_after_combination(as_list):
    flow = RecordFlow(1.0)
    flows = [flow] if as_list else flow
    result = basic_processing_with_combination_frames(frames_of([1.0, 3.0]), flows)
    # combined, then processed once
    assert len(flow.calls) == 1
    assert numpy.allclose(result[0].data, 3.0)


def test_no_flows():
    result = basic_processing_with_combination_frames(frames_of([1.0, 3.0]), [])
    assert numpy.allclose(result[0].data, 2.0)


def test_two_flows():
    """The first flow is applied to each image, the second to the combination"""
    first, second = RecordFlow(10.0), RecordFlow(1.0)
    result = basic_processing_with_combination_frames(frames_of([1.0, 3.0]), [first, second])
    assert [c[0, 0] for c in first.calls] == [1.0, 3.0]
    assert [c[0, 0] for c in second.calls] == [12.0]
    assert numpy.allclose(result[0].data, 13.0)
    assert result[0].header["NUM-NCOM"] == 2


def mediancr(arrays, crmasks, dtype):
    """A combination with masks of cosmic rays, it records the masks"""
    mediancr.masks = crmasks
    data = numpy.median(arrays, axis=0).astype(dtype)
    return data, numpy.zeros_like(data), numpy.full(data.shape, len(arrays))


def crmasks_frame():
    hdu = fits.PrimaryHDU(numpy.zeros((4, 4), dtype="uint8"))
    hdu.header["UUID"] = "0d1b2c3e-0000-0000-0000-000000000000"
    return DataFrame(frame=fits.HDUList([hdu]))


def test_flows_with_crmasks():
    """All the flows are applied to each image before the combination with the masks"""
    first, second = RecordFlow(10.0), RecordFlow(1.0)
    crmasks = crmasks_frame()
    result = basic_processing_with_combination_frames(
        frames_of([1.0, 3.0]), [first, second], method=mediancr, crmasks=crmasks
    )
    assert len(first.calls) == 2
    assert [c[0, 0] for c in second.calls] == [11.0, 13.0]
    assert numpy.allclose(result[0].data, 13.0)
    assert mediancr.masks[0].header["UUID"] == crmasks.frame[0].header["UUID"]
    assert f"Masks uuid:{crmasks.frame[0].header['UUID']}" in result[0].header["HISTORY"]


def test_crmasks_required():
    with pytest.raises(ValueError, match="Cosmic ray masks are required for mediancr"):
        basic_processing_with_combination_frames(frames_of([1.0, 3.0]), [], method=mediancr)


def test_crmasks_of_recipe_input():
    """The masks of cosmic rays are those of the recipe input"""
    rinput = types.SimpleNamespace(
        obresult=types.SimpleNamespace(frames=frames_of([1.0, 3.0])), crmasks=crmasks_frame()
    )
    result = basic_processing_with_combination(rinput, [], method=mediancr)
    assert numpy.allclose(result[0].data, 2.0)
    assert mediancr.masks[0].header["UUID"] == rinput.crmasks.frame[0].header["UUID"]


@pytest.mark.parametrize("method, value", [("mean", 3.0), ("median", 2.0)])
@pytest.mark.parametrize("errors", [False, True])
def test_main(tmp_path, method, value, errors):
    from numina.processing.combine import main

    names = []
    for idx, val in enumerate([1.0, 2.0, 6.0]):
        name = tmp_path / f"img{idx}.fits"
        create_img(val, 0.0).writeto(name)
        names.append(str(name))
    output = tmp_path / "out.fits"
    args = ["-o", str(output), "--method", method] + (["-e"] if errors else []) + names

    main(args)

    with fits.open(output) as hdul:
        numpy.testing.assert_allclose(hdul[0].data, value)
        assert len(hdul) == (3 if errors else 1)
