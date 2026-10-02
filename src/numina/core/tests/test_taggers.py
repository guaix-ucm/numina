#
# Copyright 2015-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Unit test for taggers"""

import pytest

import astropy.io.fits as fits

from numina.datamodel import DataModel
from numina.types.dataframe import DataFrame
from ..oresult import ObservationResult
from ..taggers import extract_tags_from_obsres


@pytest.fixture
def my_datamodel():
    mappings = {"filter": "filter", "read_mode": "readm"}
    model = DataModel("TEST", mappings=mappings)
    return model


def test_empty_ob(my_datamodel):

    ob = ObservationResult()
    tags = extract_tags_from_obsres(ob, tag_keys=[], datamodel=my_datamodel)

    assert len(tags) == 0


def test_init_ob(my_datamodel):

    img1 = fits.PrimaryHDU(data=[1, 2, 3])
    frame1 = DataFrame(frame=fits.HDUList(img1))

    ob = ObservationResult()
    ob.frames = [frame1]
    tags = extract_tags_from_obsres(ob, tag_keys=[], datamodel=my_datamodel)

    assert len(tags) == 0


def test_header_key1_ob(my_datamodel):

    img1 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img1.header["FILTER"] = "FILTER-A"
    img1.header["READM"] = "MOD1"
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img2.header["FILTER"] = "FILTER-A"
    img2.header["READM"] = "MOD2"
    frame2 = DataFrame(frame=fits.HDUList(img2))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]
    tags = extract_tags_from_obsres(ob, tag_keys=["filter"], datamodel=my_datamodel)

    assert tags == {"filter": "FILTER-A"}


def test_header_key1_mis(my_datamodel):
    """Test extract_tags raises ValueError if there is a missmatch"""
    img1 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img1.header["FILTER"] = "FILTER-A"
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img2.header["FILTER"] = "FILTER-B"
    frame2 = DataFrame(frame=fits.HDUList(img2))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]

    with pytest.raises(ValueError):
        extract_tags_from_obsres(ob, tag_keys=["filter"], datamodel=my_datamodel)


def test_header_key1_mis_no_strict(my_datamodel):
    """Test extract_tags returns is struct is false"""
    img1 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img1.header["FILTER"] = "FILTER-A"
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img2.header["FILTER"] = "FILTER-B"
    frame2 = DataFrame(frame=fits.HDUList(img2))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]

    tags = extract_tags_from_obsres(ob, tag_keys=["filter"], datamodel=my_datamodel, strict=False)
    assert tags == {"filter": "FILTER-A"}


def test_header_key2_ob(my_datamodel):

    img1 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img1.header["FILTER"] = "FILTER-A"
    img1.header["READM"] = "MOD1"
    frame1 = DataFrame(frame=fits.HDUList(img1))

    img2 = fits.PrimaryHDU(data=[1, 2, 3], header=fits.Header())
    img2.header["FILTER"] = "FILTER-A"
    img2.header["READM"] = "MOD1"
    frame2 = DataFrame(frame=fits.HDUList(img2))

    ob = ObservationResult()
    ob.frames = [frame1, frame2]
    tags = extract_tags_from_obsres(ob, tag_keys=["filter", "read_mode"], datamodel=my_datamodel)

    assert tags == {"filter": "FILTER-A", "read_mode": "MOD1"}
