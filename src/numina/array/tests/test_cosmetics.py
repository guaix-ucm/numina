#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numpy
import pytest

from numina.array.cosmetics import cosmetics, ccdmask


def noisy_flat(shape, seed):
    rng = numpy.random.default_rng(seed)
    return rng.normal(1.0, 0.01, size=shape)


def test_cosmetics_flat2_None():
    flat1 = numpy.ones((20, 20))
    mascara = cosmetics(flat1)
    assert not bool(mascara.all())


def test_cosmetics_no_mask():
    flat1 = numpy.ones((20, 20))
    flat2 = flat1
    mascara = cosmetics(flat1, flat2)
    assert not mascara.all()


def test_cosmetics_mask():
    flat1 = numpy.ones((2, 2))
    flat2 = flat1
    mask = numpy.zeros((2, 2), dtype="int")
    mascara = cosmetics(flat1, flat2, mask)
    assert not mascara.all()


def test_cosmetics_mask_2():
    size = 200
    flat1 = numpy.ones((size, size))
    flat1[0:100, 0] = 0
    flat2 = flat1
    mask = numpy.zeros((size, size), dtype="int")
    mask[0:10, 0] = 1
    mascara = cosmetics(flat1, flat2, mask)
    expected_mask = numpy.zeros((size, size), dtype="int")
    expected_mask[0:100, 0] = 1
    expected_mask = expected_mask.astype("bool")
    assert mascara.all() == expected_mask.all()


def test_ccdmask_flat2_None():
    flat1 = noisy_flat((20, 20), seed=1)
    bpm2 = ccdmask(flat1)
    assert not bpm2[1].all()


@pytest.mark.parametrize("with_mask", [False, True])
def test_ccdmask(with_mask):
    flat1 = noisy_flat((2000, 2000), seed=1)
    flat1[0:100, 0] = 0
    flat2 = noisy_flat((2000, 2000), seed=2)
    mask = numpy.zeros((2000, 2000), dtype="int") if with_mask else None
    ratio, bpm, sigma = ccdmask(flat1, flat2, mask, mode="full")
    assert bpm.sum() == 100
    assert bpm[0:100, 0].all()
    assert numpy.isfinite(ratio).all()
    assert sigma.min() > 0


def test_ccdmask_mask():
    flat1 = noisy_flat((2000, 2000), seed=1)
    flat1[0:100, 0] = 0
    flat2 = noisy_flat((2000, 2000), seed=2)
    mask = numpy.zeros((2000, 2000), dtype="int")
    mask[0:10, 1] = 1
    ratio, bpm, sigma = ccdmask(flat1, flat2, mask, mode="full")
    assert bpm.sum() == 110
    assert bpm[0:100, 0].all()
    assert bpm[0:10, 1].all()


def test_ccdmask_region():
    flat1 = noisy_flat((200, 200), seed=1)
    flat1[0:10, 0] = 0
    flat2 = noisy_flat((200, 200), seed=2)
    ratio, bpm, sigma = ccdmask(flat1, flat2)
    assert bpm.sum() == 10
    assert bpm[0:10, 0].all()
