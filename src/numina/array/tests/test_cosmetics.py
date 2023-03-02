#
# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest

import numpy

from numina.array.cosmetics import cosmetics, ccdmask

def test_cosmetics_flat2_None():
    flat1 = numpy.ones((20, 20))
    mascara = cosmetics(flat1)
    assert mascara.all() == False

def test_cosmetics_no_mask():
    flat1 = numpy.ones((20, 20))
    flat2 = flat1
    mascara = cosmetics(flat1,flat2)
    assert mascara.all() == False


def test_cosmetics_mask():
    flat1 = numpy.ones((2, 2))
    flat2 = flat1
    mask = numpy.zeros((2, 2), dtype='int')
    mascara = cosmetics( flat1,flat2, mask)
    assert mascara.all() == False

def test_cosmetics_mask_2():
    size = 200
    flat1 = numpy.ones((size, size))
    flat1[0:100,0] = 0
    flat2 = flat1
    mask = numpy.zeros((size, size), dtype='int')
    mask[0:10,0] = 1
    mascara = cosmetics( flat1,flat2, mask)
    expected_mask = numpy.zeros((size, size), dtype='int')
    expected_mask[0:100,0] = 1
    expected_mask = expected_mask.astype('bool')
    assert mascara.all() == expected_mask.all()

def test_ccdmask_flat2_None():
    flat1 = numpy.ones((20, 20))
    bpm2 = ccdmask(flat1)
    assert bpm2[1].all() == False

def test_ccdmask_no_mask():
    flat1 = numpy.ones((2000, 2000))
    flat1[0:100,0] = 0
    flat2 = flat1
    bpm2 = ccdmask(flat1, flat2, mode='full')
    assert bpm2[1].sum() == 100

def test_ccdmask_empty_mask():
    flat1 = numpy.ones((2000, 2000))
    flat1[0:100,0] = 0
    flat2 = flat1
    mask = numpy.zeros((2000, 2000), dtype='int')
    bpm2 = ccdmask(flat1, flat2, mask, mode='full')
    assert bpm2[1].sum() == 100

def test_ccdmask_mask():
    flat1 = numpy.ones((2000, 2000))
    flat1[0:100,0] = 0
    flat2 = flat1
    mask = numpy.zeros((2000, 2000), dtype='int')
    mask[0:10,0] = 0
    bpm2 = ccdmask(flat1, flat2, mask, mode='full')
    assert bpm2[1].sum() == 100

if __name__ == "__main__":
    test_cosmetics_flat2_None()
    test_cosmetics_mask()
    test_cosmetics_no_mask()
    test_cosmetics_mask_2()
    test_ccdmask_no_mask()
    test_ccdmask_empty_mask()
    test_ccdmask_mask()