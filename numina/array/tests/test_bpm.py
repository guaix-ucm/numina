

import numpy
from numina.array.bpm import process_bpm_median


def test_process_bpm():

    data = numpy.zeros((10, 10), dtype='float32') + 3.0
    mask = numpy.zeros((10, 10), dtype='int32')
    mask[3,3] = 1

    result1 = process_bpm_median(data, mask)

    assert result1[3,3] == 3.0

    result2, subs2 = process_bpm_median(data, mask, subs=True)

    assert result2[3,3] == 3.0
    assert subs2.min() == 1


def test_process_bpm_large_hole():

    data = numpy.zeros((100, 100), dtype='float32') + 3.0
    mask = numpy.zeros((100, 100), dtype='int32')
    mask[30:40,30:40] = 1

    fill = 0.1
    result, subs = process_bpm_median(data, mask, fill=fill, subs=True)

    assert result[35,35] == fill
    assert subs.sum() == 100 * 100 - 36

    result1, subs1 = process_bpm_median(data, mask, reuse_values=True, subs=True)

    assert result1[35,35] == 3.0
    assert subs1.sum() == 100 * 100

    result2 = process_bpm_median(data, mask, reuse_values=True, subs=False)

    assert result2[35,35] == 3.0
