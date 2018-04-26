

import numpy
from numina.array.bpm import process_bpm_median


def test_process_bpm():

    data = numpy.zeros((10, 10), dtype='float32') + 3.0
    mask = numpy.zeros((10, 10), dtype='int32')
    mask[3,3] = 1

    result = process_bpm_median(data, mask)

    assert result[3,3] == 3.0


