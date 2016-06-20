
import pytest
import numpy
from numina.array._bpm import _process_bpm_intl
from numina.array.combine import mean_method


def test_process_bpm():

    data = numpy.zeros((10.0, 10.0), dtype='float32') + 3.0
    result = numpy.empty_like(data)
    mask = numpy.zeros((10.0, 10.0), dtype='int32')
    mask[3,3] = 1

    _process_bpm_intl(mean_method(), data, mask, result)

    assert result[3,3] == 3.0

if __name__ == '__main__':
    test_process_bpm()


