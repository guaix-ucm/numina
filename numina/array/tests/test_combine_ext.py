

import numpy

import numina.array.combine as c


def test_issue34():

    data1 = [[1, 2 ,3]]
    data2 = [[8, 2, 3]]
    data3 = [[1, 2, 3]]
    data4 = [[9, 2, 3]]
    result = c.median([data1, data2, data3, data4])
    assert numpy.allclose(result[0, 0, 0], 4.5)
