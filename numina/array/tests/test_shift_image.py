import numpy as np
from numina.array.distortion import shift_image2d


def test_simple_offset():
    a = np.zeros(25, dtype=float).reshape(5, 5)
    a_expected = np.zeros(25, dtype=float).reshape(5, 5)

    for i in range(1, 3):
        for j in range(1, 3):
            a[i, j] = 1

    a_expected[1, 1] = 0.25
    a_expected[1, 2] = 0.50
    a_expected[1, 3] = 0.25
    a_expected[2, 1] = 0.50
    a_expected[2, 2] = 1.00
    a_expected[2, 3] = 0.50
    a_expected[3, 1] = 0.25
    a_expected[3, 2] = 0.50
    a_expected[3, 3] = 0.25

    a_shifted = shift_image2d(a, xoffset=0.5, yoffset=0.5, resampling=2)

    assert np.allclose([a.sum()], [a_expected.sum()])
    assert np.allclose([a.sum()], [a_shifted.sum()])
    assert np.allclose(a_shifted, a_expected)


if __name__ == '__main__':
    test_simple_offset()
