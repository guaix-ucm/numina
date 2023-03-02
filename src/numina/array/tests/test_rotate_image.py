import numpy as np
from numina.array.distortion import rotate_image2d


def test_simple_rotation():
    a = np.zeros(64, dtype=float).reshape(8, 8)
    for i in range(2, 6):
        for j in range(2, 6):
            a[i, j] = 1

    a_expected = np.array(
        [[0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0.34314575, 0.34314575, 0., 0., 0.],
         [0., 0., 0.34314575, 0.98528137, 0.98528137, 0.34314575, 0., 0.],
         [0., 0.34314575, 0.98528137, 1., 1., 0.98528137, 0.34314575, 0.],
         [0., 0.34314575, 0.98528137, 1., 1., 0.98528137, 0.34314575, 0.],
         [0., 0., 0.34314575, 0.98528137, 0.98528137, 0.34314575, 0., 0.],
         [0., 0., 0., 0.34314575, 0.34314575, 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0.]]
    )

    a_rotated = rotate_image2d(
        a, theta_deg=45, xcenter=4.5, ycenter=4.5, resampling=2
    )

    assert np.allclose([a.sum()], [a_expected.sum()])
    assert np.allclose([a.sum()], [a_rotated.sum()])
    assert np.allclose(a_rotated, a_expected)


if __name__ == '__main__':
    test_simple_rotation()
