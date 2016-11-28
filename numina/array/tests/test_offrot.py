
import math
import numpy

from ..offrot import fit_offset_and_rotation


def test_off_rot():

    coords0 = numpy.array(
        [[0, 0], [0, 0.5], [0, 0.1], [0, 0.2], [0.5, 0], [0.1, 0], [0.3, 0], [0.2, 0]]
    )
    ang = -0.5
    off = numpy.array([0.23, -0.1])

    ca = math.cos(ang)
    sa = math.sin(ang)
    rot = numpy.array([[ca, -sa], [sa, ca]])

    coords1 = numpy.dot(rot, coords0.T).T + off

    Off, R = fit_offset_and_rotation(coords0, coords1)

    assert numpy.allclose(Off, off)
    assert numpy.allclose(rot, R)
