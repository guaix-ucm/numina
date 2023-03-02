
import math
import numpy

from ..offrot import fit_offset_and_rotation


def test_off_rot():
    ang = -0.5
    off = numpy.array([0.23, -0.1])
    ca = math.cos(ang)
    sa = math.sin(ang)
    rot = numpy.array([[ca, -sa], [sa, ca]])

    coords0 = numpy.array(
        [[0, 0], [0, 0.5], [0, 0.1], [0, 0.2], [0.5, 0],
         [0.1, 0], [0.3, 0], [0.2, 0]]
    )
    coords1 = numpy.dot(rot, coords0.T).T + off

    off01, rot01 = fit_offset_and_rotation(coords0, coords1)

    assert numpy.allclose(off, off01)
    assert numpy.allclose(rot, rot01)


def test_off_rot_fb():

    ang = -0.5
    ca = math.cos(ang)
    sa = math.sin(ang)
    rot = numpy.array([[ca, -sa], [sa, ca]])
    off = numpy.array([0.23, -0.1])

    coords0 = numpy.array(
        [[0, 0], [0, 0.5], [1.3, 0.1], [-3.0, 0.2], [0.5, 0],
         [0.1, 0], [0.3, 0], [0.2, 0]]
    )
    c0m = coords0.mean(axis=0)
    cb0 = coords0 - c0m
    coords1 = numpy.dot(rot, cb0.T).T + off
    c1m = coords1.mean(axis=0)

    off_b = -numpy.dot(rot, c0m.T).T + off
    off_a = -numpy.dot(rot.T, c1m.T).T + c0m

    off1, rot1 = fit_offset_and_rotation(coords0, coords1)
    assert numpy.allclose(off1, off_b)
    assert numpy.allclose(rot, rot1)

    off2, rot2 = fit_offset_and_rotation(coords1, coords0)
    assert numpy.allclose(off2, off_a)
    assert numpy.allclose(numpy.transpose(rot), rot2)

def test_off_rot_182():
    p2 = [(1444.96997070312, 468.980010986328),
          (1352.80004882812, 1517.91003417969)]
    p1 = numpy.array(p2)

    q2 = numpy.array(
        [[1445.99857531, 458.24017547],
         [1353.24937729, 1507.67677833]]
    )
    q1 = numpy.array(q2)

    off1, rot1 = fit_offset_and_rotation(p1, q1)
    assert numpy.allclose(off1, [1.24147529, -11.19375707])

    off2, rot2 = fit_offset_and_rotation(q1, p1)
    assert numpy.allclose(off2, [-1.23581507, 11.19438338])

    assert numpy.allclose(rot1, rot2.T)
