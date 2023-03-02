
from ..detector import DetectorBase


def test_detector_base():

    det_shape = (120, 240)
    dev = DetectorBase('detector', shape=det_shape)
    arr = dev.readout()
    assert arr.shape == det_shape
