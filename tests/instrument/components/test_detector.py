from numina.instrument.components.detector import DetectorBase


def test_detector_base():

    det_shape = (120, 240)
    dev = DetectorBase("detector", shape=det_shape)
    arr = dev.readout()
    assert arr.shape == det_shape


def test_detector_qe_wl_default():
    import numpy

    dev = DetectorBase("detector", shape=(10, 10))
    wl = numpy.linspace(4000.0, 9000.0, 5)
    numpy.testing.assert_array_equal(dev.qe_wl(wl), numpy.ones_like(wl))


def test_detector_qe_wl():
    import numpy

    class HalfEfficiency:
        def response(self, wl):
            return 0.5 * numpy.ones_like(wl)

    dev = DetectorBase("detector", shape=(10, 10), qe_wl=HalfEfficiency())
    numpy.testing.assert_array_equal(dev.qe_wl(numpy.array([5000.0])), [0.5])
