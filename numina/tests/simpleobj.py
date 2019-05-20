import numpy

"""Create simple objects for testing"""


def create_simple_hdul(shape=(1, 1), value=0.0):
    """Create a simple image for testing"""
    import astropy.io.fits as fits

    data = value * numpy.ones(shape, dtype='float32')
    header = fits.Header()
    header['DATE-OBS'] = '2001-09-12T22:45:12.343'
    simple_img = fits.HDUList([fits.PrimaryHDU(data=data, header=header)])
    return simple_img


def create_simple_frame(shape=(1, 1), value=0.0):
    """Create a simple image frame for testing"""
    from numina.types.frame import DataFrame

    simple_img = create_simple_hdul(shape, value=value)
    simple_frame = DataFrame(frame=simple_img)
    return simple_frame


def create_simple_structured():
    """Create a simple structred object for testing"""
    from numina.types.structured import BaseStructuredCalibration
    import numina.types.qc as qc

    obj = BaseStructuredCalibration()
    obj.quality_control = qc.QC.BAD
    return obj
