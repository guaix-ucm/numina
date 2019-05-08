

"""Create simple objects for testing"""


def create_simple_hdul():
    """Create a simple image for testing"""
    import astropy.io.fits as fits

    simple_img = fits.HDUList([fits.PrimaryHDU()])
    return simple_img


def create_simple_frame():
    """Create a simple image frame for testing"""
    from numina.types.frame import DataFrame

    simple_img = create_simple_hdul()
    simple_frame = DataFrame(frame=simple_img)
    return simple_frame


def create_simple_structured():
    """Create a simple structred object for testing"""
    from numina.types.structured import BaseStructuredCalibration
    import numina.types.qc as qc

    obj = BaseStructuredCalibration()
    obj.quality_control = qc.QC.BAD
    return obj
