import sys

import pytest
import numpy
import astropy.io.fits as fits

from ..correction import header_add_barycentric_correction


def create_header1():
    hdr = fits.Header()
    hdr['DATE-OBS'] = '2017-08-23T21:38:30.55'
    # GTC
    hdr['OBSGEO-X'] = 5327285.0921
    hdr['OBSGEO-Y'] = -1718777.1125
    hdr['OBSGEO-Z'] = 3051786.7327

    hdr['RADEG'] = 285.481037748898
    hdr['DECDEG'] = 42.4882140636786

    hdr['CTYPE1'] = 'AWAV'
    hdr['CRPIX1'] = 1
    hdr['CRVAL1'] = 362.0
    hdr['CDELT1'] = 1.86
    hdr['CUNIT1'] = 'nm'

    hdr['CRPIX2'] = 0
    hdr['CRVAL2'] = 0
    hdr['CDELT2'] = 1
    return hdr


@pytest.mark.skipif(sys.version_info < (3, 5), reason="requires python3.5 or higher")
def test_add_barycentric():
    import astropy.wcs
    import astropy.time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.constants as cons

    hdr = create_header1()
    hdr = header_add_barycentric_correction(hdr, key='b')

    # Get main WCS
    wcs0 = astropy.wcs.WCS(hdr)

    gtc = EarthLocation.from_geocentric(wcs0.wcs.obsgeo[0], wcs0.wcs.obsgeo[1], wcs0.wcs.obsgeo[2], unit='m')
    date_obs = astropy.time.Time(wcs0.wcs.dateobs, format='fits')
    # if frame='fk5', we need to pass the epoch and equinox
    sc = SkyCoord(ra=hdr['RADEG'], dec=hdr['DECDEG'], unit='deg')
    rv = sc.radial_velocity_correction(obstime=date_obs, location=gtc)
    factor = (1 + rv / cons.c).to('').value

    # velocity
    rv = rv.to('m / s').value
    # (1 + rv / c)
    #factor = 0.9999762685198925

    assert hdr['WCSNAMEB'] == 'Barycentric correction'
    assert hdr['CTYPE1B'] == hdr['CTYPE1']
    assert hdr['CRPIX1B'] == hdr['CRPIX1']
    assert numpy.allclose(hdr['CRVAL1B'], hdr['CRVAL1'] * factor)
    assert numpy.allclose(hdr['CDELT1B'], hdr['CDELT1'] * factor)
    assert hdr['CUNIT1B'] == hdr['CUNIT1']
    #
    assert hdr['CRPIX2B'] == hdr['CRPIX2']
    assert hdr['CRVAL2B'] == hdr['CRVAL2']
    assert hdr['CDELT2B'] == hdr['CDELT2']
    #
    assert numpy.allclose(hdr['VELOSYSB'], rv)
    assert hdr['SPECSYSB'] == 'BARYCENT'
    assert hdr['SSYSOBSB'] == 'TOPOCENT'


def test_add_barycentric_missing1():
    hdr = create_header1()
    del hdr['RADEG']

    with pytest.raises(KeyError):
        header_add_barycentric_correction(hdr, key='b')


def test_add_barycentric_missing2():
    hdr = create_header1()
    del hdr['DATE-OBS']

    with pytest.raises(KeyError):
        header_add_barycentric_correction(hdr, key='b')


def test_add_barycentric_missing3():
    hdr = fits.Header()
    hdr['DATE-OBS'] = '2017-08-23T21:38:30.55'
    # GTC
    hdr['OBSGEO-X'] = 5327285.0921
    hdr['OBSGEO-Y'] = -1718777.1125
    hdr['OBSGEO-Z'] = 3051786.7327

    hdr['RADEG'] = 285.481037748898
    hdr['DECDEG'] = 42.4882140636786

    with pytest.raises(TypeError):
        header_add_barycentric_correction(hdr, key='b')
