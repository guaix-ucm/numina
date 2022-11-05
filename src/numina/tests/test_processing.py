

import numpy
import astropy.io.fits as fits
import numina.processing as proc


def test_bias_corrector():
    biasmap = numpy.zeros((10, 10)) + 200.0
    calibid = 'testbiasmapid'
    corrector = proc.BiasCorrector(biasmap=biasmap, calibid=calibid)

    data = numpy.zeros((10, 10)) + 600
    result = numpy.zeros((10, 10)) + 400
    hdu = fits.PrimaryHDU(data)
    hdu.header['DATE-OBS'] = '2010-03-11T04:34:12'
    hdul = fits.HDUList([hdu])
    newhdul = corrector.run(hdul)

    assert newhdul[0].header['NUM-BS'] == calibid
    assert numpy.allclose(newhdul[0].data, result)


def test_dark_corrector():
    darkmap = numpy.zeros((10, 10)) + 2.0
    calibid = 'testdarkmapid'
    corrector = proc.DarkCorrector(darkmap=darkmap, calibid=calibid)

    data = numpy.zeros((10, 10)) + 102.0
    result = numpy.zeros((10, 10)) + 100.0
    hdu = fits.PrimaryHDU(data)
    hdu.header['DATE-OBS'] = '2010-03-11T04:34:12'
    hdul = fits.HDUList([hdu])
    newhdul = corrector.run(hdul)

    assert newhdul[0].header['NUM-DK'] == calibid
    assert numpy.allclose(newhdul[0].data, result)
