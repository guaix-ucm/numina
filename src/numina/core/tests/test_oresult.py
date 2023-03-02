
import astropy.io.fits as fits

from ..oresult import ObservationResult
import numina.datamodel
import numina.core
import numina.types.dataframe as dataframe


def test_oresult_empty():

    datamodel = numina.datamodel.DataModel()
    ob = ObservationResult()
    meta = ob.metadata_with(datamodel)
    assert meta == {'info': []}


def test_oresult1():

    datamodel = numina.datamodel.DataModel()
    ob = ObservationResult()
    for i in range(3):
        hdu = fits.PrimaryHDU()
        hdu.header['BLCKUUID'] = '1'
        hdu.header['INSTRUME'] = 'TEST'
        hdu.header['UUID'] = '1'
        hdu.header['DATE-OBS'] = '2018-04-12T22:44:12.3'
        hdu.header['INSCONF'] = 'v1'
        hdu.header['FILENAME'] = f'img{i}.fits'
        hdu.header['EXPTIME'] = 3.0

        hdulist = fits.HDUList(hdu)
        ob.frames.append(dataframe.DataFrame(frame=hdulist))

    meta = ob.metadata_with(datamodel)
    
    assert meta['block_uuid'] == '1'
    assert len(meta['info']) == 3