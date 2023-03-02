#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import warnings

import numina.util.convert as conv


class QueryAttribute(object):
    def __init__(self, name, tipo, description=""):
        self.name = name
        self.type = tipo
        self.description = description


class KeyDefinition(object):
    def __init__(self, key, ext=None, default=None, convert=None):
        self.key = key
        self.ext = 0 if ext is None else ext
        self.default = default
        self.convert = convert

    def __call__(self, hdulist):
        value = hdulist[self.ext].header.get(self.key, self.default)
        if self.convert:
            return self.convert(value)
        return value


class FITSKeyExtractor(object):
    """Extract values from FITS images"""
    def __init__(self, values):
        self.map = {}
        for key, entry in values.items():
            if isinstance(entry, KeyDefinition):
                newval = entry
            elif isinstance(entry, tuple):
                if len(entry) == 3:
                    keyname = entry[0]
                    hduname = entry[1]
                    convert = entry[2]
                    default = None
                elif len(entry) == 2:
                    keyname = entry[0]
                    default = entry[1]
                    hduname = 0
                    convert = None
                else:
                    raise ValueError

                newval = KeyDefinition(
                    keyname,
                    ext=hduname,
                    convert=convert,
                    default=default
                )
            elif isinstance(entry, str):
                newval = KeyDefinition(
                    entry
                )
            else:
                newval = entry

            self.map[key] = newval

    def extract(self, value, hdulist):
        extractor = self.map[value]
        return extractor(hdulist)


class DataModel(object):
    """Model of the Data being processed

    Parameters
    ==========
    name : str
        Name of the DataModel
    mappings: dict, optional
        A dictionary of str keys and values convertibles to FITSKeyExtractor

    """

    db_info_keys = [
        'instrument',
        'object',
        'observation_date',
        'uuid',
        'type',
        'mode',
        'exptime',
        'darktime',
        'insconf',
        'blckuuid',
        'quality_control',
    ]
    db_info_keys_extra = []
    query_attrs = {}
    meta_dinfo_headers = [
        'instrument',
        'object',
        'observation_date',
        'uuid',
        'type',
        'mode',
        'exptime',
        'darktime',
        'insconf',
        'blckuuid',
        'quality_control',
        'block_uuid',  # Alias
        'insconf_uuid',  # Alias
        'imgid',
    ]

    def __init__(self, name='UNKNOWN', mappings=None):

        self.name = name
        values = self.default_mappings()
        more = {} if mappings is None else mappings
        values.update(more)

        self.extractor_map = dict()
        self.extractor_map['fits'] = FITSKeyExtractor(values)
        self.extractor_map['json'] = None
        self.extractor = self.extractor_map['fits']

    def default_mappings(self):
        return {
            'instrument': 'INSTRUME',
            'object': 'OBJECT',
            'observation_date': ('DATE-OBS', 0, conv.convert_date),
            'uuid': 'uuid',
            'type': 'numtype',
            'mode': 'obsmode',
            'exptime': self.get_exptime,
            'darktime': self.get_darktime,
            'quality_control': ('NUMRQC', 0, conv.convert_qc),
            'insmode': ('INSMODE', 'undefined'),
            'imgid': self.get_imgid,
            'insconf': 'INSCONF',
            'blckuuid': lambda x: '1',
            'insconf_uuid': 'insconf', # Alias
            'block_uuid': 'blckuuid', # Alias
        }

    def get_data(self, img):
        """Obtain the primary data from the image"""
        return img['primary'].data

    def get_header(self, img):
        """Obtain the primary header from the image"""
        return img['primary'].header

    def get_variance(self, img):
        """Obtain the variance of the primary data from the image"""
        return img['variance'].data

    def get_imgid(self, img):
        """Obtain a unique identifier of the image.

        Parameters
        ----------
        img : astropy.io.fits.HDUList

        Returns
        -------
        str:
             Identification of the image

        """
        return get_imgid(img, prefix=False)

    def get_darktime(self, img):
        """Obtain DARKTIME"""
        header = self.get_header(img)
        if 'DARKTIME' in header.keys():
            return header['DARKTIME']
        else:
            return self.get_exptime(img)

    def get_exptime(self, img):
        """Obtain EXPTIME"""
        header = self.get_header(img)
        if 'EXPTIME' in header.keys():
            etime = header['EXPTIME']

        elif 'EXPOSED' in header.keys():
            etime = header['EXPOSED']
        else:
            etime = 1.0
        return etime

    def do_sky_correction(self, img):
        """Perform sky correction"""
        return True

    def gather_info(self, dframe):
        """Obtain a summary of information about the image."""
        with dframe.open() as hdulist:
            info = self.gather_info_hdu(hdulist)
        return info

    def gather_info_dframe(self, img):
        """Obtain a summary of information about the image."""
        return self.gather_info(img)

    def gather_info_hdu(self, hdulist):
        """Obtain a summary of information about the image."""
        values = {}
        values['n_ext'] = len(hdulist)
        extnames = [hdu.header.get('extname', '') for hdu in hdulist[1:]]
        values['name_ext'] = ['PRIMARY'] + extnames

        fits_extractor = self.extractor_map['fits']
        for key in self.meta_dinfo_headers:
            values[key] = fits_extractor.extract(key, hdulist)

        return values

    def gather_info_oresult(self, ob):
        return [self.gather_info_dframe(f) for f in ob.images]

    def get_quality_control(self, img):
        """Obtain quality control flag from the image."""
        fits_extractor = self.extractor_map['fits']
        return fits_extractor.extract('quality_control', img)


def get_imgid(img, prefix=True):
    """Obtain a unique identifier of the image.

    Parameters
    ----------
    img : astropy.io.fits.HDUList

    Returns
    -------
    str:
         Identification of the image

    """
    hdr = img[0].header
    try:
        return get_imgid_header(hdr, prefix=prefix)
    except ValueError:
        warnings.warn("no method to identity image", RuntimeWarning)
        value = repr(img)
        pre = 'py:{}'
        base = '{}'
        if prefix:
            return pre.format(value)
        else:
            return base.format(value)


def get_imgid_header(hdr, prefix=True):
    """Obtain a unique identifier of the image.

    Parameters
    ----------
    hdr : astropy.io.fits.Header

    Returns
    -------
    str:
         Identification of the image

    """
    base = "{}"
    if 'UUID' in hdr:
        pre = 'uuid:{}'
        value = hdr['UUID']
    elif 'EMIRUUID' in hdr:  # EMIRISM
        pre = 'uuid:{}'
        value = hdr['EMIRUUID']
    elif 'DATE-OBS' in hdr:
        pre = 'dateobs:{}'
        value = hdr['DATE-OBS']
    elif 'TSUTC1' in hdr:   # EMIRISM
        pre = 'tsutc:{:16.5f}'
        value = hdr['TSUTC1']
    elif 'checksum' in hdr:
        pre = 'checksum:{}'
        value = hdr['checksum']
    elif 'filename' in hdr:
        pre = 'file:{}'
        value = hdr['filename']
    else:
        raise ValueError('no method to identity image')
    if prefix:
        return pre.format(value)
    else:
        return base.format(value)
