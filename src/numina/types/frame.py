# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import warnings

from astropy.io import fits


from numina.exceptions import ValidationError
from numina.ext.gtc import DF
from numina.frame.schema import Schema

from .dataframe import DataFrame
from .datatype import DataType

_base_schema = {
    'keywords': {
        'INSTRUME': {'valid': True},
        'READMODE': {'valid': True},
        'EXPTIME': {'value': float},
        'NUMINAID': {'value': int}
        }
    }


class DataFrameType(DataType):
    """A type of DataFrame."""

    db_info_keys = [
        'instrument',
        'object',
        'observation_date',
        'uuid',
        'type',
        'mode',
        'exptime',
        'darktime',
        # 'insconf',
        # 'blckuuid',
        'quality_control'
    ]

    tags_keys = []

    def __init__(self, datamodel=None):
        super(DataFrameType, self).__init__(DataFrame, datamodel=datamodel)
        self.headerschema = Schema(_base_schema)

        self.add_dialect_info('gtc', DF.TYPE_FRAME)

    def convert(self, obj):
        """Convert"""
        # We accept None representing No Image
        if obj is None:
            return None
        elif isinstance(obj, str):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        elif isinstance(obj, fits.HDUList):
            return DataFrame(frame=obj)
        elif isinstance(obj, fits.PrimaryHDU):
            return DataFrame(frame=fits.HDUList([obj]))
        else:
            msg = f'object of type {obj!r} cannot be converted to DataFrame'
            raise TypeError(msg)

    def validate(self, value):
        """validate"""
        # value must be None or convertible to HDUList
        # obj can be None or a DataFrame
        if value is None:
            return True
        else:
            if isinstance(value, fits.HDUList):
                hdulist = value
            else:
                hdulist = value.open()

            self.validate_hdulist(hdulist)


    def validate_hdulist(self, hdulist):
        pass

    def _datatype_dump(self, obj, where):
        return dump_dataframe(obj, where)

    def _datatype_load(self, obj):
        if obj is None:
            return None
        else:
            return DataFrame(filename=obj)

    def extract_db_info(self, obj, keys):
        """Extract tags from serialized file"""

        objl = self.convert(obj)

        result = super(DataFrameType, self).extract_db_info(objl, keys)
        ext = self.datamodel.extractor_map['fits']
        if objl:
            with objl.open() as hdulist:
                for field in keys:
                    result[field] = ext.extract(field, hdulist)

                tags = result['tags']
                #for field in self.tags_keys:
                for field in self.names_t:
                    tags[field] = ext.extract(field, hdulist)

                return result
        else:
            return result


def get_filename_dataframe(obj, where):
    # save fits file
    if obj is None or obj.frame is None:
        return None
    elif obj.filename:
        return obj.filename
    else:
        return get_filename_hdulist(obj, where)


def get_filename_hdulist(hdul, hint):
    ext = '.fits'

    if isinstance(hint, str):
        filename = hint + ext
    elif callable(hint):
        filename = hint(hdul)
    else:
        raise ValueError('hint is neither string nor callable')
    return filename


def dump_dataframe(obj, where):
    # save fits file

    fname = get_filename_dataframe(obj, where)

    if fname:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            obj.frame.writeto(fname, overwrite=True, output_verify='warn')
    return fname
