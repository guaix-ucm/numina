# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import sys
import warnings

import six
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
        elif isinstance(obj, six.string_types):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        elif isinstance(obj, fits.HDUList):
            return DataFrame(frame=obj)
        elif isinstance(obj, fits.PrimaryHDU):
            return DataFrame(frame=fits.HDUList([obj]))
        else:
            msg = 'object of type %r cannot be converted to DataFrame' % obj
            raise TypeError(msg)

    def validate(self, value):
        """validate"""
        # obj can be None or a DataFrame
        if value is None:
            return True
        else:
            try:
                with value.open() as hdulist:
                    self.validate_hdulist(hdulist)
            except Exception:
                _type, exc, tb = sys.exc_info()
                six.reraise(ValidationError, exc, tb)

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
        if objl:
            with objl.open() as hdulist:
                ext = self.datamodel.extractor
                for field in keys:
                    result[field] = ext.extract(field, hdulist)

                tags = result['tags']
                for field in self.tags_keys:
                    tags[field] = ext.extract(field, hdulist)

                return result
        else:
            return result


def dump_dataframe(obj, where):
    # save fits file
    if obj is None:
        return None
    if obj.frame is None:
        # assume filename contains a FITS file
        return None
    else:
        if obj.filename:
            filename = obj.filename
        elif 'FILENAME' in obj.frame[0].header:
            filename = obj.frame[0].header['FILENAME']
        elif hasattr(where, 'destination'):
            filename = where.destination + '.fits'
        else:
            filename = where.get_next_basename('.fits')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            obj.frame.writeto(filename, overwrite=True, output_verify='warn')
        return filename
