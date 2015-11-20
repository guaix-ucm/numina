# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

import sys

import six
import numpy
from astropy.io import fits

from .qc import QC
from .pipeline import InstrumentConfiguration
from .objimport import import_object
from .oresult import ObservationResult
from .dataframe import DataFrame
from .types import DataType
from numina.frame.schema import Schema
from numina.exceptions import ValidationError
import warnings



class DataProductTag(object):
    pass


class DataProductType(DataType, DataProductTag):
    def __init__(self, ptype, default=None):
        super(DataProductType, self).__init__(ptype, default=default)


_base_schema = {
    'keywords': {
        'INSTRUME': {'valid': True},
        'READMODE': {'valid': True},
        'EXPTIME': {'value': float},
        'NUMINAID': {'value': int}
        }
    }


class DataFrameType(DataType):
    def __init__(self):
        super(DataFrameType, self).__init__(DataFrame)
        self.headerschema = Schema(_base_schema)

    def convert(self, obj):
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


class ArrayType(DataType):
    def __init__(self, default=None):
        super(ArrayType, self).__init__(ptype=numpy.ndarray, default=default)

    def convert(self, obj):
        return self.convert_to_array(obj)

    def convert_to_array(self, obj):
        result = numpy.array(obj)
        return result

    def _datatype_dump(self, obj, where):
        return dump_numpy_array(obj, where)

    def _datatype_load(self, obj):
        if isinstance(obj, six.string_types):
            # if is a string, it may be a pathname, try to load it

            # heuristics, by extension
            if obj.endswith('.csv'):
                # try to open as a CSV file
                res = numpy.loadtxt(obj, delimiter=',')
            else:
                res = numpy.loadtxt(obj)
        else:
            res = obj
        return res


class ArrayNType(ArrayType):
    def __init__(self, dimensions, default=None):
        super(ArrayNType, self).__init__(default=default)
        self.N = dimensions


# FIXME: this is hack, thus should be provided by DRPS
def _gimme_validator_for(instrument, mode):
    validators = {
        'EMIR': {
            'IMAGE_BIAS': 'emir.dataproducts.RawBias',
            'IMAGE_DARK': 'emir.dataproducts.RawDark',
            'INTENSITY_FLAT_FIELD': 'emir.dataproducts.RawIntensityFlat'
            }
        }

    if instrument not in validators:
        return DataFrameType
    else:
        modes = validators[instrument]
        if mode not in modes:
            return DataFrameType
        else:
            fqn = modes[mode]
            return import_object(fqn)
    return DataFrameType


class ObservationResultType(DataType):
    '''The type of ObservationResult.'''

    def __init__(self, rawtype=None):
        super(ObservationResultType, self).__init__(ptype=ObservationResult)
        if rawtype:
            self.rawtype = rawtype
        else:
            self.rawtype = DataFrameType

    def validate(self, obj):
        RawType = _gimme_validator_for(obj.instrument, obj.mode)
        imgtype = RawType()
        for f in obj.frames:
            imgtype.validate(f)
        return True


class InstrumentConfigurationType(DataType):
    '''The type of InstrumentConfiguration.'''

    def __init__(self):
        super(InstrumentConfigurationType, self).__init__(
            ptype=InstrumentConfiguration
            )

    def validate(self, obj):
        return True


class QualityControlProduct(DataProductType):
    def __init__(self):
        super(QualityControlProduct, self).__init__(
            ptype=QC,
            default=QC.UNKNOWN
            )


class LinesCatalog(DataProductType):
    def __init__(self):
        super(LinesCatalog, self).__init__(ptype=numpy.ndarray)

    def __numina_load__(self, obj):
        with open(obj, 'r') as fd:
            linecat = numpy.loadtxt(fd)
        return linecat


def dump_dataframe(obj, where):
    # save fits file
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
            obj.frame.writeto(filename, clobber=True)
        return filename


def dump_numpy_array(obj, where):
    # FIXME:
    #filename = where.get_next_basename('.txt')
    filename = where.destination + '.txt'
    numpy.savetxt(filename, obj)
    return filename