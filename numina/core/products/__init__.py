# Copyright 2008-2017 Universidad Complutense de Madrid
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

import datetime
import sys
import uuid
import warnings
from itertools import chain

import numpy
import six
from astropy.io import fits

from numina.datamodel import DataModel
from numina.exceptions import NoResultFound
from numina.exceptions import ValidationError
from numina.ext.gtc import DF
from numina.frame.schema import Schema
from ..dataframe import DataFrame
from ..oresult import ObservationResult
from ..pipeline import InstrumentConfiguration
from ..qc import QC
from ..types import DataType, DataTypeBase


class DataProductTag(DataTypeBase):
    """A type that is a data product."""

    def __init__(self, *args, **kwds):
        super(DataProductTag, self).__init__(*args, **kwds)
        self.quality_control = QC.UNKNOWN

    def generators(self):
        return []

    @classmethod
    def isproduct(cls):
        return True

    def name(self):
        """Unique name of the datatype"""
        sclass = type(self).__name__
        return "%s" % (sclass,)

    def query(self, name, dal, ob, options=None):

        try:
            return self.query_on_ob(name, ob)
        except NoResultFound:
            pass

        # If ob declares a particular ID, check that
        for g in chain([self.name()], self.generators()):
            if g in ob.results:
                resultid = ob.results[g]
                prod = dal.search_result(name, self, ob, resultid)
                break
        else:
            # if not, the normal query
            prod = dal.search_product(name, self, ob)

        return prod.content

    def extract_meta_info(self, obj):
        """Extract metadata from serialized file"""
        # print('Tag.extract_meta_info', obj)
        result = {}

        if isinstance(obj, dict):
            qc = obj['quality_control']
        elif isinstance(obj, DataFrame):
            # FIXME: use datamodel if available
            with obj.open() as hdulist:
                value = hdulist[0].header.get('NUMRQC', 'UNKNOWN')
                qc = QC[value]
        else:
            qc = QC.UNKNOWN

        result['quality_control'] = qc
        other = super(DataProductTag, self).extract_meta_info(obj)
        result.update(other)
        return result

    def __getstate__(self):
        st = {}
        st['quality_control'] = self.quality_control

        other = super(DataProductTag, self).__getstate__()
        st.update(other)
        return st

class ConfigurationTag(object):
    """A type that is part of the instrument configuration."""

    @classmethod
    def isconfiguration(cls):
        return True


class DataProductType(DataProductTag, DataType):
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


def convert_date(value):
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")


def convert_qc(value):
    if value:
        return QC[value]
    else:
        return QC.UNKNOWN


class DataFrameType(DataType):
    """A type of DataFrame."""

    meta_info_headers = {
        'instrument': 'INSTRUME',
        'object': 'OBJECT',
        'observation_date': ('DATE-OBS', 0, convert_date),
        'uuid': 'uuid',
        'type': 'numtype',
        'mode': 'obsmode',
        'exptime': 'exptime',
        'darktime': 'darktime',
        'insconf': 'insconf',
        'blckuuid': 'blckuuid',
        'quality_control': ('NUMRQC', 0, convert_qc),
    }
    tags_headers = {}

    def __init__(self, datamodel=None):
        super(DataFrameType, self).__init__(DataFrame)
        self.headerschema = Schema(_base_schema)
        if datamodel is None:
            self.datamodel = DataModel()
        else:
            self.datamodel = datamodel

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

    def extract_meta_info(self, obj):
        """Extract tags from serialized file"""

        objl = self.convert(obj)
        # empty result
        result = super(DataFrameType, self).extract_meta_info(objl)
        if objl:
            with objl.open() as hdulist:
                ext = self.datamodel.extractor
                for field in self.datamodel.meta_info_headers:
                    result[field] = ext.extract(field, hdulist)

                tags = result['tags']
                for field in self.tags_headers.items():
                    tags[field] = ext.extract(field, hdulist)

                return result
        else:
            return result



class ArrayType(DataType):
    """A type of array."""
    def __init__(self, default=None):
        super(ArrayType, self).__init__(ptype=numpy.ndarray, default=default)

    def convert(self, obj):
        return self.convert_to_array(obj)

    def convert_to_array(self, obj):
        return numpy.array(obj)

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


def _obtain_validator_for(instrument, mode_key):
    import numina.drps
    drps = numina.drps.get_system_drps()

    lol = drps.query_by_name(instrument)

    for mode in lol.modes:
        if mode.key == mode_key:
            if mode.validator:
                return mode.validator
            else:
                break

    return lambda obj: True


class ObservationResultType(DataType):
    """The type of ObservationResult."""

    def __init__(self, rawtype=None):
        super(ObservationResultType, self).__init__(ptype=ObservationResult)
        if rawtype:
            self.rawtype = rawtype
        else:
            self.rawtype = DataFrameType

    def validate(self, obj):
        super(ObservationResultType, self).validate(obj)
        validator = _obtain_validator_for(obj.instrument, obj.mode)
        return validator(obj)

    def query(self, name, dal, ob, options=None):
        return ob

    def on_query_not_found(self, notfound):
        raise notfound


class InstrumentConfigurationType(DataType):
    """The type of InstrumentConfiguration."""

    def __init__(self):
        super(InstrumentConfigurationType, self).__init__(
            ptype=InstrumentConfiguration
            )

    def validate(self, obj):
        return True

    def query(self, name, dal, ob, options=None):
        if not isinstance(ob.configuration, InstrumentConfiguration):
            warnings.warn(RuntimeWarning, 'instrument configuration not configured')
            return {}
        else:
            return ob.configuration

    def on_query_not_found(self, notfound):
        raise notfound


class QualityControlProduct(DataType):
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

    def extract_meta_info(self, obj):
        """Extract metadata from serialized file"""

        objl = self.convert(obj)

        result = super(LinesCatalog, self).extract_meta_info(objl)

        result['tags'] = {}
        result['type'] = 'LinesCatalog'
        result['uuid'] = str(uuid.uuid1())
        result['observation_date'] = datetime.datetime.utcnow()
        result['quality_control'] = QC.GOOD
        result['origin'] = {}

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
            obj.frame.writeto(filename, clobber=True, output_verify='warn')
        return filename


def dump_numpy_array(obj, where):
    # FIXME:
    #filename = where.get_next_basename('.txt')
    filename = where.destination + '.txt'
    numpy.savetxt(filename, obj)
    return filename
