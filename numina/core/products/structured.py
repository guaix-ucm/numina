# Copyright 2017 Universidad Complutense de Madrid
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

import json
import uuid
import datetime

import numpy
import numina.core.types
import numina.core.products
import numina.core.qc
from numina.ext.gtc import DF


def convert_date(value):
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")


class ExtEncoder(json.JSONEncoder):
    """"Encode numpy.floats and numpy.integer"""
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.timedelta):
            return obj.total_seconds()
        elif isinstance(obj, numina.core.qc.QC):
            return obj.name
        else:
            return super(ExtEncoder, self).default(obj)


class BaseStructuredCalibration(numina.core.products.DataProductTag,
                                numina.core.types.AutoDataType):
    """Base class for structured calibration data

    Parameters
    ----------

    instrument: str
        Instrument name

    Attributes
    ----------
    tags: dict
        dictionary of selection fields
    uuid: str
       UUID of the result

    """
    def __init__(self, instrument='unknown'):
        super(BaseStructuredCalibration, self).__init__()
        self.instrument = instrument
        self.tags = {}
        self.uuid = str(uuid.uuid1())

        self.meta_info = {}
        #
        self.add_dialect_info('gtc', DF.TYPE_STRUCT)

    @property
    def calibid(self):
        return 'uuid:{}'.format(self.uuid)

    @property
    def default(self):
        return None

    def __getstate__(self):

        st = super(BaseStructuredCalibration, self).__getstate__()

        keys = ['instrument', 'tags', 'uuid', 'meta_info']
        for key in keys:
            st[key] = self.__dict__[key]

        st['type'] = self.name()
        return st

    def __setstate__(self, state):
        self.instrument = state['instrument']
        self.tags = state['tags']
        self.uuid = state['uuid']
        self.meta_info = {}

        for key in state:
            if key not in ['contents']:
                setattr(self, key, state[key])

    def __str__(self):
        sclass = type(self).__name__
        if self.instrument != 'unknown':
            return "{}(instrument={}, uuid={})".format(sclass, self.instrument, self.uuid)
        else:
            return "{}()".format(sclass)

    @classmethod
    def _datatype_dump(cls, obj, where):
        filename = where.destination + '.json'

        with open(filename, 'w') as fd:
            json.dump(obj.__getstate__(), fd, indent=2, cls=ExtEncoder)

        return filename

    @classmethod
    def _datatype_load(cls, obj):
        try:
            with open(obj, 'r') as fd:
                state = json.load(fd)
        except IOError as e:
            raise e

        result = cls.__new__(cls)
        result.__setstate__(state=state)
        return result

    def extract_meta_info(self, obj):
        """Extract metadata from serialized file"""

        objl = self.convert(obj)

        try:
            with open(objl, 'r') as fd:
                state = json.load(fd)
        except IOError as e:
            raise e

        result = super(BaseStructuredCalibration, self).extract_meta_info(state)

        minfo = state['meta_info']
        origin = minfo['origin']
        date_obs = origin['date_obs']
        result['instrument'] = state['instrument']
        result['uuid'] = state['uuid']
        result['tags'] = state['tags']
        result['type'] = state['type']
        result['observation_date'] = convert_date(date_obs)
        result['origin'] = origin

        return result