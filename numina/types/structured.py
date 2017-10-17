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

import numina.types.product
import numina.types.datatype
from numina.ext.gtc import DF
import numina.util.convert as conv
from numina.util.jsonencoder import ExtEncoder


def writeto(obj, name):
    with open(name, 'w') as fd:
        json.dump(obj.__getstate__(), fd, indent=2, cls=ExtEncoder)


class BaseStructuredCalibration(numina.types.product.DataProductTag,
                                numina.types.datatype.AutoDataType):
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

        self.meta_info = self.create_meta_info()
        self.meta_info['instrument_name'] = self.instrument
        self.meta_info['creation_date'] = datetime.datetime.utcnow().isoformat()
        self._base_info = {}
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
        super(BaseStructuredCalibration, self).__setstate__(state)
        # self.add_dialect_info('gtc', DF.TYPE_STRUCT)

        self.instrument = state['instrument']
        self.tags = state['tags']
        self.uuid = state['uuid']
        self.meta_info = {}
        self._base_info = {}
        for key in state:
            if key not in ['contents', 'quality_control']:
                setattr(self, key, state[key])

    def __str__(self):
        sclass = type(self).__name__
        if self.instrument != 'unknown':
            return "{}(instrument={}, uuid={})".format(sclass, self.instrument, self.uuid)
        else:
            return "{}()".format(sclass)

    def writeto(self, name):
        return  writeto(self, name)

    @classmethod
    def _datatype_dump(cls, obj, where):

        filename = where.destination + '.json'
        writeto(obj, filename)
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

    @staticmethod
    def create_meta_info():
        meta_info = {}
        meta_info['mode_name'] = 'unknown'
        meta_info['instrument_name'] = 'unknown'
        meta_info['recipe_name'] = 'unknown'
        meta_info['recipe_version'] = 'unknown'
        meta_info['origin'] = {}
        return meta_info

    def extract_db_info(self, obj):
        """Extract metadata from serialized file"""

        objl = self.convert_in(obj)

        try:
            with open(objl, 'r') as fd:
                state = json.load(fd)
        except IOError as e:
            raise e

        result = super(BaseStructuredCalibration, self).extract_db_info(state)

        try:
            minfo = state['meta_info']
            result['mode'] = minfo['mode_name']
            origin = minfo['origin']
            date_obs = origin['date_obs']
        except KeyError:
            origin = {}
            date_obs = "1970-01-01T00:00:00.00"

        result['instrument'] = state['instrument']
        result['uuid'] = state['uuid']
        result['tags'] = state['tags']
        result['type'] = state['type']
        result['observation_date'] = conv.convert_date(date_obs)
        result['origin'] = origin

        return result

    def update_meta_info(self):
        """Extract metadata from myself"""
        result = super(BaseStructuredCalibration, self).update_meta_info()

        result['instrument'] = self.instrument
        result['uuid'] = self.uuid
        result['tags'] = self.tags
        result['type'] = self.name()

        minfo = self.meta_info
        try:
            result['mode'] = minfo['mode_name']
            origin = minfo['origin']
            date_obs = origin['date_obs']
        except KeyError:
            origin = {}
            date_obs = "1970-01-01T00:00:00.00"

        result['observation_date'] = conv.convert_date(date_obs)
        result['origin'] = origin

        return result

    @property
    def meta(self):
        self._base_info = self.update_meta_info()
        return self._base_info

    def update_metadata(self, recipe):
        self.meta_info['mode_name'] = recipe.mode
        self.meta_info['instrument_name'] = recipe.instrument
        self.meta_info['recipe_name'] = recipe.__class__.__name__
        self.meta_info['recipe_version'] = recipe.__version__

    def update_metadata_origin(self, obresult_meta):
        origin = self.meta_info['origin']
        origin['block_uuid'] = obresult_meta['block_uuid']
        origin['insconf_uuid'] = obresult_meta['insconf_uuid']
        origin['date_obs'] = obresult_meta['date_obs']
        origin['frames'] = obresult_meta['frames']
