#
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

"""Results of the Observing Blocks"""

import six

from astropy.io import fits

from numina.types.dataframe import DataFrame


class ObservationResult(object):
    """The result of a observing block.

    """
    def __init__(self, instrument='UNKNOWN', mode='UNKNOWN'):
        self.id = 1
        self.mode = mode
        self.instrument = instrument
        self.frames = []
        self.parent = None
        self.children = []  # other ObservationResult
        self.pipeline = 'default'
        self.configuration = 'default'
        self.prodid = None
        self.tags = {}
        self.results = {}
        self.requirements = {}

    def update_with_product(self, prod):
        self.tags = prod.tags
        self.frames = [prod.content]
        self.prodid = prod.id

    @property
    def images(self):
        return self.frames

    @images.setter
    def images(self, value):
        self.frames = value

    def __str__(self):
        return 'ObservationResult(id={}, instrument={}, mode={})'.format(
            self.id,
            self.instrument,
            self.mode
        )

    def metadata_with(self, datamodel):
        origin = {}
        imginfo = datamodel.gather_info_oresult(self)
        origin['info'] = imginfo
        if imginfo:
            first = imginfo[0]
            origin["block_uuid"] = first['block_uuid']
            origin['insconf_uuid'] = first['insconf_uuid']
            origin['date_obs'] = first['observation_date']
            origin['observation_date'] = first['observation_date']
            origin['frames'] = [img['imgid'] for img in imginfo]
        return origin

    def get_sample_frame(self):
        """Return first available image in observation result"""
        for frame in self.frames:
            return frame.open()

        for res in self.results.values():
            return res.open()

        return None


def dataframe_from_list(values):
    """Build a DataFrame object from a list."""
    if(isinstance(values, six.string_types)):
        return DataFrame(filename=values)
    elif(isinstance(values, fits.HDUList)):
        return DataFrame(frame=values)
    else:
        return None


def obsres_from_dict(values):
    """Build a ObservationResult object from a dictionary."""

    obsres = ObservationResult()

    ikey = 'frames'
    # Workaround
    if 'images' in values:
        ikey = 'images'

    obsres.id = values.get('id', 1)
    obsres.mode = values['mode']
    obsres.instrument = values['instrument']
    obsres.configuration = values.get('configuration', 'default')
    obsres.pipeline = values.get('pipeline', 'default')
    obsres.children = values.get('children',  [])
    obsres.parent = values.get('parent', None)
    obsres.results = values.get('results', {})
    obsres.requirements = values.get('requirements', {})
    try:
        obsres.frames = [dataframe_from_list(val) for val in values[ikey]]
    except Exception:
        obsres.frames = []

    return obsres
