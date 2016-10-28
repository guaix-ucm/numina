#
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

"""Results of the Observing Blocks"""

import six

from astropy.io import fits

from .dataframe import DataFrame


class ObservationResult(object):
    """The result of a observing block.

    """
    def __init__(self, mode=None):
        self.id = 1
        self.mode = mode
        self.instrument = None
        self.frames = []
        self.parent = None
        self.children = []  # other ObservationResult
        self.pipeline = 'default'
        self.configuration = 'default'
        self.prodid = None
        self.tags = {}

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
    try:
        obsres.frames = [dataframe_from_list(val) for val in values[ikey]]
    except Exception:
        obsres.frames = []

    return obsres
