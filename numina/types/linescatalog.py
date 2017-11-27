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
import uuid

import numpy

from numina.core.qc import QC

from .datatype import DataType
from .product import DataProductTag


class DataProductType(DataProductTag, DataType):
    def __init__(self, ptype, default=None):
        super(DataProductType, self).__init__(ptype, default=default)


class LinesCatalog(DataProductType):
    def __init__(self):
        super(LinesCatalog, self).__init__(ptype=numpy.ndarray)

    def __numina_load__(self, obj):
        with open(obj, 'r') as fd:
            linecat = numpy.loadtxt(fd)
        return linecat

    def extract_db_info(self, obj, keys):
        """Extract metadata from serialized file"""

        objl = self.convert(obj)

        result = super(LinesCatalog, self).extract_db_info(objl, keys)

        result['tags'] = {}
        result['type'] = 'LinesCatalog'
        result['uuid'] = str(uuid.uuid1())
        result['observation_date'] = datetime.datetime.utcnow()
        result['quality_control'] = QC.GOOD
        result['origin'] = {}

        return result
