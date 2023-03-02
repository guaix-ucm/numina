# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import datetime
import uuid

import numpy

from numina.types.qc import QC

from .datatype import DataType
from .product import DataProductMixin


class DataProductType(DataProductMixin, DataType):
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
