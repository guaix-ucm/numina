#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import json
import datetime

import numpy

import numina.types.qc


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
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, numina.types.qc.QC):
            return obj.name
        else:
            return super(ExtEncoder, self).default(obj)
