#
# Copyright 2017-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Convert strings to values in data load """

import datetime

import dateutil.parser

# Try to use isoparse ISO-8601, if not available
# use generic parser
try:
    isoparse = dateutil.parser.isoparse
except AttributeError:
    isoparse = dateutil.parser.parse


def convert_date(value):
    if isinstance(value, datetime.datetime):
        return value
    if value:
        return isoparse(value)
    else:
        return None


def convert_qc(value):
    from numina.types.qc import QC
    if value:
        if isinstance(value, QC):
            return value
        else:
            return QC[value]
    else:
        return QC.UNKNOWN