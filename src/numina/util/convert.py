#
# Copyright 2017-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Convert strings to values in data load """

import datetime

import dateutil.parser


def convert_date(value):
    """
    Convert any string to datetime

    Parameters
    ----------
    value : str or datetime

    Returns
    -------

    """
    if isinstance(value, datetime.datetime):
        return value
    if value:
        return dateutil.parser.isoparse(value)
    else:
        return None


def convert_qc(value):
    """
    Convert string to QC

    Parameters
    ----------
    value : str

    Returns
    -------
    numina.types.qc.QC

    """
    from numina.types.qc import QC
    if value:
        if isinstance(value, QC):
            return value
        else:
            return QC[value]
    else:
        return QC.UNKNOWN
