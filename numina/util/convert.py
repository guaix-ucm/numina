#
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

"""Convert strings to values in data load """

import datetime


def convert_date(value):
    if value:
        return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
    else:
        return datetime.datetime.utcnow()


def convert_qc(value):
    from numina.types.qc import QC
    if value:
        if isinstance(value, QC):
            return value
        else:
            return QC[value]
    else:
        return QC.UNKNOWN