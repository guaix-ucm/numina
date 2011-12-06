#
# Copyright 2011 Universidad Complutense de Madrid
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

# MJD 0 is 1858-11-17 00:00:00.00 
_MJDREF = datetime.datetime(year=1858, month=11, day=17)

def datetime_to_mjd(dt):
    diff = dt - _MJDREF
    result  = diff.days + (diff.seconds + diff.microseconds / 1e6) / 86400.0
    return result

