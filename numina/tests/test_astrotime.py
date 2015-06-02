#
# Copyright 2015 Universidad Complutense de Madrid
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

'''Unit test for astrotime'''

from datetime import datetime

from  ..astrotime import datetime_to_mjd

def test_datetime_to_mjd():

    ref = datetime(year=1858, month=11, day=17)    

    assert(datetime_to_mjd(ref) == 0.0)

    current = datetime(year=2012, month=4, day=12,
                       hour=23, minute=38, second=44)

    assert(datetime_to_mjd(current) == 56029.98523148148)

