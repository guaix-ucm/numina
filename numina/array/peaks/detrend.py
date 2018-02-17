#
# Copyright 2016 Universidad Complutense de Madrid
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

import numpy


def detrend(arr, x=None, deg=5, tol=1e-3, maxloop=10):
    """Compute a baseline trend of a signal"""

    xx = numpy.arange(len(arr)) if x is None else x
    base = arr.copy()
    trend = base
    pol = numpy.ones((deg + 1,))
    for _ in range(maxloop):
        pol_new = numpy.polyfit(xx, base, deg)
        pol_norm = numpy.linalg.norm(pol)
        diff_pol_norm = numpy.linalg.norm(pol - pol_new)
        if diff_pol_norm / pol_norm < tol:
            break
        pol = pol_new
        trend = numpy.polyval(pol, xx)
        base = numpy.minimum(base, trend)
    return trend