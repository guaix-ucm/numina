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

'''Interpolation in FITS header templates.'''


def isinterpolable(v):
    if isinstance(v, basestring) and v[:2] == '%(' and v[-1] == ')':
        return v[2:-1]
    else:
        return None


def interpolate(meta, v):
    key = isinterpolable(v)
    if key is not None:
        ival = meta[key]
        if callable(ival):
            return ival()
        else:
            return ival
    else:
        return v
