#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

'''Quality asses for Numina-based applications.'''

GOOD = 100
FAIR = 90
BAD = 70
UNKNOWN = -1

_level_names = {GOOD: 'GOOD',
                FAIR: 'FAIR',
                BAD: 'BAD',
                UNKNOWN: 'UNKNOWN'}

# A base Enum
class QA(object):
    GOOD = 1
    FAIR = 2
    BAD = 3
    UNKNOWN = 4
