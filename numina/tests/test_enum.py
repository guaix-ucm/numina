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

from ..enum import Enum


def test_enum():
    
    class Color(Enum):
        red = 1
        green = 2
        blue = 3
    
    assert Color.red is Color.red
    
    for i in Color:
        assert i
        
    class OtherColor(Enum):
        red = 1
        green = 2
        blue = 3
    
    # The same name in different classes can't be compared
    assert Color.red != OtherColor.red

