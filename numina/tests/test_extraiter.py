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

from six.moves.queue import Queue

from  ..extraiter import braid, iterqueue

def test_braid():
    a = iter([1,2,3,4])
    b = iter(['a', 'b'])
    c = iter([1,1,1,1,'a', 'c'])
    d = iter([1,1,1,1,1,1])
    assert(list(braid(a, b, c, d)) == [1, 'a', 1, 1, 2, 'b', 1, 1])


def test_iterqueue():
    qu = Queue()
    qu.put(1)
    qu.put(2)
    qu.put(3)
    res = list(iterqueue(qu))
    assert(res == [1, 2, 3])
