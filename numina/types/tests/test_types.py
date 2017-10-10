#
# Copyright 2008-2016 Universidad Complutense de Madrid
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


'''Unit test for types'''

from ..datatype import NullType, DataType

def test_null_type():
    '''Test NullType.'''

    nullt = NullType()

    values = [None, 1, 1.0, [1,2,3], {'a': 1, 'b': 2}]

    for val in values:
        assert nullt.convert(val) is None

    for val in values:
        assert nullt.validate(nullt.convert(val))


def test_convert_in_out():

    class AVal(object):
        pass

    class A(DataType):

        def convert(self, obj):
            return 1

    a = A(ptype=AVal)
    assert a.convert_in(None) == 1
    assert a.convert_out(None) == 1

    class B(A):

        def convert_out(self, obj):
            return 2

    b = B(ptype=AVal)
    assert b.convert_in(None) == 1
    assert b.convert_out(None) == 2