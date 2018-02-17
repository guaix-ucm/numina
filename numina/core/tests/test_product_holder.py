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


'''Unit test for types'''

from numina.types.datatype import DataType
from ..dataholders import Product
from ..recipeinout import RecipeResult


def test_product_out():

    class AVal(object):
        def __init__(self, val):
            self.val = val

    class A(DataType):

        def __init__(self):
            super(A, self).__init__(ptype=AVal)

        def convert(self, obj):
            return AVal(val=1)

        def convert_out(self, obj):
            return AVal(val=2)

    class RR(RecipeResult):
        prod = Product(A)

    rr = RR(prod=100)

    assert rr.prod.val == 2