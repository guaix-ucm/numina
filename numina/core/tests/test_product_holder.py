#
# Copyright 2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


'''Unit test for types'''

from numina.types.datatype import DataType
from ..dataholders import Result
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
        prod = Result(A)

    rr = RR(prod=100)

    assert rr.prod.val == 2