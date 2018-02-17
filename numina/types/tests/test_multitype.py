#
# Copyright 2017 Universidad Complutense de Madrid
#
# This file is part of Numina DRP
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Unit test for MultiType adapter"""

import pytest

from ..datatype import DataType
from ..multitype import MultiType


def test_convert_in():

    class AVal(object):
        pass

    class BVal(object):
        pass

    class A(DataType):

        def convert(self, obj):
            return 1

    class B(DataType):

        def convert(self, obj):
            return 2

    a = A(ptype=AVal)
    b = B(ptype=BVal)

    multi = MultiType(a, b)

    with pytest.raises(ValueError):
        multi.convert_in(None)
