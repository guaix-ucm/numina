#
# Copyright 2008-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
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