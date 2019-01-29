
import sys
import warnings

import pytest

from ..load import load


def test_load_base():

    class A(object):
        pass

    tag = A()
    obj = 0

    assert obj == load(tag, obj)


def test_load_method():

    class A(object):

        def _datatype_load(self, obj):
            return obj + 1

    tag = A()
    obj = 1

    assert obj + 1 == load(tag, obj)


def test_load_method_deprecated():

    class B(object):

        def __numina_load__(self, obj):
            return obj + 2

    tag = B()
    obj = 1

    assert obj + 2 == load(tag, obj)


# @pytest.mark.skipif(sys.version_info < (3, 4),
#                     reason="https://github.com/pytest-dev/pytest/issues/840")
# def test_dump_method_deprecated_warning(recwarn):
#     warnings.simplefilter('always')
#     test_load_method_deprecated()
#
#     assert recwarn.pop(DeprecationWarning)


def test_load_method_register():

    class C(object):
        pass

    def numina_load_func(tag, obj):
        return obj + 3

    load.register(C, numina_load_func)

    tag = C()
    obj = 1

    assert obj + 3 == load(tag, obj)

