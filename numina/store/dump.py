from __future__ import print_function


try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch


@singledispatch
def dump(tag, obj, where):
    return obj

