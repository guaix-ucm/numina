from __future__ import print_function


try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch


@singledispatch
def load(tag, obj):

    if hasattr(tag, '__numina_load__'):
        return tag.__numina_load__(obj)

    return obj

