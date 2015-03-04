from __future__ import print_function


try:
    from functools import singledispatch
except ImportError:
    from singledispatch import singledispatch

from numina.core import RecipeResult

@singledispatch
def dump(tag, obj, where):
    print('default impl for', type(tag))
    return obj

