
from numina.core import DataFrameType
from numina.core import DataFrame

from .load import load

'''Register basic types with load.'''

@load.register(DataFrameType)
def _(tag, obj):
    return load.registry[DataFrame](tag, obj)


@load.register(DataFrame)
def _(tag, obj):

    if obj is None:
        return None
    else:
        return DataFrame(filename=obj)

def load_cli_storage():
    return 0