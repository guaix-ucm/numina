
# Compatibility
import sys
import warnings


from numina.types.datatype import ListOfType, DataType
import numina.types.datatype as datatype


warnings.warn('Module deprecated, use numina.types instead',
              DeprecationWarning, stacklevel=2)


sys.modules['numina.core.types.datatype'] = datatype
