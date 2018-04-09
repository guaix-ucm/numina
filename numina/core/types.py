
# Compatibility
import sys
import warnings

from numina.types.datatype import ListOfType
import numina.types.datatype as datatype


warnings.warn('Module deprecated, use numina.types instead',
              DeprecationWarning)


sys.modules['numina.core.types.datatype'] = datatype
