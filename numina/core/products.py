# Compatibility

import warnings

warnings.warn("deprecated, use numina.types instead", DeprecationWarning, stacklevel=2)

from numina.types.array import ArrayNType
from numina.types.array import ArrayType
from numina.types.product import DataProductMixin
from numina.types.product import DataProductTag
from numina.types.linescatalog import LinesCatalog