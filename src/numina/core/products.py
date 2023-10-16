# Compatibility

import warnings

from numina.types.array import ArrayNType  # noqa: F401
from numina.types.array import ArrayType  # noqa: F401
from numina.types.product import DataProductMixin  # noqa: F401
from numina.types.product import DataProductTag  # noqa: F401
from numina.types.linescatalog import LinesCatalog  # noqa: F401

warnings.warn("deprecated, use numina.types instead",
              DeprecationWarning, stacklevel=2)
