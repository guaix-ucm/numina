# Compatibility

import warnings

warnings.warn("deprecated, use numina.types instead", DeprecationWarning)

from numina.types.array import ArrayNType
from numina.types.array import ArrayType
from numina.types.product import DataProductMixin
from numina.types.product import DataProductTag
from numina.types.obsresult import QualityControlProduct
from numina.types.linescatalog import LinesCatalog