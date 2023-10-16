
# Compatibility

import warnings

from numina.types.qc import QC  # noqa: F401

warnings.warn("deprecated, use numina.types.qc instead",
              DeprecationWarning, stacklevel=2)
