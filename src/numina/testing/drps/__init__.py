"""Deprecated, the DRPs for the tests of numina are in its tests"""

import warnings

warnings.warn(
    "numina.testing.drps is deprecated and will be removed in the next version of numina, "
    "it is only used by the tests of numina",
    DeprecationWarning,
    stacklevel=2,
)
