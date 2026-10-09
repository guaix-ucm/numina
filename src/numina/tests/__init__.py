#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Deprecated, the utilities for testing are in numina.testing"""

import importlib
import sys
import warnings

warnings.warn("numina.tests is deprecated, use numina.testing", DeprecationWarning, stacklevel=2)

# The modules of numina.testing are also available with their old names
for _name in [
    "drpmocker",
    "drptest",
    "nobenchmark",
    "plugins",
    "plugins_pattri",
    "pytest_resultcmp",
    "recipes",
    "seffect",
    "simpleobj",
    "testcache",
]:
    _module = importlib.import_module(f"numina.testing.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
