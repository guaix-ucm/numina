#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Quality control for Numina-based applications."""

import enum


class QC(enum.Enum):
    GOOD = 1
    PARTIAL = 2
    BAD = 3
    UNKNOWN = 4
