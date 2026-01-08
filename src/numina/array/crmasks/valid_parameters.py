#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Valid parameters for cosmic ray masks computation and application."""

VALID_LACOSMIC_CLEANTYPE = ["median", "medmask", "meanmask", "idw"]
VALID_CRMETHODS = [
    "lacosmic",
    "mm_lacosmic",
    "pycosmic",
    "mm_pycosmic",
    "deepcr",
    "mm_deepcr",
    "conn",
    "mm_conn",
]
VALID_BOUNDARY_FITS = ["spline", "piecewise"]
VALID_COMBINATIONS = ["mean", "median", "min", "mediancr", "meancrt", "meancr"]

DEFAULT_WEIGHT_FIXED_POINTS_IN_BOUNDARY = 10000.0
