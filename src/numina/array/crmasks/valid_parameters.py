#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Valid parameters for cosmic ray masks computation and application."""

#: Valid values of ``la_cleantype`` in :func:`~numina.array.crmasks.compute_crmasks.compute_crmasks`, besides ``'none'``
VALID_LACOSMIC_CLEANTYPE = ["median", "medmask", "meanmask", "idw"]
#: Valid values of ``crmethod`` in :func:`~numina.array.crmasks.compute_crmasks.compute_crmasks`
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
#: Valid values of ``mm_boundary_fit`` in :func:`~numina.array.crmasks.compute_crmasks.compute_crmasks`
VALID_BOUNDARY_FITS = ["spline", "piecewise"]
#: Valid values of ``combination`` in :func:`~numina.array.crmasks.apply_crmasks.apply_crmasks`
VALID_COMBINATIONS = ["mean", "median", "min", "mediancr", "meancrt", "meancr", "meancr2"]

#: Weight of the fixed points in the boundary given as (x, y), without weight
DEFAULT_WEIGHT_FIXED_POINTS_IN_BOUNDARY = 10000.0
