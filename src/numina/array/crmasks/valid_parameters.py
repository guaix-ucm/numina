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
VALID_CRMETHODS = ["mmcosmic", "lacosmic", "mm_lacosmic", "pycosmic", "mm_pycosmic"]
VALID_BOUNDARY_FITS = ["spline", "piecewise"]
VALID_COMBINATIONS = ["mediancr", "meancrt", "meancr"]
