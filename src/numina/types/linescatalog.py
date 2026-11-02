# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import datetime
import uuid

import numpy

from numina.types.qc import QC

from .array import is_ecsv, load_ecsv_array
from .product import DataProductType


class LinesCatalog(DataProductType):
    """Data product type of a catalog of lines, as a numpy array.

    The files are read in two formats:

    1. Text files, and ECSV files without envelope. The array has the
       numeric columns of the file.
    2. ECSV files with envelope, with the columns ``wavelength``,
       ``flag`` (1 for the lines used in the initial calibration),
       ``intensity`` and, optionally, ``comment``. The array has the
       columns wavelength, flag and intensity.
    """

    __numina_formats__ = {1: "_datatype_load", 2: "_load_ecsv_table"}

    #: Columns of the array read from a file in format 2
    ecsv_columns = ("wavelength", "flag", "intensity")

    def __init__(self):
        super().__init__(ptype=numpy.ndarray)

    def _load_ecsv_table(self, path):
        """Load the catalog from an ECSV file in format 2, see ``ecsv_columns``."""
        return load_ecsv_array(path, self.ecsv_columns)

    def _datatype_load(self, obj):
        """Load the catalog from an ECSV file or a text file."""
        if is_ecsv(obj):
            return load_ecsv_array(obj)
        with open(obj, "r") as fd:
            linecat = numpy.loadtxt(fd)
        return linecat

    def extract_db_info(self, obj, keys):
        """Extract metadata from serialized file"""

        objl = self.convert(obj)

        result = super().extract_db_info(objl, keys)

        result["tags"] = {}
        result["type"] = "LinesCatalog"
        result["uuid"] = str(uuid.uuid4())
        result["observation_date"] = datetime.datetime.now(datetime.UTC)
        result["quality_control"] = QC.GOOD
        result["origin"] = {}

        return result
