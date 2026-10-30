# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


import numpy

from .datatype import DataType


class ArrayType(DataType):
    """A type of array.

    Parameters
    ----------
    fmt : str, optional
        Format of the numbers when the array is saved to a text file.
    default : optional
        Default value.
    """

    def __init__(self, fmt="%.18e", default=None):
        super().__init__(ptype=numpy.ndarray, default=default)
        self.fmt = fmt

    def convert(self, obj):
        """Convert `obj` to a numpy array."""
        return self.convert_to_array(obj)

    def convert_to_array(self, obj):
        """Return ``numpy.array(obj)``."""
        return numpy.array(obj)

    def _datatype_dump(self, obj, where):
        """Save the array to a text file, return the name of the file."""
        return dump_numpy_array(obj, where, self.fmt)

    def _datatype_load(self, obj):
        """Load the array from an ECSV, CSV or text file, if `obj` is a path; else return `obj`."""
        if isinstance(obj, str):
            # if is a string, it may be a pathname, try to load it

            # heuristics, by extension
            if is_ecsv(obj):
                res = load_ecsv_array(obj)
            elif obj.endswith(".csv"):
                # try to open as a CSV file
                res = numpy.loadtxt(obj, delimiter=",")
            else:
                res = numpy.loadtxt(obj)
        else:
            res = obj
        return res


class ArrayNType(ArrayType):
    """A type of array with a number of dimensions.

    Parameters
    ----------
    dimensions : int
        Number of dimensions, stored in the attribute ``N``. It is not checked.
    default : optional
        Default value.
    """

    def __init__(self, dimensions, default=None):
        super().__init__(default=default)
        self.N = dimensions


def is_ecsv(path):
    """True if the file is in ECSV format

    The first line of an ECSV file is ``# %ECSV VERSION``.
    """
    try:
        with open(path) as fd:
            return fd.readline().startswith("# %ECSV")
    except (OSError, UnicodeDecodeError):
        return False


def load_ecsv_array(path):
    """Load the numeric columns of an ECSV file in a 2D array

    The array has a row for each row of the table, and a column for
    each numeric column, in the same order. The other columns (for
    example, comments) are ignored. It is the array that loadtxt
    returns for a text file with only these columns.
    """
    from astropy.table import Table

    table = Table.read(path, format="ascii.ecsv")
    columns = [col for col in table.itercols() if col.dtype.kind in "biuf"]
    if not columns:
        raise ValueError(f"{path}: the table does not have numeric columns")
    return numpy.column_stack([numpy.asarray(col, dtype=float) for col in columns])


def get_filename_obj(hdul, hint, ext):
    """Return the name of the file where `hdul` is saved.

    Parameters
    ----------
    hdul
        The object to save.
    hint : str or callable
        If a string, the name of the file without the extension `ext`.
        If callable, it is called with `hdul` and returns the name of the file.
    ext : str
        Extension added to `hint` if it is a string.

    Raises
    ------
    ValueError
        If `hint` is neither a string nor callable.
    """

    if isinstance(hint, str):
        filename = hint + ext
    elif callable(hint):
        filename = hint(hdul)
    else:
        raise ValueError('"hint" is neither string nor callable')
    return filename


def get_filename_numpy_array(obj, where):
    """Return the name of the text file where the array `obj` is saved."""
    filename = get_filename_obj(obj, where, ".txt")
    return filename


def dump_numpy_array(obj, where, fmt="%.18e"):
    """Save the array `obj` to a text file with ``numpy.savetxt``, return the name of the file."""
    filename = get_filename_numpy_array(obj, where)
    numpy.savetxt(filename, obj, fmt=fmt)
    return filename
