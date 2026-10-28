# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import warnings

from astropy.io import fits

import numina.schemas
from numina.ext.gtc import DF
from numina.frame.schema import Schema
from .dataframe import DataFrame
from .datatype import DataType

_base_schema = {
    "keywords": {
        "INSTRUME": {"valid": True},
        "READMODE": {"valid": True},
        "EXPTIME": {"value": float},
        "NUMINAID": {"value": int},
    }
}


class DataFrameType(DataType):
    """A type of DataFrame."""

    db_info_keys = [
        "instrument",
        "object",
        "observation_date",
        "uuid",
        "type",
        "mode",
        "exptime",
        "darktime",
        # 'insconf',
        # 'blckuuid',
        "quality_control",
    ]

    tags_keys = []

    #: JSON schema of the headers of the frames, converted with
    #: :func:`numina.schemas.convert_headers`, or its validator. None,
    #: the headers are not validated.
    header_schema = None

    def __init__(self, datamodel=None):
        super().__init__(DataFrame, datamodel=datamodel)
        self.headerschema = Schema(_base_schema)

        self.add_dialect_info("gtc", DF.TYPE_FRAME)

    def convert(self, obj):
        """Convert"""
        # We accept None representing No Image
        if obj is None:
            return None
        elif isinstance(obj, str):
            return DataFrame(filename=obj)
        elif isinstance(obj, DataFrame):
            return obj
        elif isinstance(obj, fits.HDUList):
            return DataFrame(frame=obj)
        elif isinstance(obj, fits.PrimaryHDU):
            return DataFrame(frame=fits.HDUList([obj]))
        else:
            msg = f"object of type {obj!r} cannot be converted to DataFrame"
            raise TypeError(msg)

    def validate(self, value):
        """Validate a frame, opening it if needed.

        `value` can be None, an HDUList or a DataFrame. The contents
        are validated with :meth:`validate_hdulist`.

        Raises
        ------
        numina.exceptions.ValidationError
            If the frame is not valid.
        """
        if value is None:
            return True
        if isinstance(value, fits.HDUList):
            self.validate_hdulist(value)
        elif getattr(value, "frame", None) is not None:
            # the HDUList is in memory, it must not be closed
            self.validate_hdulist(value.frame)
        else:
            with value.open() as hdulist:
                self.validate_hdulist(hdulist)
        return True

    def validate_hdulist(self, hdulist):
        """Validate the contents of the frame.

        The headers are validated with :attr:`header_schema`, if defined,
        see :func:`numina.schemas.validate_headers`. The types of the DRPs
        override this method to check the headers and data of their frames.
        """
        if self.header_schema is not None:
            numina.schemas.validate_headers(hdulist, self.header_schema)

    def _datatype_dump(self, obj, where):
        return dump_dataframe(obj, where)

    def _datatype_load(self, obj):
        if obj is None:
            return None
        else:
            return DataFrame(filename=obj)

    def extract_db_info(self, obj, keys):
        """Extract tags from serialized file"""

        objl = self.convert(obj)

        result = super().extract_db_info(objl, keys)
        ext = self.datamodel.extractor_map["fits"]
        if objl:
            with objl.open() as hdulist:
                for field in keys:
                    result[field] = ext.extract(field, hdulist)

                tags = result["tags"]
                # for field in self.tags_keys:
                for field in self.names_t:
                    tags[field] = ext.extract(field, hdulist)

                return result
        else:
            return result


def get_filename_dataframe(obj, where):
    # save fits file
    if obj is None or obj.frame is None:
        return None
    elif obj.filename:
        return obj.filename
    else:
        return get_filename_hdulist(obj, where)


def get_filename_hdulist(hdul, hint):
    ext = ".fits"

    if isinstance(hint, str):
        filename = hint + ext
    elif callable(hint):
        filename = hint(hdul)
    else:
        raise ValueError("hint is neither string nor callable")
    return filename


def dump_dataframe(obj, where):
    # save fits file

    fname = get_filename_dataframe(obj, where)

    if fname:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj.frame.writeto(fname, overwrite=True, output_verify="warn")
    return fname
