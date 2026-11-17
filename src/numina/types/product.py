# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


from numina.util.convert import convert_qc

from .base import DataTypeBase
from .datatype import DataType
from .dataframe import DataFrame
from .qc import QC


class DataProductMixin(DataTypeBase):
    """A type that is a data product."""

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.quality_control = QC.UNKNOWN

    @classmethod
    def isproduct(cls):
        """Return True, the type is a data product."""
        return True

    def name(self):
        """Unique name of the datatype"""
        sclass = type(self).__name__
        return f"{sclass}"

    def extract_db_info(self, obj, keys):
        """Extract metadata from serialized file"""
        import astropy.io.fits as fits

        result = {}
        if isinstance(obj, dict):
            try:
                qc = obj["quality_control"]
            except KeyError:
                qc = QC.UNKNOWN
        elif isinstance(obj, DataFrame):
            with obj.open() as hdulist:
                qc = self.datamodel.get_quality_control(hdulist)
        elif isinstance(obj, fits.HDUList):
            qc = self.datamodel.get_quality_control(obj)
        else:
            qc = QC.UNKNOWN

        result["quality_control"] = qc

        other = super().extract_db_info(obj, keys)
        result.update(other)
        return result

    def update_meta_info(self):
        """Return the metadata of the base class, with the quality control."""
        result = super().update_meta_info()
        result["quality_control"] = self.quality_control.name
        return result

    def __getstate__(self):
        """Return the state of the base class, with the quality control."""
        st = {}
        st["quality_control"] = self.quality_control

        other = super().__getstate__()
        st.update(other)
        return st

    def __setstate__(self, state):
        """Restore the quality control (UNKNOWN if missing), then the state of the base class."""
        qcval = state.get("quality_control", "UNKNOWN")
        self.quality_control = convert_qc(qcval)

        super().__setstate__(state)


class DataProductType(DataProductMixin, DataType):
    """Data type of a data product.

    Parameters
    ----------
    ptype : type
        Python type of the values.
    default : optional
        Default value.
    """

    def __init__(self, ptype, default=None):
        super().__init__(ptype, default=default)


if __name__ == "__main__":

    import numina.util.namespace as nm
    import numina.core.tagexpr as tagexpr

    class TagsNamespace(nm.Namespace):

        @staticmethod
        def p_(name):
            return tagexpr.Placeholder(name)

    class Tagged:

        def __init__(self, tags_ids):
            self.tag_ids = tags_ids
            tg = {key: tagexpr.TagRepr(key) for key in self.tag_ids}
            self.tags = TagsNamespace(**tg)
            self.tags_dict = tg
