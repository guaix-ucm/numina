# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#


import warnings

from numina.util.convert import convert_qc
from numina.exceptions import NoResultFound

from .base import DataTypeBase
from .datatype import DataType
from .dataframe import DataFrame
from .qc import QC


class DataProductMixin(DataTypeBase):
    """A type that is a data product."""

    def __init__(self, *args, **kwds):
        super(DataProductMixin, self).__init__(*args, **kwds)
        self.quality_control = QC.UNKNOWN

    def generators(self):
        return []

    @classmethod
    def isproduct(cls):
        return True

    def name(self):
        """Unique name of the datatype"""
        sclass = type(self).__name__
        return "%s" % (sclass,)

    def query(self, name, dal, ob, options=None):

        try:
            return self.query_on_ob(name, ob)
        except NoResultFound:
            pass

        # if not, the normal query
        prod = dal.search_product(name, self, ob)

        return prod.content

    def query_constraints(self):
        import numina.core.query
        return numina.core.query.Constraint()

    def extract_db_info(self, obj, keys):
        """Extract metadata from serialized file"""
        result = {}
        if isinstance(obj, dict):
            try:
                qc = obj['quality_control']
            except KeyError:
                qc = QC.UNKNOWN
        elif isinstance(obj, DataFrame):
            with obj.open() as hdulist:
                qc = self.datamodel.get_quality_control(hdulist)
        else:
            qc = QC.UNKNOWN

        result['quality_control'] = qc
        other = super(DataProductMixin, self).extract_db_info(obj, keys)
        result.update(other)
        return result

    def update_meta_info(self):
        result = super(DataProductMixin, self).update_meta_info()
        result['quality_control'] = self.quality_control.name
        return result

    def __getstate__(self):
        st = {}
        st['quality_control'] = self.quality_control

        other = super(DataProductMixin, self).__getstate__()
        st.update(other)
        return st

    def __setstate__(self, state):
        qcval = state.get('quality_control', 'UNKNOWN')
        self.quality_control = convert_qc(qcval)

        super(DataProductMixin, self).__setstate__(state)


class DataProductTag(DataProductMixin):
    def __init__(self, *args, **kwargs):
        warnings.warn("The 'DataProductTag' class was renamed to 'DataProductMixin'",
                      DeprecationWarning)
        super(DataProductTag, self).__init__(*args, **kwargs)


class DataProductType(DataProductMixin, DataType):
    def __init__(self, ptype, default=None):
        super(DataProductType, self).__init__(ptype, default=default)


class ConfigurationTag(object):
    """A type that is part of the instrument configuration."""

    @classmethod
    def isconfiguration(cls):
        return True


if __name__ == '__main__':

    import numina.util.namespace as nm
    import numina.core.tagexpr as tagexpr


    class TagsNamespace(nm.Namespace):

        @staticmethod
        def p_(name):
            return tagexpr.Placeholder(name)


    class Tagged(object):

        def __init__(self, tags_ids):
            self.tag_ids = tags_ids
            tg = {key: tagexpr.TagRepr(key) for key in self.tag_ids}
            self.tags = TagsNamespace(**tg)
            self.tags_dict = tg
