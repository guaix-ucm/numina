# Copyright 2008-2017 Universidad Complutense de Madrid
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


from itertools import chain

from numina.util.convert import convert_qc
from numina.exceptions import NoResultFound

from .base import DataTypeBase
from .datatype import DataType
from .dataframe import DataFrame
from .qc import QC


class DataProductTag(DataTypeBase):
    """A type that is a data product."""

    def __init__(self, *args, **kwds):
        super(DataProductTag, self).__init__(*args, **kwds)
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

        # If ob declares a particular ID, check that
        for g in chain([self.name()], self.generators()):
            if g in ob.results:
                resultid = ob.results[g]
                prod = dal.search_result(name, self, ob, resultid)
                break
        else:
            # if not, the normal query
            prod = dal.search_product(name, self, ob)

        return prod.content

    def extract_meta_info(self, obj):
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
        other = super(DataProductTag, self).extract_meta_info(obj)
        result.update(other)
        return result

    def __getstate__(self):
        st = {}
        st['quality_control'] = self.quality_control

        other = super(DataProductTag, self).__getstate__()
        st.update(other)
        return st

    def __setstate__(self, state):
        qcval = state['quality_control']
        self.quality_control = convert_qc(qcval)

        super(DataProductTag, self).__setstate__(state)


class DataProductType(DataProductTag, DataType):
    def __init__(self, ptype, default=None):
        super(DataProductType, self).__init__(ptype, default=default)


class ConfigurationTag(object):
    """A type that is part of the instrument configuration."""

    @classmethod
    def isconfiguration(cls):
        return True
