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

import six
from numina.exceptions import NoResultFound
from numina.core.oresult import ObservationResult
from numina.core.query import Result
from numina.core.pipeline import InstrumentConfiguration
from numina.types.qc import QC
from .frame import DataFrameType
from .datatype import DataType


def _obtain_validator_for(instrument, mode_key):
    import numina.drps
    drps = numina.drps.get_system_drps()

    lol = drps.query_by_name(instrument)

    for mode in lol.modes:
        if mode.key == mode_key:
            if mode.validator:
                return mode.validator
            else:
                break

    return lambda obj: True


class ObservationResultType(DataType):
    """The type of ObservationResult."""

    def __init__(self, rawtype=None):
        super(ObservationResultType, self).__init__(ptype=ObservationResult)
        if rawtype:
            self.rawtype = rawtype
        else:
            self.rawtype = DataFrameType

    def validate(self, obj):
        # super(ObservationResultType, self).validate(obj)
        validator = _obtain_validator_for(obj.instrument, obj.mode)
        return validator(obj)

    def query(self, name, dal, ob, options=None):

        # Complete values with previous Results
        if isinstance(options, Result):
            tipo = DataFrameType()
            mode = options.mode
            field = options.field
            node = options.node
            query_opts = {}
            query_opts['ignore_fail'] = options.ignore_fail
            val = dal.search_result_relative(name, tipo, ob, mode, field, node, options=query_opts)
            ob.results = []
            for r in val:
                ob.results.append(r.content)

        return ob

    def on_query_not_found(self, notfound):
        six.raise_from(NoResultFound('unable to complete ObservationResult'), notfound)


class InstrumentConfigurationType(DataType):
    """The type of InstrumentConfiguration."""

    def __init__(self):
        super(InstrumentConfigurationType, self).__init__(
            ptype=InstrumentConfiguration
            )

    def validate(self, obj):
        return True

    def query(self, name, dal, ob, options=None):
        if not isinstance(ob.configuration, InstrumentConfiguration):
            warnings.warn(RuntimeWarning, 'instrument configuration not configured')
            return {}
        else:
            return ob.configuration

    def on_query_not_found(self, notfound):
        raise notfound


class QualityControlProduct(DataType):
    def __init__(self):
        super(QualityControlProduct, self).__init__(
            ptype=QC,
            default=QC.UNKNOWN
            )
