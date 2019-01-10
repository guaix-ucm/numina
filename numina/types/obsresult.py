# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import warnings

import six
from numina.exceptions import NoResultFound
from numina.core.oresult import ObservationResult
from numina.core.insconf import InstrumentConfiguration
from numina.types.qc import QC
from .frame import DataFrameType
from .datatype import DataType


def _obtain_validator_for(instrument, mode_key):
    import numina.drps
    drps = numina.drps.get_system_drps()

    lol = drps.query_by_name(instrument)

    mode = lol.modes[mode_key]
    if mode.validator:
        return mode.validator
    else:
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

    def query(self, name, dal, obsres, options=None):
        from numina.core.query import ResultOf
        if isinstance(options, ResultOf):
            dest_field = 'frames'
            dest_type = list
            # Field to insert the results
            if not hasattr(obsres, dest_field):
                setattr(obsres, dest_field, dest_type())

            dest_obj = getattr(obsres, dest_field)

            values = dal.search_result_relative(name, self, obsres,
                                                result_desc=options)

            for partial in values:
                dest_obj.append(partial.content)

        return obsres

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
