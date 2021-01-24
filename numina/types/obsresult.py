# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


from numina.core.oresult import ObservationResult
# import numina.core.instrument.insconf as insconf

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
        return lambda mod, obj: True


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
        return validator(self, obj)
