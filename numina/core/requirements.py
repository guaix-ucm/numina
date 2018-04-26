#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Recipe requirement holders
"""


import numina.types.obsresult as obtypes

from .dataholders import Requirement


class ObservationResultRequirement(Requirement):
    """The Recipe requires the result of an observation."""
    def __init__(self, query_opts=None):

        super(ObservationResultRequirement, self).__init__(
            obtypes.ObservationResultType, "Observation Result",
            query_opts=query_opts
            )

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(dest=%r, description='%s')"
        msg = fmt % (sclass, self.dest, self.description)
        return msg


class InstrumentConfigurationRequirement(Requirement):
    """The Recipe requires the configuration of the instrument."""
    def __init__(self):

        super(InstrumentConfigurationRequirement, self).__init__(
            obtypes.InstrumentConfigurationType,
            "Instrument Configuration",
            validation=False
            )

    def __repr__(self):
        sclass = type(self).__name__
        fmt = "%s(dest=%r, description='%s')"
        msg = fmt % (sclass, self.dest, self.description)
        return msg
