#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Recipe requirement holders
"""


import numina.exceptions
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

    def query(self, dal, obsres, options=None):
        """

        Parameters
        ----------
        dal
        obsres
        options

        Returns
        -------

        """
        from numina.core.query import ResultOf

        q_options = self.query_options(options)

        if isinstance(q_options, ResultOf):
            dest_field = 'frames'
            dest_type = list
            # Field to insert the results
            if not hasattr(obsres, dest_field):
                setattr(obsres, dest_field, dest_type())

            dest_obj = getattr(obsres, dest_field)

            values = dal.search_result_relative(self.dest, self.type, obsres,
                                                result_desc=q_options)

            for partial in values:
                dest_obj.append(partial.content)

        return obsres

    def on_query_not_found_from_type(self, notfound):
        """

        Parameters
        ----------
        notfound

        Returns
        -------

        """
        raise numina.exceptions.NoResultFound('unable to complete ObservationResult') from notfound
