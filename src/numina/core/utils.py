#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Recipes for system checks. """

import logging

from numina.core import BaseRecipe, DataFrameType
from numina.core.requirements import ObservationResultRequirement
import numina.core.dataholders as dh
import numina.core.query as qry


_logger = logging.getLogger(__name__)


class AlwaysFailRecipe(BaseRecipe):
    """A Recipe that always fails."""

    def __init__(self, *args, **kwargs):
        super(AlwaysFailRecipe, self).__init__(
            version="1"
        )

    def run(self, requirements):
        raise TypeError('This Recipe always fails')


class AlwaysSuccessRecipe(BaseRecipe):
    """A Recipe that always successes."""

    def __init__(self, *args, **kwargs):
        super(AlwaysSuccessRecipe, self).__init__(
            version=1
        )

    def run(self, recipe_input):
        return self.create_result()


class OBSuccessRecipe(BaseRecipe):
    """A Recipe that always successes, it requires an OB"""

    obresult = ObservationResultRequirement()

    def __init__(self, *args, **kwargs):
        super(OBSuccessRecipe, self).__init__(
            version=1
        )

    def run(self, recipe_input):
        return self.create_result()


class Combine(BaseRecipe):

    obresult = ObservationResultRequirement()
    method = dh.Parameter('mean', "Method of combination")
    method_kwargs = dh.Parameter({}, "Arguments passed to the combination method")
    field = dh.Parameter('image', "Extract field of previous result")
    result = dh.Result(DataFrameType)

    def run(self, recipe_input):
        import numina.array.combine as c
        from numina.processing.combine import combine_frames

        method = getattr(c, recipe_input.method)
        obresult = recipe_input.obresult
        method_kwargs = recipe_input.method_kwargs
        result = combine_frames(obresult.frames,
                                method, method_kwargs=method_kwargs
                                )
        return self.create_result(result=result)

    def build_recipe_input(self, obsres, dal):
        import numina.exceptions

        result = {}
        result['obresult'] = obsres
        for key in ['method', 'field']:
            req = self.requirements()[key]
            query_option = self.query_options.get(key)
            try:
                result[key] = req.query(dal, obsres, options=query_option)
            except numina.exceptions.NoResultFound as notfound:
                req.on_query_not_found(notfound)
        if 'field' in result:
            # extract values:
            obreq = self.requirements()['obresult']
            qoptions = qry.ResultOf(field=result['field'])

            obsres = obreq.query(dal, obsres, options=qoptions)
            result['obresult'] = obsres

        rinput = self.create_input(**result)
        return rinput
