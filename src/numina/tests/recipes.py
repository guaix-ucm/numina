#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Recipes for testing"""

import numina.core
import numina.types.product as prod
import numina.types.frame as frame
import numina.core.requirements as reqs


class MasterBias(prod.DataProductMixin, frame.DataFrameType):
    pass


class MasterDark(prod.DataProductMixin, frame.DataFrameType):
    pass


class ImageTest(frame.DataFrameType):
    pass


class BiasRecipe(numina.core.BaseRecipe):
    master_bias = numina.core.Result(MasterBias)

    def __init__(self, *args, **kwds):
        super(BiasRecipe, self).__init__(*args, **kwds)
        self.simulate_error = kwds.get('simulate_error', False)


class DarkRecipe(numina.core.BaseRecipe):
    master_dark = numina.core.Result(MasterDark)


class ImageRecipe(numina.core.BaseRecipe):
    obresult = reqs.ObservationResultRequirement()
    master_bias = numina.core.Requirement(MasterBias, "Master Bias")
    master_dark = numina.core.Requirement(MasterDark, "Master Dark")
    result_image = numina.core.Result(ImageTest)


class ImageRecipeCom(numina.core.BaseRecipe):
    obresult = reqs.ObservationResultRequirement()
    accum_in = numina.core.Requirement(ImageTest, 'previous accum')
    result_image = numina.core.Result(ImageTest)
    accum = numina.core.Result(ImageTest)
