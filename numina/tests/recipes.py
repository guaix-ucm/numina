#
# Copyright 2016-2017 Universidad Complutense de Madrid
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


class MasterBias(prod.DataProductMixin, frame.DataFrameType):
    pass


class MasterDark(prod.DataProductMixin, frame.DataFrameType):
    pass


class BiasRecipe(numina.core.BaseRecipe):
    master_bias = numina.core.Result(MasterBias)

    def __init__(self, *args, **kwds):
        super(BiasRecipe, self).__init__(*args, **kwds)
        self.simulate_error = kwds.get('simulate_error', False)


class DarkRecipe(numina.core.BaseRecipe):
    master_dark = numina.core.Result(MasterDark)
