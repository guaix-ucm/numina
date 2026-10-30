#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Recipes for testing"""

import warnings
import numina.core
import numina.types.product as prod
import numina.types.frame as frame
import numina.core.requirements as reqs

warnings.warn(
    "numina.testing.recipes is deprecated and will be removed in the next version of numina, "
    "it is only used by the tests of numina",
    DeprecationWarning,
    stacklevel=2,
)


class MasterBias(prod.DataProductMixin, frame.DataFrameType):
    """Master bias product for testing."""

    pass


class MasterDark(prod.DataProductMixin, frame.DataFrameType):
    """Master dark product for testing."""

    pass


class ImageTest(frame.DataFrameType):
    """Image type for testing."""

    pass


class BiasRecipe(numina.core.BaseRecipe):
    """Recipe for testing that produces a master bias."""

    master_bias = numina.core.Result(MasterBias)

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.simulate_error = kwds.get("simulate_error", False)


class DarkRecipe(numina.core.BaseRecipe):
    """Recipe for testing that produces a master dark."""

    master_dark = numina.core.Result(MasterDark)


class ImageRecipe(numina.core.BaseRecipe):
    """Recipe for testing that requires a master bias and a master dark."""

    obresult = reqs.ObservationResultRequirement()
    master_bias = numina.core.Requirement(MasterBias, "Master Bias")
    master_dark = numina.core.Requirement(MasterDark, "Master Dark")
    result_image = numina.core.Result(ImageTest)


class ImageRecipeCom(numina.core.BaseRecipe):
    """Recipe for testing that requires a previous accumulated image and produces an image and a new accumulated one."""

    obresult = reqs.ObservationResultRequirement()
    accum_in = numina.core.Requirement(ImageTest, "previous accum")
    result_image = numina.core.Result(ImageTest)
    accum = numina.core.Result(ImageTest)
