#
# Copyright 2008-2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Recipes for system checks. """

from numina.core import BaseRecipe
from numina.core.requirements import ObservationResultRequirement


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
