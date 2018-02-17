#
# Copyright 2008-2015 Universidad Complutense de Madrid
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
