#
# Copyright 2016-2017 Universidad Complutense de Madrid
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


"""Recipes for testing"""

import numina.core
import numina.types.product as prod
import numina.types.frame as frame

class MasterBias(prod.DataProductTag, frame.DataFrameType):
    pass


class MasterDark(prod.DataProductTag, frame.DataFrameType):
    pass


class BiasRecipe(numina.core.BaseRecipe):
    master_bias = numina.core.Product(MasterBias)

    def __init__(self, *args, **kwds):
        super(BiasRecipe, self).__init__(*args, **kwds)
        self.simulate_error = kwds.get('simulate_error', False)


class DarkRecipe(numina.core.BaseRecipe):
    master_dark = numina.core.Product(MasterDark)
