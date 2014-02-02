#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''Recipes for system checks.  '''

from .recipes import BaseRecipe

from numina.core import BaseRecipe, RecipeRequirements, DataFrame
from numina.core import Requirement, Product, DataProductRequirement
from numina.core import define_requirements, define_result
from numina.core.requirements import ObservationResultRequirement



class AlwaysFailRecipe(BaseRecipe):
    '''A Recipe that always fails.'''

    def __init__(self):
        super(AlwaysFailRecipe, self).__init__(
            author="Sergio Pascual <sergiopr@fis.ucm.es>",
            version="0.1.0"
        )

    def run(self, requirements):
        raise TypeError('This Recipe always fails')

class AlwaysSuccessRecipe(BaseRecipe):
    '''A Recipe that always successes.'''

    def __init__(self):
        super(AlwaysSuccessRecipe, self).__init__(
            author="Sergio Pascual <sergiopr@fis.ucm.es>",
            version="0.1.0"
        )

    def run(self, requirements):
        return self.RecipeResult()

class OBSuccessRecipeRequirements(RecipeRequirements):
    obresult = ObservationResultRequirement()

@define_requirements(OBSuccessRecipeRequirements)
class OBSuccessRecipe(BaseRecipe):
    '''A Recipe that always successes, it requires an OB'''

    def __init__(self):
        super(OBSuccessRecipe, self).__init__(
            author="Sergio Pascual <sergiopr@fis.ucm.es>",
            version="0.1.0"
        )

    def run(self, requirements):
        return self.RecipeResult()



