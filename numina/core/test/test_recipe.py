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

'''Unit test for RecipeBase.'''

from ..recipes import RecipeType
from ..recipeinout import RecipeRequirements, RecipeResult
from ..requirements import ObservationResultRequirement
from ..dataholders import Product

def test_metaclass_empty_base():
    
    class TestRecipe(object):
        __metaclass__ = RecipeType

    assert hasattr(TestRecipe, 'RecipeRequirements')
    
    assert hasattr(TestRecipe, 'RecipeResult')
    
    assert issubclass(TestRecipe.RecipeRequirements, RecipeRequirements)
    
    assert issubclass(TestRecipe.RecipeResult, RecipeResult)
    
    assert TestRecipe.RecipeRequirements.__name__ == 'RecipeRequirements'
    
    assert TestRecipe.RecipeResult.__name__ == 'RecipeResult'
    
def test_metaclass():

    class TestRecipe(object):
        __metaclass__ = RecipeType
        
        obsresult = ObservationResultRequirement()
        someresult = Product(int, 'Some integer')

    assert hasattr(TestRecipe, 'RecipeRequirements')
    
    assert hasattr(TestRecipe, 'RecipeResult')
    
    assert issubclass(TestRecipe.RecipeRequirements, RecipeRequirements)
    
    assert issubclass(TestRecipe.RecipeResult, RecipeResult)
    
    assert TestRecipe.RecipeRequirements.__name__ == 'TestRecipeRequirements'
    
    assert TestRecipe.RecipeResult.__name__ == 'TestRecipeResult'

