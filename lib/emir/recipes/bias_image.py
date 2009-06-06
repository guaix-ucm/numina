#
# Copyright 2008-2009 Sergio Pascual, Nicolas Cardiel
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$

'''Bias image recipe and associates.

Recipe to process bias images. Bias images only appear in Simple Readout mode
(check this).

**Inputs:**

 * A list of bias images
 * A model of the detector (gain, RN)

**Outputs:**

 * A combined bias frame, with variance extension and quality flag. 

**Procedure:**

The list of images can be readly processed by combining them with a typical
sigma-clipping algoritm.

'''

from numina.recipes import RecipeBase
from numina.recipes import RecipeResult
#from numina.exceptions import RecipeError

__version__ = "$Revision$"


class Result(RecipeResult):
    '''Result of the recipe.'''
    def __init__(self):
        super(Result, self).__init__()
        
    def store(self):
        '''Description of store.
        
        :rtype: None'''
        pass


class Recipe(RecipeBase):
    '''Recipe to process data taken in Bias image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        
    def process(self):
        return Result() 
