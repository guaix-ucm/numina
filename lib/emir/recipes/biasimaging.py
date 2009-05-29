#
# Copyright 2008-2009 Sergio Pascual
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

'''Bias image recipe and associates'''

from numina.recipes import RecipeBase
from numina.recipes import RecipeResult as RR
#from numina.exceptions import RecipeError

__version__ = "$Revision$"


class RecipeResult(RR):
    '''Result of the recipe.'''
    def __init__(self, image, filename):
        super(RecipeResult, self).__init__()
        self.hdulist = image
        self.filename = filename
        
    def store(self):
        pass


class Recipe(RecipeBase):
    '''Recipe to process data taken in Bias image Mode.
    
    Here starts the long description...
    It continues several lines
    
    '''
    def __init__(self):
        super(Recipe, self).__init__()
        
    def _process(self):
        pass       