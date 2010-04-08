#
# Copyright 2008-2010 Sergio Pascual
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

'''Basic tools and classes used to generate recipe modules.

A recipe is a module that complies with the *reduction recipe API*:

 * It must provide a `Recipe` class that derives from :class:`numina.recipes.RecipeBase`.

'''

from numina.diskstorage import store
from numina.exceptions import RecipeError, ParameterError

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Abstract Base class for Recipes.'''    
    def __init__(self):
        self.inputs = {}
        self.optional = {}
        self.repeat = 1
                
    def setup(self, param):
        '''Initialize structures only once before recipe execution.'''
        self.inputs = param.inputs
        self.optional = param.optional
      
    def cleanup(self):
        '''Cleanup structures after recipe execution.'''
        pass
    
    def run(self):
        '''Run the recipe, don't override.'''
        try:
            self.repeat -= 1
            result = self.process()
            return result            
        except RecipeError:
            raise
        
#    @abc.abstractmethod
    def process(self):
        ''' Override this method with custom code.
        
        :rtype: RecipeResult
        '''
        raise NotImplementedError
    
    def complete(self):
        '''True once the recipe is completed.
        
        :rtype: bool
        '''
        return self.repeat <= 0
    
class RecipeResult:
    '''Result of the run method of the Recipe.'''
#    __metaclass__ = abc.ABCMeta
    def __init__(self, qa):
        self.qa = qa
        self.products = {}

@store.register(RecipeResult)
def _store_rr(obj, where=None):
    # We store the values inside obj.products
    for key, val in obj.products.iteritems():
        store(val, key)

class ParameterDescription(object):
    def __init__(self, inputs, optional):
        self.inputs = inputs
        self.optional = optional
    
    def complete_group(self, obj, group):
        d = dict(getattr(self, group))
        d.update(getattr(obj, group))
        return d

    def complete(self, obj):        
        newvals = {}
        for group in ['inputs', 'optional']:            
            newvals[group] = self.complete_group(obj, group)

        return Parameters(**newvals)

class Parameters:
    def __init__(self, inputs, optional):
        self.inputs = inputs
        self.optional = optional
