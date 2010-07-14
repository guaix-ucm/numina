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
import warnings
import inspect
import pkgutil

from numina.diskstorage import store
from numina.exceptions import RecipeError, ParameterError

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Abstract Base class for Recipes.'''
    
    required_parameters = []
    
    capabilities = []
    
    def __init__(self, param):
        self.values = param
        self.repeat = 1
                
    def setup(self, _param):
        warnings.warn("the setup method is deprecated", DeprecationWarning, stacklevel=2)
      
    def cleanup(self):
        '''Cleanup structures after recipe execution.'''
        pass
    
    def __call__(self):
        '''Run the recipe, don't override.'''
        try:
            self.repeat -= 1
            result = self.run()
            return result            
        except RecipeError:
            raise
        
#    @abc.abstractmethod
    def run(self):
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

def init_recipe_system(modules):
    for mod in modules:
        module = __import__(mod, fromlist="dummy")
        for _importer, modname, _ispkg in pkgutil.iter_modules(module.__path__):
            __import__('.'.join([mod, modname]), fromlist="dummy")
            
def list_recipes():
    return RecipeBase.__subclasses__()
    
def list_recipes_by_obs_mode(obsmode):
    return [rclass for rclass in list_recipes() 
            if obsmode in rclass.capabilities]
    
    
