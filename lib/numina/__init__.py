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

'''The Numen GTC recipe runner'''

import inspect
import logging
import sys
import pkgutil

import pkg_resources

from recipes import RecipeBase, RecipeResult

# pylint: disable-msg=E0611
try:
    from logging import NullHandler
except ImportError:
    from logger import NullHandler


__version__ = "0.2.1"

# Top level NullHandler
logging.getLogger("numina").addHandler(NullHandler())
    
def get_module(name):
    '''
    Get the module object from the module name.
    
    :param name: name of the module
    :rtype: module object
    
    This recipe has been extracted from the python
    documentation, from the reference of the `__import__`_
    function. 
    
    It should work this way:
    
      >>> m = get_module('emir.recipes.dark_image')
      >>> m.__name__
      'emir.recipes.dark_image'
      
    Other approach that seems to work also is:
    
      >>> m = __import__('emir.recipes.dark_image', globals(), locals(), [""], -1)
      >>> m.__name__
      'emir.recipes.dark_image'
    
    .. _`__import__`: http://docs.python.org/library/functions.html?highlight=import#__import__
    
    '''
    __import__(name)
    module_object = sys.modules[name]
    return module_object

def list_recipes(path):
    '''List all the recipes in a module'''
    module = __import__(path, fromlist="dummy")
    result = []
    for _importer, modname, _ispkg in pkgutil.iter_modules(module.__path__):
        rmodule = __import__('.'.join([path, modname]), fromlist="dummy")
        if check_recipe(rmodule):
            result.append((modname, rmodule.__doc__.splitlines()[0]))
    return result

def check_recipe(module):
    '''Check if a module has the Recipe API.'''
    
    def has_class(module, name, BaseClass):
        if hasattr(module, name):
            cls = getattr(module, name)
            if inspect.isclass(cls) and issubclass(cls, BaseClass) and not cls is BaseClass:
                return True
        return False
    
    if (has_class(module, 'Recipe', RecipeBase) and 
        has_class(module, 'Result', RecipeResult) and
        hasattr(module, '__doc__')):
        return True
    
    return False


def load_pkg_recipes():
    '''Return a dictionary of capabilities and recipes, using setuptools mechanism.
    ''' 
    ENTRY_POINT = 'numina.recipes'
    env = pkg_resources.Environment()
    recipes = {}
    for name in env:
        egg = env[name][0]
        for tname in egg.get_entry_map(ENTRY_POINT):
            ep = egg.get_entry_info(ENTRY_POINT, tname)
            mod = ep.load()
            if not hasattr(mod, 'capabilities'):
                mod.capabilities = ['recipe']
            for c in mod.capabilities:
                recipes.setdefault(c, []).append(mod)
    return recipes
