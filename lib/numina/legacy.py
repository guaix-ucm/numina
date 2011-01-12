#
# Copyright 2008-2011 Sergio Pascual
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

'''Legacy functions.

The functions and classes in this module are not used anymore,
but are maintained here.
'''

import sys

import pkg_resources
    
def get_module(name):
    '''
    Get the module object from the module name.
    
    :param name: name of the module
    :rtype: module object
    
    This recipe has been extracted from the python
    documentation, from the reference of the `__import__`_
    function. 
    
    It should work this way:
    
      >>> m = get_module('emir.recipes.darkimage')
      >>> m.__name__
      'emir.recipes.darkimage'
      
    Other approach that seems to work also is:
    
      >>> m = __import__('emir.recipes.darkimage', globals(), locals(), [""], -1)
      >>> m.__name__
      'emir.recipes.darkimage'
    
    .. _`__import__`: http://docs.python.org/library/functions.html?highlight=import#__import__
    
    '''
    __import__(name)
    module_object = sys.modules[name]
    return module_object

def load_pkg_recipes():
    '''Return a dictionary of capabilities and recipes, using setuptools mechanism.''' 
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
