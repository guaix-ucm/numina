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

'''The Numen GTC recipe runner'''


from optparse import OptionParser
from ConfigParser import SafeConfigParser
import inspect
import logging
import sys

from .recipes import RecipeBase
from tests import tests as all_tests

# pylint: disable-msg=E0611
try:
    from logging import NullHandler
except ImportError:
    from logger import NullHandler

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

# Top level NullHandler
logging.getLogger("numina").addHandler(NullHandler())


      
class Null:
    '''Idempotent class'''
    def __init__(self, *args, **kwargs):
        "Ignore parameters."
        return None

    # object calling
    def __call__(self, *args, **kwargs):
        "Ignore method calls."
        return self

    # attribute handling
    def __getattr__(self, mame):
        "Ignore attribute requests."
        return self

    def __setattr__(self, name, value):
        "Ignore attribute setting."
        return self

    def __delattr__(self, name):
        "Ignore deleting attributes."
        return self

    # misc.
    @staticmethod
    def __repr__():
        "Return a string representation."
        return "<Null>"
    
    @staticmethod
    def __str__():
        "Convert to a string and return it."
        return "Null"
    
def get_module(name):
    '''
    Get the module object from the module name.
    
    :param name: name of the module
    :rtype: module object
    
    This recipe has been extracted from the python
    documentation, from the reference of the `__import__`_
    function. 
    
    It should work this way:
    
      >>> m = get_module('emir.recipes.darkimaging')
      >>> m.__name__
      'emir.recipes.darkimaging'
      
    Other approach that seems to work also is:
    
      >>> m = __import__('emir.recipes.darkimaging', globals(), locals(), [""], -1)
      >>> m.__name__
      'emir.recipes.darkimaging'
    
    .. _`__import__`: http://docs.python.org/library/functions.html?highlight=import#__import__
    
    '''
    __import__(name)
    module_object = sys.modules[name]
    return module_object

  
    
def load_class(path, default_module, logger=Null()):
    '''Load a class from path'''
    comps = path.split('.')
    recipe = comps[-1]
    logger.debug('recipe is %s', recipe)
    if len(comps) == 1:
        modulen = default_module
    else:
        modulen = '.'.join(comps[0:-1])
    logger.debug('module is %s', modulen)

    module = __import__(modulen)
    
    for part in modulen.split('.')[1:]:
        module = getattr(module, part)
        try:
            RecipeClass = getattr(module, recipe)
        except AttributeError:
            logger.error("% s doesn't exist in %s", recipe, modulen)
            raise
        
    if inspect.isclass(RecipeClass) and \
       issubclass(RecipeClass, RecipeBase) and \
       not RecipeClass is RecipeBase:
        return RecipeClass
    # In other case
    return None


def list_recipes(path, docs=True):
    '''List all the recipes in a module'''
    module = __import__(path)
    # Import submodules
    for part in path.split('.')[1:]:
        module = getattr(module, part)
    
    # Check members of the module
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj) and issubclass(obj, RecipeBase) and not obj is RecipeBase:
            docs = obj.__doc__
            # pylint: disable-msg=E1103
            if docs:
                mydoc = docs.splitlines()[0]
            else:
                mydoc = ''
            print name, mydoc

