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

# $Id: __init__.py 401 2009-02-27 19:02:07Z spr $

'''Utilities used by Numina'''


from optparse import OptionParser
from ConfigParser import SafeConfigParser
import inspect

__version__ = "$Id$"

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Base class for Recipes of all kinds'''
    def __init__(self, optusage=None):
        if optusage is None:
            optusage = "usage: %prog [options] recipe [recipe-options]" 
        self.cmdoptions = OptionParser(usage = optusage)
        self.cmdoptions.add_option('--docs', action="store_true", dest="docs", 
                                   default=False, help="prints documentation")
        self.iniconfig = SafeConfigParser()
        self._repeat = 1
        
    def setup(self):
        '''Initialize structures only once before recipe execution'''
        pass
    
    def run(self):
        '''Run the recipe, don't override'''
        result = self.process()
        self._repeat -= 1
        return result
    
    def process(self):
        ''' Override this method with custom code'''
        pass
    
    def complete(self):
        '''True once the recipe is completed'''
        return self._repeat <= 0
    
    @property
    def repeat(self):
        '''Number of times the recipe has to be repeated yet'''
        return self._repeat
      
      
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
    def __getattr__(self, mname):
        "Ignore attribute requests."
        return self

    def __setattr__(self, name, value):
        "Ignore attribute setting."
        return self

    def __delattr__(self, name):
        "Ignore deleting attributes."
        return self

    # misc.
    def __repr__(self):
        "Return a string representation."
        return "<Null>"

    def __str__(self):
        "Convert to a string and return it."
        return "Null"
    
    
def class_loader(path, default_module, logger=Null()):
    '''Loads a class from path'''
    comps = path.split('.')
    recipe = comps[-1]
    logger.debug('recipe is %s',recipe)
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
    '''Lists all the recipes in a module'''
    comps = path.split('.')
    module = __import__(path)
    # Import submodules
    for part in path.split('.')[1:]:
        module = getattr(module, part)
    
    # Check members of the module
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj) and issubclass(obj, RecipeBase) and not obj is RecipeBase:
            docs = obj.__doc__
            if docs is not None and docs:
                mydoc = docs.splitlines()[0]
            else:
                mydoc = ''
            print name, mydoc
