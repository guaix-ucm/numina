#
# Copyright 2008-2011 Sergio Pascual
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

'''Basic tools and classes used to generate recipe modules.

A recipe is a class that complies with the *reduction recipe API*:

 * The class must derive from :class:`numina.recipes.RecipeBase`.

'''

import abc
import importlib
import warnings
import inspect
import pkgutil
import datetime

from numina.exceptions import RecipeError, ParameterError

# Classes are new style
__metaclass__ = type

ERROR = 1
OK = 0
UNKNOWN = -1

_level_names = {ERROR: 'ERROR',
                OK: 'OK',
                UNKNOWN: 'UNKNOWN'}

def find_recipe(instrument, mode):
    base = '%s.recipes' % instrument
    try:
        mod = importlib.import_module(base)
    except ImportError:
        msg = 'No instrument %s' % instrument
        raise ValueError(msg)

    try:
        entry = mod.find_recipe(mode)
    except KeyError:
        msg = 'No recipe for mode %s' % mode
        raise ValueError(msg)
        
    return '%s.%s' % (base, entry)

def find_parameters(recipe_name):
    # query somewhere for the precomputed parameters
    return {}

class RecipeBase(object):
    '''Base class for all instrument recipes'''

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwds):
        super(RecipeBase, self).__init__()
        self.__author__ = 'Unknown'
        self.__version__ = '0.0.0'
        self.environ = {}
        self.parameters = {}
        self.instrument = None

        self.configure(**kwds)
    
    def configure(self, **kwds):
        if 'author' in kwds:
            self.__author__ = kwds['author']
        if 'version' in kwds:
            self.__version__ = kwds['version']
        if 'parameters' in kwds:
            self.parameters = kwds['parameters']
        if 'instrument' in kwds:
            self.instrument = kwds['instrument']
        if 'runinfo' in kwds:
            self.runinfo = kwds['runinfo']

    @abc.abstractmethod
    def run(self, block):
        return

    def __call__(self, block, environ=None):

        if environ is not None:
            self.environ.update(environ)

        self.environ['block_id'] = block.id

        try:

            result = self.run(block)

        except Exception as exc:
            result = {'error': {'type': exc.__class__.__name__, 
                                'message': str(exc)}}

        return result

class RecipeResult(dict):
    '''Result of the __call__ method of the Recipe.'''
    pass

class RecipeType(object):
    def __init__(self, tag, comment='', default=None):
        self.tag = tag
        self.comment = comment
        self.default = default

class Keyword(RecipeType):
    def __init__(self, tag, comment='', default=None):
        RecipeType.__init__(self, tag, comment, default)        

class Image(RecipeType):
    def __init__(self, tag, comment='', default=None):
        RecipeType.__init__(self, tag, comment, default)        

class Map(RecipeType):
    def __init__(self, tag, comment='', default=None):
        RecipeType.__init__(self, tag, comment, default)        

def list_recipes():
    '''List all defined recipes'''
    return RecipeBase.__subclasses__() # pylint: disable-msgs=E1101
    
def walk_modules(mod):
    module = importlib.import_module(mod)
    for _, nmod, _ in pkgutil.walk_packages(path=module.__path__,
                                    prefix=module.__name__ + '.'):
        yield nmod
        
def init_recipe_system(modules):
    '''Load all recipe classes in modules'''
    for mod in modules:
        for sub in walk_modules(mod):
            mod = importlib.import_module(sub)
        
if __name__ == '__main__':
    import json
    import tempfile
    
    from numina.user import main
    from numina.jsonserializer import to_json 
        
    p = {'niterations': 1, 'observing_mode': 'sample'}
    
    f = tempfile.NamedTemporaryFile(prefix='tmp', suffix='numina', delete=False)
    try:
        json.dump(p, f, default=to_json, encoding='utf-8', indent=2)
    finally:
        f.close()
            
    main(['--module', 'numina.recipes', '--run', f.name])
