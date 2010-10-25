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

A recipe is a class that complies with the *reduction recipe API*:

 * The class must derive from :class:`numina.recipes.RecipeBase`.

'''
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


class RecipeBase:
    '''Abstract Base class for Recipes.'''
    
    required_parameters = []
    
    capabilities = []
    
    __version__ = 1
        
    @classmethod
    def info(cls):
        return dict(name=cls.__name__, module=cls.__module__, version=cls.__version__)
        
    def __init__(self, param, run):
        self.parameters = param
        self.runinfo = run
        self.repeat = run.get('repeat', 1)
        self._current = 0
                
    def setup(self):
        pass
      
    def cleanup(self):
        '''Cleanup structures after recipe execution.'''
        pass
    
    def __call__(self):
        '''Run the recipe, don't override.'''
        
        TIMEFMT = '%FT%T'
        
        for self._current in range(self.repeat):
            try:
                product = RecipeResult(recipe=self.info(), 
                               run=dict(start=None,
                                        end=None,
                                        status=UNKNOWN,
                                        error={},
                                        repeat=self.repeat,
                                        current=self._current + 1),
                               result={}
                               )
                
                run_info = dict(repeat=self.repeat, 
                                current=self._current + 1)
                
                now1 = datetime.datetime.now()
                run_info['start'] = now1.strftime(TIMEFMT)
                
                product['result'] = self.run()
                
                now2 = datetime.datetime.now()                
                run_info['end'] = now2.strftime(TIMEFMT)
                # Status 0 means all correct
                run_info['status'] = OK
                         
            except (IOError, OSError, RecipeError), e:
                now2 = datetime.datetime.now()                
                run_info['end'] = now2.strftime(TIMEFMT)
                # Status 0 means all correct
                run_info['status'] = ERROR                
                run_info['error'] = dict(type=e.__class__.__name__, 
                                         message=str(e))
            finally:
                product['run'].update(self.runinfo)
                product['run'].update(run_info)
                
            
            yield product

    def run(self):
        ''' Override this method with custom code.
        
        :rtype: RecipeResult
        '''
        raise NotImplementedError
    
    def complete(self):
        '''Return True once the recipe is completed.
        
        :rtype: bool
        '''
        return self.repeat <= 0
    
    @property
    def current(self):
        return self._current
        
class RecipeResult(dict):
    '''Result of the __call__ method of the Recipe.'''
    pass
            
def list_recipes():
    return RecipeBase.__subclasses__()
    
def list_recipes_by_obs_mode(obsmode):
    return list(recipes_by_obs_mode(obsmode))
    
def recipes_by_obs_mode(obsmode):
    for rclass in list_recipes():
        if obsmode in rclass.capabilities:
            yield rclass
    
def walk_modules(mod):
    module = __import__(mod, fromlist="dummy")
    for _, nmod, _ in pkgutil.walk_packages(path=module.__path__,
                                                prefix=module.__name__ + '.'):
        yield nmod
        
def init_recipe_system(modules):
    for mod in modules:
        for sub in walk_modules(mod):
            __import__(sub, fromlist="dummy")
            
            
            
if __name__ == '__main__':
    import simplejson as json
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
