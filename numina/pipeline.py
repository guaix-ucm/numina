#
# Copyright 2011-2012 Universidad Complutense de Madrid
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

'''Pipeline loader.'''

import os.path
import ConfigParser
import importlib
import pkgutil
import logging
import fnmatch

from numina.config import pipeline_path

_logger = logging.getLogger('numina')

_pipelines = {}

_FILE_EXTENSION = 'ini'
_FILE_PATTERN = '*.%s' % _FILE_EXTENSION

class Pipeline(object):
    '''A pipeline.'''
    def __init__(self, name, version, recipes):
        self.name = name
        self.version = version
        self.recipes = recipes

    def get_recipe(self, mode):
        return self.recipes[mode]

def get_pipeline(name):
    '''Get a pipeline from the global register.
    
    :param name: Name of the pipeline
    :raises: ValueError if the named pipeline does not exist
    '''
    return _pipelines[name]

def get_recipe(name, mode):
    '''Find the Recipe suited to process a given observing mode.''' 
    try:
        pipe = _pipelines[name]
    except KeyError:
        msg = 'No pipeline named %s' % name
        raise ValueError(msg)
    
    try:
        klass = pipe.get_recipe(mode)
    except KeyError:
        msg = 'No recipe for mode %s' % mode
        raise ValueError(msg)
        
    return klass

def register_pipeline(pipe):
    '''Register a pipeline in the global register.
    
    :param pipe: a Pipeline instance
    '''
    
    global _pipelines

    _pipelines[pipe.name] = pipe

def register_recipes(name, recipes):
    '''Register a group of recipes.
    
    :param name: name of the pipeline
    :param recipes: a dictionary with recipe classes
    
    '''
    pipe = Pipeline(name, recipes)
    register_pipeline(pipe)


# pkgutil.iter_modules([path]):
def init_pipeline_system():
    '''Load all available pipelines.'''
    import numina.pipelines as master
    import pkgutil

    #for i in pkgutil.iter_modules(master.__path__):
    for i in pkgutil.walk_packages(master.__path__, master.__name__ + '.'):
        imp, name, is_pkg = i
        loader = imp.find_module(name)
        mod = loader.load_module(name)
    return _pipelines
