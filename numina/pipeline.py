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

import logging
import pkgutil

import numina.pipelines as namespace

_logger = logging.getLogger('numina')

_pipelines = {}
_instruments = {}

class Pipeline(object):
    '''A pipeline.'''
    def __init__(self, name, version, recipes):
        self.name = name
        self.version = version
        self.recipes = recipes

    def get_recipe(self, mode):
        return self.recipes[mode]

class BaseInstrument(object):
    name = 'Undefined'
    modes = []
        
class ObservingMode(object):
    def __init__(self):
        self.name = ''
        self.uuid = ''
        self.key = ''
        self.url = ''
        self.instrument = ''
        self.summary = ''
        self.description = ''
        self.recipe = ''
        self.status = ''
        self.date = ''
        self.reference = ''
        
def get_instruments():
    return _instruments

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


def init_pipeline_system():
    '''Load all available pipelines.'''
    
    for i in pkgutil.walk_packages(namespace.__path__, namespace.__name__ + '.'):
        imp, name, _is_pkg = i
        loader = imp.find_module(name)
        _mod = loader.load_module(name)
        
        
    # Loaded all DRP modules
    
    # populate instruments list
    for InsCls in BaseInstrument.__subclasses__():
        ins = InsCls()
        global _instruments
        _instruments[ins.name] = ins 
        
    return _pipelines


# Quick constructor and representer
# for Observing Modes in YAML 

import yaml

def om_repr(dumper, data):
    return dumper.represent_mapping(u'!om', data.__dict__)

def om_cons(loader, node):
    om = ObservingMode()
    value = loader.construct_mapping(node)
    om.__dict__ = value
    return om



yaml.add_representer(ObservingMode, om_repr)
yaml.add_constructor(u'!om', om_cons)

