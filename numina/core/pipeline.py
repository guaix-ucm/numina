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
import importlib

import numina.pipelines as namespace

_logger = logging.getLogger('numina')

_pipelines = {}
_instruments = {}

def import_object(path):
    spl = path.split('.')
    cls = spl[-1]
    mods = '.'.join(spl[:-1])
    mm = importlib.import_module(mods)
    Cls = getattr(mm, cls)
    return Cls

class BasePipeline(object):
    '''Base class for pipelines.'''
    def __init__(self, name, version, recipes):
        self.name = name
        self.version = version
        self.recipes = recipes

    def get_recipe(self, mode):
        return self.recipes[mode]

class InstrumentConfiguration(object):
    def __init__(self, name, values):
        self.version = name
        self.configuration = values

class BaseInstrument(object):
    name = 'Undefined'
    modes = []
    configurations = {'default': None}

    def __init__(self, name, config_name='default'):
        self.name = name
        self.config_name = config_name
        self.config_version = '0.0.0'

        cup = self.configurations.get(self.config_name)
        if cup is not None:
            self.config_version = cup.version
            for key, val in cup.configuration.items():
                # ignore the following
                if key == 'instrument':
                    if val != name:
                        raise ValueError('instrument name in file and assigned differ')
                elif key == 'name':
                    if val != config_name:
                        raise ValueError('config name in file and assigned differ')
                elif key == 'pipeline':
                    pass #ignore
                else:
                    setattr(self, key, val)
        else:
            raise ValueError('no configuration named %s', self.config_name)
        
        
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
        self.recipe_class = None
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

def init_pipeline_system():
    '''Load all available pipelines.'''
    
    for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__, namespace.__name__ + '.'):
        try:
            loader = imp.find_module(name)
            _mod = loader.load_module(name)
        except StandardError as error:
            _logger.warning('Problem importing %s, error of type %s with message "%s"', name, type(error), error)
        
    # Loaded all DRP modules
    
    # populate instruments list
    for InsCls in BaseInstrument.__subclasses__():
        ins = InsCls()
        global _instruments
        _instruments[ins.name] = ins 
        
        
    for PipeCls in BasePipeline.__subclasses__():
        pipe = PipeCls()
        global _pipelines
        _pipelines[pipe.name] = pipe
        
    return _instruments, _pipelines


# Quick constructor and representer
# for Observing Modes in YAML 

import yaml
import uuid

def om_repr(dumper, data):
    return dumper.represent_mapping('!om', data.__dict__)

def om_cons(loader, node):
    om = ObservingMode()
    value = loader.construct_mapping(node)
    om.__dict__ = value
    om.uuid = uuid.UUID(om.uuid)
    return om



yaml.add_representer(ObservingMode, om_repr)
yaml.add_constructor('!om', om_cons)

