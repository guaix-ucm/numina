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

import os
import os.path
import ConfigParser
import importlib
import pkgutil
import logging

from numina.config import pipeline_path

_logger = logging.getLogger('numina')

_pipelines = {}

class Pipeline(object):
    def __init__(self, name, version, recipes):
        self.name = name
        self.version = version
        self.recipes = recipes

    def get_recipe(self, mode):
        return self.recipes[mode]

def get_pipeline(name):
    return _pipelines[name]

def register_pipeline(pipe):
    global _pipelines

    _pipelines[pipe.name] = pipe

def register_recipes(name, recipes):
    pipe = Pipeline(name, recipes)
    register_pipeline(pipe)

def load_pipelines_from(paths):
    '''Load pipelines from ini files in paths.'''
    for path in paths:
        load_pipelines_from_ini(path)
    return _pipelines

def import_pipeline(name, path):
    '''Import a pipeline module from a given path.'''
    if not path:
        _logger.debug('Import pipeline %s from system', name)
        try:
            mod = importlib.import_module(name)
            return mod
        except ImportError:
            _logger.warning('No module named %s', name)
    else:
        _logger.debug('Import pipeline %s from %s', name, path)
        for impt, mname, isp in pkgutil.iter_modules([path]):
            if mname == name:
                loader = impt.find_module(mname)
                mod = loader.load_module(mname)
                return mod
        else:
            _logger.warning('No module named %s', name)
            
def load_pipelines_from_ini(path):
    '''Load files in ini format from path'''
    
    for base, sub, files in os.walk(path):
        files.sort()
        for fname in files:
            defaults = {'name': fname, 'path': None}
            config = ConfigParser.SafeConfigParser(defaults=defaults,
                                                   allow_no_value=True)
            config.read(os.path.join(base, fname))
            try:
                name = config.get('pipeline', 'name')
                path = config.get('pipeline', 'path')
                path = os.path.expanduser(path)
                import_pipeline(name, path)
            except ConfigParser.NoSectionError:
                _logger.warning('Not valid ini file', fname)
    
def init_pipeline_system():
    '''Load all available pipelines.'''
    paths = pipeline_path()
    return load_pipelines_from(paths)

def get_recipe(name, mode):
    '''Find the Recipe suited to process a given observing mode.''' 
    try:
        drp = _pipelines[name]
    except KeyError:
        msg = 'No pipeline named %s' % name
        raise ValueError(msg)
    
    try:
        klass = pipe.get_recipe(mode)
    except KeyError:
        msg = 'No recipe for mode %s' % mode
        raise ValueError(msg)
        
    return klass

