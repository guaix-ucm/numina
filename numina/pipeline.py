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

import pkgutil
import importlib

from numina.config import pipeline_path

def load_pipelines_from(paths):
    '''Load all recipe classes in modules'''
    for _, name, _ in pkgutil.iter_modules(paths):
        yield name

def pipelines():
    paths = pipeline_path()
    return load_pipelines_from(paths)
        
def load_pipelines_from(paths):
    '''Load all recipe classes in modules'''
    pipelines = {}
    for impt, name, isp in pkgutil.iter_modules(paths):
        loader = impt.find_module(name)
        mod = loader.load_module(name)
        pipelines[name] = mod

    for key in pipelines:
        # import recipes
        # import everything under 'name.recipes'
        recipes = importlib.import_module('%s.recipes' % key)
        for impt, nmod, ispkg in pkgutil.walk_packages(path=recipes.__path__,
                                            prefix=recipes.__name__ + '.'):
            loader = impt.find_module(nmod)
            try:
                mod = loader.load_module(nmod)
            except ImportError as ex:
                print 'error loading', nmod

    return pipelines

def init_pipeline_system():
    paths = pipeline_path()
    return load_pipelines_from(paths)
