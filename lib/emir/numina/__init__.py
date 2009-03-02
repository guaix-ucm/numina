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

from optparse import OptionParser
from ConfigParser import SafeConfigParser

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Base class for Recipes of all kinds'''
    def __init__(self, optusage = None):
        if optusage is None:
            optusage = "usage: %prog [options] recipe [recipe-options]" 
        self.cmdoptions = OptionParser(usage = optusage)
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
        return self._repeat <= 0
    @property
    def repeat(self):
        '''Number of times the recipe has to be repeated yet'''
        return self._repeat