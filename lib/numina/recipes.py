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

'''Recipe Abstract Base class.'''


from optparse import OptionParser
from ConfigParser import SafeConfigParser
#import abc

from numina.exceptions import RecipeError

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

class RecipeBase:
    '''Abstract Base class for Recipes.'''
#    __metaclass__ = abc.ABCMeta
    def __init__(self, optusage=None):
        if optusage is None:
            optusage = "usage: %prog [options] recipe [recipe-options]" 
        self.cmdoptions = OptionParser(usage = optusage)
        self.cmdoptions.add_option('--docs', action="store_true", dest="docs", 
                                   default=False, help="prints documentation")
        self.iniconfig = SafeConfigParser()
        self._repeat = 1
        
    def setup(self):
        '''Initialize structures only once before recipe execution.'''
        pass
    
    def run(self):
        '''Run the recipe, don't override.'''
        try:
            self._repeat -= 1
            result = self.process()
            return result            
        except RecipeError:
            raise
        
#    @abc.abstractmethod
    def process(self):
        ''' Override this method with custom code.
        
        :rtype: RecipeResult
        '''
        raise NotImplementedError
    
    def complete(self):
        '''True once the recipe is completed.
        
        :rtype: bool
        '''
        return self._repeat <= 0
    
    @property
    def repeat(self):
        '''Number of times the recipe has to be repeated yet.
        
        :rtype: int
        '''
        return self._repeat
    
class RecipeResult:
    '''Result of the run method of the Recipe.'''
#    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass
    
#    @abc.abstractmethod
    def store(self):
        '''Store the result'''
        pass
