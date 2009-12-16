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

'''Basic tools and classes used to generate recipe modules.

A recipe is a module that complies with the *reduction recipe API*:

 * It must provide a `Recipe` class that derives from :class:`numina.recipes.RecipeBase`.

'''

__version__ = "$Revision$"

from numina.exceptions import RecipeError, ParameterError

# Classes are new style
__metaclass__ = type

class RecipeType(type):
    registry = {}
    def __init__(cls, name, bases, dictionary):
        """Initialise the new class-object"""
        if name != 'RecipeBase':
            cls.registry[cls] = name

class RecipeBase:
    '''Abstract Base class for Recipes.'''
#    __metaclass__ = abc.ABCMeta
    __metaclass__ = RecipeType
    
    def __init__(self, parameters):
        #self.iniconfig = SafeConfigParser()
        self.parameters = parameters
        self._repeat = 1
        
    def setup(self):
        '''Initialize structures only once before recipe execution.'''
        pass
      
    def cleanup(self):
        '''Cleanup structures after recipe execution.'''
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
    def __init__(self, qa):
        self.qa = qa

class ParametersDescription:
    def __init__(self, inputs, outputs, optional, pipeline, systemwide):
        self.inputs = inputs
        self.outputs = outputs
        self.optional = optional
        self.pipeline = pipeline
        self.systemwide = systemwide
        
    def complete(self, obj):
        for key in self.inputs:
            if key not in obj.inputs:
                raise ParameterError('error in inputs')
        for key in self.outputs:
            if key not in obj.outputs:
                raise ParameterError('error in outputs')
        
        new_optional = dict(self.optional)
        new_optional.update(obj.optional)
        
        new_pipeline = dict(self.pipeline)
        new_pipeline.update(obj.pipeline)

        new_systemwide = dict(self.systemwide)
        new_systemwide.update(obj.systemwide)
        
        return Parameters(obj.inputs, obj.outputs, new_optional, new_pipeline, new_systemwide)

class Parameters:
    def __init__(self, inputs, outputs, optional, pipeline, systemwide):
        self.inputs = inputs
        self.outputs = outputs
        self.optional = optional
        self.pipeline = pipeline
        self.systemwide = systemwide


_systemwide_parameters = {'compute_qa': True}

def systemwide_parameters():
    return _systemwide_parameters

