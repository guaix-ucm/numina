#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

 * The class must derive from :class:`numina.core.BaseRecipe`.

'''


import abc
import traceback
import logging

from numina import __version__
from numina.exceptions import RecipeError
from .reciperesult import ErrorRecipeResult, RecipeResult

_logger = logging.getLogger('numina')

def list_recipes():
    '''List all defined recipes'''
    return BaseRecipe.__subclasses__() # pylint: disable-msgs=E1101
    
class BaseRecipe(object):
    '''Base class for all instrument recipes'''

    __metaclass__ = abc.ABCMeta
    
    __requires__ = {}
    __provides__ = {}

    # Recipe own logger
    logger = _logger

    def __init__(self, *args, **kwds):
        super(BaseRecipe, self).__init__()
        self.__author__ = 'Unknown'
        self.__version__ = '0.0.0'
        # These two are maintained
        # for the moment
        self.environ = {}
        self.runinfo = {}
        #
        self.instrument = None
        self.configure(**kwds)
    
    def configure(self, **kwds):
        if 'author' in kwds:
            self.__author__ = kwds['author']
        if 'version' in kwds:
            self.__version__ = kwds['version']
        if 'instrument' in kwds:
            self.instrument = kwds['instrument']
        if 'runinfo' in kwds:
            self.runinfo = kwds['runinfo']
        

    @abc.abstractmethod
    def run(self, observation_result, requirements):
        return

    def __call__(self, observation_result, requirements, environ=None):
        '''
        Process ``observation_result`` with the Recipe.
        
        Process the result of the observing block with the
        Recipe.
        
        :param observation_result: the result of a observing block
        :param type: ObservationResult
        :param requirements: requirements of the Recipe
        :param type: RecipeRequirement
        :param environ: a dictionary with custom parameters
        :rtype: a RecipeResult object or an error 
        
        '''

        if environ is not None:
            self.environ.update(environ)

        try:
            result = self.run(observation_result, requirements)
        except Exception as exc:
            _logger.error("During recipe execution %s", exc)
            return ErrorRecipeResult(exc.__class__.__name__, 
                                     str(exc),
                                     traceback.format_exc())

        
        return result
