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

 * The class must derive from :class:`numina.core.RecipeBase`.

'''


import abc
import traceback
import logging

from numina.exceptions import RecipeError
from .reciperesult import ErrorRecipeResult

_logger = logging.getLogger('numina')

def list_recipes():
    '''List all defined recipes'''
    return RecipeBase.__subclasses__() # pylint: disable-msgs=E1101
    
class RecipeBase(object):
    '''Base class for all instrument recipes'''

    __metaclass__ = abc.ABCMeta
    
    # Recipe own logger
    logger = _logger

    def __init__(self, *args, **kwds):
        super(RecipeBase, self).__init__()
        self.__author__ = 'Unknown'
        self.__version__ = '0.0.0'
        self.environ = {}
        self.parameters = {}
        self.instrument = None
        self.runinfo = {}

        self.configure(**kwds)
    
    def configure(self, **kwds):
        if 'author' in kwds:
            self.__author__ = kwds['author']
        if 'version' in kwds:
            self.__version__ = kwds['version']
        if 'parameters' in kwds:
            self.parameters = kwds['parameters']
        if 'instrument' in kwds:
            self.instrument = kwds['instrument']
        if 'runinfo' in kwds:
            self.runinfo = kwds['runinfo']
        if 'requirements' in kwds:
            self.requirements = kwds['requirements']

    @abc.abstractmethod
    def run(self, block):
        return

    def __call__(self, block, environ=None):
        '''
        Process ``block`` with the Recipe.
        
        Process the result of the observing block with the
        Recipe.
        
        :param block: the result of a observing block
        :param type: ObservationResult
        :param environ: a dictionary with custom parameters
        :rtype: a dictionary with the result of the processing or an error 
        
        '''

        if environ is not None:
            self.environ.update(environ)

        try:
            result = self.run(block)
        except Exception as exc:
            _logger.error("During recipe execution %s", exc)
            return ErrorRecipeResult(exc.__class__.__name__, 
                                     str(exc),
                                     traceback.format_exc())

        if isinstance(val, RecipeResult):
            return val
        else:
            return self.convert(val)

    @classmethod
    def convert(cls, value):
        '''Convert from a dictionary to a RecipeResult'''
        if 'error' in value:
            err = value['error']
            if _valid_err(err):
                return ErrorRecipeResult(err['type'], err['message'], err['traceback'])
            else:
                raise ValueError('malformed value to convert')
        elif 'products' in value:
            prod = value['products']
            products = {'product%d' % i: prod for i, prod in enumerate(prod)}
            return cls.RecipeResult(**products)
        else:
            raise ValueError('malformed value to convert')


def _valid_err(err):
    if not isinstance(err, dict):
        return False
    if not 'type' in err:
        return False
    if not 'message' in err:
        return False
    if not 'traceback' in err:
        return False
    return True
