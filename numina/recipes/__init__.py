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

 * The class must derive from :class:`numina.recipes.RecipeBase`.

'''


import abc
import traceback
import logging

from numina.exceptions import RecipeError

from .products import DataFrame, DataProduct 
from .oblock import obsres_from_dict
from .requirements import Parameter

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

    @abc.abstractmethod
    def run(self, block):
        return

    def __call__(self, block, environ=None):
        '''
        Process ``block`` with the Recipe.
        
        Process the result of the observing block with the
        Recipe.
        
        :param block: the result of a observing block
        :param type: ObservingResult
        :param environ: a dictionary with custom parameters
        :rtype: a dictionary with the result of the processing or an error 
        
        '''

        if environ is not None:
            self.environ.update(environ)

        self.environ['block_id'] = block.id

        try:

            result = self.run(block)

        except Exception as exc:
            result = {'error': {'type': exc.__class__.__name__, 
                                'message': str(exc), 
                                'traceback': traceback.format_exc()}
                      }
            _logger.error("During recipe execution %s", exc)
        return result

class RecipeResult(dict):
    '''Result of the __call__ method of the Recipe.'''
    pass

# FIXME: check if this class can be removed
class ReductionResult(object):
    def __init__(self):
        self.id = None
        self.reduction_block = None
        self.other = None
        self.status = 0
        self.picklable = {}

class provides(object):
    '''Decorator to add the list of provided products to recipe'''
    def __init__(self, *products):
        self.products = products

    def __call__(self, klass):
        if hasattr(klass, '__provides__'):
            klass.__provides__.extend(self.products)
        else:
            klass.__provides__ = list(self.products)
        return klass
    
class requires(object):
    '''Decorator to add the list of required parameters to recipe'''
    def __init__(self, *parameters):
        self.parameters = parameters

    def __call__(self, klass):
        if hasattr(klass, '__requires__'):
            klass.__requires__.extend(self.parameters)
        else:
            klass.__requires__ = list(self.parameters)
        return klass



