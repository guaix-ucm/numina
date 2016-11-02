#
# Copyright 2008-2016 Universidad Complutense de Madrid
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

"""
Basic tools and classes used to generate recipe modules.

A recipe is a class that complies with the *reduction recipe API*:

 * The class must derive from :class:`numina.core.BaseRecipe`.

"""

import traceback
import logging
import warnings

from six import with_metaclass

from .. import __version__
from .recipeinout import ErrorRecipeResult
from .recipeinout import RecipeResult as RecipeResultClass
from .recipeinout import RecipeInput as RecipeInputClass
from .metarecipes import RecipeType
from .oresult import ObservationResult
from ..dal.stored import ObservingBlock
from ..exceptions import NoResultFound
from .products import ObservationResultType
from .products import InstrumentConfigurationType
from .products import DataProductTag
from .pipeline import InstrumentConfiguration


class BaseRecipe(with_metaclass(RecipeType, object)):
    """Base class for all instrument recipes"""

    RecipeResult = RecipeResultClass
    RecipeInput = RecipeInputClass

    def __init__(self, *args, **kwds):
        super(BaseRecipe, self).__init__()
        self.__author__ = 'Unknown'
        self.__version__ = 1
        # These two are maintained
        # for the moment
        self.environ = {}
        self.runinfo = {}
        #
        self.instrument = None
        self.configure(**kwds)

        # Recipe own logger
        self.logger = logging.getLogger('numina')

    def configure(self, **kwds):
        if 'author' in kwds:
            self.__author__ = kwds['author']
        if 'version' in kwds:
            self.__version__ = kwds['version']
        if 'instrument' in kwds:
            self.instrument = kwds['instrument']
        if 'runinfo' in kwds:
            self.runinfo = kwds['runinfo']


    @classmethod
    def create_input(cls, *args, **kwds):
        '''
        Pass the result arguments to the RecipeInput constructor
        '''
        return cls.RecipeInput(*args, **kwds)

    @classmethod
    def create_result(cls, *args, **kwds):
        '''
        Pass the result arguments to the RecipeResult constructor
        '''
        return cls.RecipeResult(*args, **kwds)

    @classmethod
    def requirements(cls):
        return cls.RecipeInput.stored()

    @classmethod
    def products(cls):
        return cls.RecipeResult.stored()

    def run(self, recipe_input):
        return self.create_result()

    def run_qc(self, recipe_input, recipe_result):
        """Run Quality Control checks"""
        return recipe_result

    def __call__(self, recipe_input):
        """
        Process the result of the observing block with the
        Recipe.

        :param recipe_input: the input appropriated for the Recipe
        :param type: RecipeInput
        :rtype: a RecipeResult object or an error
        """

        result = self.run(recipe_input)

        # Update QC in the result
        self.run_qc(recipe_input, result)

        return result

    def validate_input(self, recipe_input):
        "Validate the input of the recipe"
        recipe_input.validate()

    def validate_result(self, recipe_result):
        "Validate the result of the recipe"
        recipe_result.validate()

    def set_base_headers(self, hdr):
        '''Set metadata in FITS headers.'''
        hdr['NUMXVER'] = (__version__, 'Numina package version')
        hdr['NUMRNAM'] = (self.__class__.__name__, 'Numina recipe name')
        hdr['NUMRVER'] = (self.__version__, 'Numina recipe version')
        return hdr

    @classmethod
    def build_recipe_input(cls, ob, dal, pipeline='default'):
        """Build a RecipeInput object."""

        result = {}
        # We have to decide if the ob input
        # is a plain description (ObservingBlock)
        # or if it contains the nested results (Obsres)
        #
        # it has to contain the tags corresponding to the observing modes...
        if isinstance(ob, ObservingBlock):
            # We have to build an Obsres
            obsres = dal.obsres_from_oblock_id(ob.id)
        elif isinstance(ob, ObservationResult):
            # We have one
            obsres = ob
        else:
            raise ValueError('ob input is neither a ObservingBlock'
                             ' nor a ObservationResult')

        tags = getattr(obsres, 'tags', {})

        for key, req in cls.requirements().items():

            # First check if the requirement is embedded
            # in the observation result
            # it can happen in GTC

            # Using NoResultFound instead of None
            # None can be a valid result
            val = getattr(obsres, key, NoResultFound)

            if val is not NoResultFound:
                result[key] = val
                continue

            # Then, continue checking the rest

            if isinstance(req.type, ObservationResultType):
                result[key] = obsres
            elif isinstance(req.type, InstrumentConfigurationType):
                if not isinstance(obsres.configuration, InstrumentConfiguration):
                    warnings.warn(RuntimeWarning, 'instrument configuration not configured')
                    result[key] = {}
                else:
                    result[key] = obsres.configuration

            elif isinstance(req.type, DataProductTag):
                try:
                    prod = dal.search_prod_req_tags(req, obsres.instrument,
                                                    tags, pipeline)
                    result[key] = prod.content
                except NoResultFound:
                    pass
            else:
                # Still not clear what to do with the other types
                try:
                    param = dal.search_param_req(req, obsres.instrument,
                                                 obsres.mode, pipeline)
                    result[key] = param.content
                except NoResultFound:
                    pass

        return cls.create_input(**result)

    # An alias required by GTC
    buildRI = build_recipe_input

