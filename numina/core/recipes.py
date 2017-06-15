#
# Copyright 2008-2017 Universidad Complutense de Madrid
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
from .recipeinout import RecipeResult as RecipeResultClass
from .recipeinout import RecipeInput as RecipeInputClass
from .metarecipes import RecipeType
from .oresult import ObservationResult
from ..dal.stored import ObservingBlock
from ..exceptions import NoResultFound


class BaseRecipe(with_metaclass(RecipeType, object)):
    """Base class for all instrument recipes"""

    RecipeResult = RecipeResultClass
    RecipeInput = RecipeInputClass
    # Recipe own logger
    logger = logging.getLogger('numina.recipes.numina')

    def __new__(cls, *args, **kwargs):
        recipe = super(BaseRecipe, cls).__new__(cls)
        recipe.instrument = kwargs.get('instrument', 'UNKNOWN')
        recipe.mode = kwargs.get('mode', 'UNKNOWN')
        recipe.intermediate_results = kwargs.get('intermediate_results', False)
        recipe.runinfo = kwargs.get('runinfo', {})
        recipe.environ = {}
        recipe.__version__ = 1
        recipe.query_options = {}
        recipe.configure(**kwargs)
        return recipe

    def __init__(self, *args, **kwargs):
        super(BaseRecipe, self).__init__()
        self.configure(**kwargs)

    def configure(self, **kwds):
        if 'version' in kwds:
            self.__version__ = kwds['version']

        base_kwds = ['instrument', 'mode', 'runinfo', 'intermediate_results']
        for kwd in base_kwds:
            if kwd in kwds:
                setattr(self, kwd, kwds[kwd])

    def __setstate__(self, state):
        self.configure(**state)

    @classmethod
    def create_input(cls, *args, **kwds):
        """Pass the result arguments to the RecipeInput constructor"""

        return cls.RecipeInput(*args, **kwds)

    @classmethod
    def create_result(cls, *args, **kwds):
        """Pass the result arguments to the RecipeResult constructor"""

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
        """Run Quality Control checks."""
        return recipe_result

    def __call__(self, recipe_input):
        """
        Process the result of the observing block with the Recipe.

        Parameters
        ----------
        recipe_input : RecipeInput
                       The input appropriated for the Recipe

        Returns
        -------
        a RecipeResult object or an error
        """

        result = self.run(recipe_input)

        # Update QC in the result
        self.run_qc(recipe_input, result)

        return result

    def validate_input(self, recipe_input):
        """"Validate the input of the recipe"""
        recipe_input.validate()

    def validate_result(self, recipe_result):
        """Validate the result of the recipe"""
        recipe_result.validate()

    def set_base_headers(self, hdr):
        """Set metadata in FITS headers."""
        hdr['NUMXVER'] = (__version__, 'Numina package version')
        hdr['NUMRNAM'] = (self.__class__.__name__, 'Numina recipe name')
        hdr['NUMRVER'] = (self.__version__, 'Numina recipe version')
        return hdr

    def build_recipe_input(self, ob, dal):
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

        for key, req in self.requirements().items():

            try:
                query_option = self.query_options.get(key)
                result[key] = req.query(dal, obsres, options=query_option)
            except NoResultFound as notfound:
                req.on_query_not_found(notfound)

        return self.create_input(**result)
