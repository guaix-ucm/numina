#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""
Basic tools and classes used to generate recipe modules.

A recipe is a class that complies with the *reduction recipe API*:

 * The class must derive from :class:`numina.core.BaseRecipe`.

"""


import logging
import json
import functools

from astropy.io import fits

from numina.util.jsonencoder import ExtEncoder
from numina.datamodel import DataModel
from .. import __version__
from .recipeinout import RecipeResult as RecipeResultClass
from .recipeinout import RecipeInput as RecipeInputClass
from .metarecipes import RecipeType
from .oresult import ObservationResult, ObservingBlock
from ..exceptions import NoResultFound


class BaseRecipe(metaclass=RecipeType):
    """Base class for all instrument recipes

    Parameters
    ----------
    intermediate_results : bool, optional
                           If True, save intermediate results of the Recipe


    Attributes
    ----------

    obresult : ObservationResult, requirement

    logger :
         recipe logger

    """
    RecipeResult = RecipeResultClass
    RecipeInput = RecipeInputClass
    datamodel = DataModel()
    # Recipe own logger
    logger = logging.getLogger('numina.recipes.numina')

    def __new__(cls, *args, **kwargs):
        recipe = super(BaseRecipe, cls).__new__(cls)
        recipe.instrument = kwargs.get('instrument', 'UNKNOWN')
        recipe.mode = kwargs.get('mode', 'UNKNOWN')
        recipe.pipeline = kwargs.get('pipeline', 'default')
        recipe.intermediate_results = kwargs.get('intermediate_results', False)
        recipe.validate_inputs = kwargs.get('validate_inputs', False)
        recipe.validate_results = kwargs.get('validate_results', False)
        recipe.runinfo = cls.create_default_runinfo()
        recipe.runinfo.update(kwargs.get('runinfo', {}))
        recipe.environ = {}
        recipe.__version__ = 1
        recipe.query_options = kwargs.get('query_options', {})
        recipe.configure(**kwargs)
        return recipe

    def __init__(self, *args, **kwargs):
        super(BaseRecipe, self).__init__()
        self.configure(**kwargs)

    def configure(self, **kwds):
        if 'version' in kwds:
            self.__version__ = kwds['version']

        base_kwds = ['instrument', 'mode', 'runinfo', 'intermediate_results',
                     'validate_inputs']
        for kwd in base_kwds:
            if kwd in kwds:
                setattr(self, kwd, kwds[kwd])

    def __setstate__(self, state):
        self.configure(**state)

    @staticmethod
    def create_default_runinfo():
        runinfo = {
            'data_dir': None,
            'results_dir': None,
            'work_dir': None,
            'pipeline': 'default',
            'runner': 'unknown-runner',
            'runner_version': 'unknown-version',
            'taskid': 'unknown-id'
        }
        return runinfo

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

    @classmethod
    def tag_names(cls):
        return cls.RecipeInput.tag_names()

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

        # Optional input validation
        if self.validate_inputs:
            self.validate_input(recipe_input)

        result = self.run(recipe_input)

        # Update QC in the result
        self.run_qc(recipe_input, result)

        if self.validate_results:
            self.validate_result(result)

        return result

    def validate_input(self, recipe_input):
        """"Validate the input of the recipe"""
        self.logger.debug('start validating input')
        recipe_input.validate()

    def validate_result(self, recipe_result):
        """Validate the result of the recipe"""
        self.logger.debug('start validating result')
        recipe_result.validate()

    def save_intermediate_img(self, img, name):
        """Save intermediate FITS objects."""
        if self.intermediate_results:
            img.writeto(name, overwrite=True)

    def save_intermediate_array(self, array, name):
        """Save intermediate array object as FITS."""
        if self.intermediate_results:
            fits.writeto(name, array, overwrite=True)

    def save_structured_as_json(self, structured, name):
        if self.intermediate_results:
            if hasattr(structured, '__getstate__'):
                state = structured.__getstate__()
            elif isinstance(structured, dict):
                state = structured
            else:
                state = structured.__dict__

            with open(name, 'w') as fd:
                json.dump(state, fd, indent=2, cls=ExtEncoder)

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
        ob_query_skip = False
        ob_query_field = 'obresult'

        if isinstance(ob, ObservingBlock):
            import numina.types.obsresult as obtype
            # We have to build an Obsres
            for key, req in self.requirements().items():
                if isinstance(req.type, obtype.ObservationResultType):
                    ob_query_field = key
                    ob_query_skip = True
                    query_option = self.query_options.get(key)

                    # print('req for ob is named', key, query_option)
                    new_or = ObservationResult()
                    new_or.__dict__ = ob.__dict__
                    obsres = req.query(dal, new_or, options=query_option)
                    break
            else:
                # nothing to do
                obsres = ob
        else:
            obsres = ob

        qfields = self.tag_names()
        self.logger.debug('running recipe tagger with query fields: %s', qfields)
        if qfields:
            obsres.tags = self.extract_tags_from_obsres(obsres, qfields)
        else:
            obsres.tags = {}
        self.logger.debug('obsres tags are: %s', obsres.tags)

        for key, req in self.requirements().items():

            try:
                query_option = self.query_options.get(key)
                if key == ob_query_field and ob_query_skip:
                    result[key] = obsres
                else:
                    result[key] = req.query(dal, obsres, options=query_option)
            except NoResultFound as notfound:
                req.on_query_not_found(notfound)

        return self.create_input(**result)

    def extract_tags_from_obsres(self, obsres, tag_keys):
        ref_img = obsres.get_sample_frame().open()
        final_tags = self.extract_tags_from_ref(ref_img, tag_keys, base=obsres.labels)
        return final_tags

    def extract_tags_from_ref(self, ref, tag_keys, base=None):

        base = base or {}
        fits_extractor = self.datamodel.extractor_map['fits']
        final_tags = {}
        for key in tag_keys:

            if key in base:
                final_tags[key] = base[key]
            else:
                final_tags[key]= fits_extractor.extract(key, ref)
        return final_tags


def timeit(method):
    """Decorator to measure the time used by the recipe"""

    import datetime

    @functools.wraps(method)
    def timed_method(self, rinput):

        time_start = datetime.datetime.utcnow()
        result = method(self, rinput)
        time_end = datetime.datetime.utcnow()
        result.time_it(time_start, time_end)
        self.logger.info('total time measured')
        return result

    return timed_method
