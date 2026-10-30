#
# Copyright 2008-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""
Basic tools and classes used to generate recipe modules.

A recipe is a class that complies with the *reduction recipe API*:

 * The class must derive from :class:`numina.core.BaseRecipe`.

"""

import functools
import json
import logging

from astropy.io import fits

from numina.util.jsonencoder import ExtEncoder
from numina.datamodel import DataModel
from .. import __version__
from .dataholders import EntryHolder, Requirement, Result
from .recipeinout import RecipeResult as RecipeResultClass
from .recipeinout import RecipeInput as RecipeInputClass
from .recipeinout import generate_docs
from .oresult import ObservationResult
from ..exceptions import NoResultFound
from numina.core.taggers import extract_tags_from_obsres, extract_tags_from_img


class BaseRecipe:
    """Base class for all instrument recipes

    Parameters
    ----------
    intermediate_results : bool, optional
                           If True, save intermediate results of the Recipe


    Attributes
    ----------

    obresult : ObservationResult, requirement

    logger :
         recipe logger. A recipe that does not define it has the logger
         of the module of its class; a logger defined in a recipe is
         used by its subclasses

    """

    RecipeResult = RecipeResultClass
    RecipeInput = RecipeInputClass
    # Version of the recipe, in the header NUMRVER of the products.
    # A recipe can define its own, or receive it in the argument 'version'
    __version__ = 1
    datamodel = DataModel()
    # Recipe own logger
    logger = logging.getLogger("numina.recipes.numina")
    # The subclasses that do not define a logger have the logger of their module
    _numina_module_logger = True

    def __init_subclass__(cls, **kwargs):
        """Move the requirements and the results to the input and result classes.

        The requirements of the recipe become the fields of a new class
        RECIPENAMEInput, derived from the RecipeInput of the base recipe,
        and the results the fields of RECIPENAMEResult, derived from its
        RecipeResult. If there are none, the class of the base recipe is
        used. A RecipeInput or RecipeResult defined in the class is used
        as the base class.
        """
        super().__init_subclass__(**kwargs)

        # The logger defined in the class is used by its subclasses,
        # without it, the logger of the module of the class
        if "logger" in vars(cls):
            cls._numina_module_logger = False
        elif cls._numina_module_logger:
            cls.logger = logging.getLogger(cls.__module__)

        requirements = {}
        results = {}
        for name, value in list(vars(cls).items()):
            if isinstance(value, EntryHolder):
                if isinstance(value, Requirement):
                    requirements[name] = value
                if isinstance(value, Result):
                    results[name] = value
                delattr(cls, name)

        input_class = _create_io_class(cls, "Input", cls.RecipeInput, requirements)
        result_class = _create_io_class(cls, "Result", cls.RecipeResult, results)
        for klass in [input_class, result_class]:
            setattr(cls, klass.__name__, klass)
        cls.RecipeInput = input_class
        cls.RecipeResult = result_class

    def __new__(cls, *args, **kwargs):
        """Create the recipe and set its attributes from the keyword arguments."""
        # The attributes are set here, and not in __init__, so that they
        # are defined in recipes that do not pass the arguments to
        # super().__init__. __init__ sets them again with configure,
        # with the arguments it receives.
        recipe = super().__new__(cls)
        recipe.instrument = kwargs.get("instrument", "UNKNOWN")
        recipe.mode = kwargs.get("mode", "UNKNOWN")
        recipe.pipeline = kwargs.get("pipeline", "default")
        recipe.drp_version = "undefined"
        recipe.intermediate_results = kwargs.get("intermediate_results", False)
        recipe.validate_inputs = kwargs.get("validate_inputs", False)
        recipe.validate_results = kwargs.get("validate_results", False)
        recipe.runinfo = cls.create_default_runinfo()
        recipe.runinfo.update(kwargs.get("runinfo", {}))
        recipe.environ = {}
        if "version" in kwargs:
            recipe.__version__ = kwargs["version"]
        recipe.query_options = kwargs.get("query_options", {})
        return recipe

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.configure(**kwargs)

    def configure(self, **kwds):
        """Set attributes of the recipe from keyword arguments.

        The keywords used are ``version``, ``instrument``, ``mode``,
        ``runinfo``, ``intermediate_results`` and ``validate_inputs``,
        the others are ignored.
        """
        if "version" in kwds:
            self.__version__ = kwds["version"]

        base_kwds = ["instrument", "mode", "runinfo", "intermediate_results", "validate_inputs"]
        for kwd in base_kwds:
            if kwd in kwds:
                setattr(self, kwd, kwds[kwd])

    def __setstate__(self, state):
        """Configure the recipe with the dictionary `state`, see ``configure``."""
        self.configure(**state)

    @staticmethod
    def create_default_runinfo():
        """Return the default information of the run, a dict."""
        runinfo = {
            "data_dir": None,
            "results_dir": None,
            "work_dir": None,
            "pipeline": "default",
            "runner": "unknown-runner",
            "runner_version": "unknown-version",
            "taskid": "unknown-id",
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
        """Return the requirements of the recipe, by name."""
        return cls.RecipeInput.stored()

    @classmethod
    def products(cls):
        """Return the results of the recipe, by name."""
        return cls.RecipeResult.stored()

    @classmethod
    def tag_names(cls):
        """Return the set of the names of the tags of the requirements."""
        return cls.RecipeInput.tag_names()

    def run(self, recipe_input):
        """Process `recipe_input` and return the result, the recipes override it.

        This implementation returns a result without values.
        """
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
        """Validate the input of the recipe

        The values are validated by their types and by the checks of the
        input, see :meth:`numina.core.recipeinout.RecipeInOut.validate`.
        """
        self.logger.debug("start validating input")
        recipe_input.validate()

    def validate_result(self, recipe_result):
        """Validate the result of the recipe"""
        self.logger.debug("start validating result")
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
        """Save the state of `structured` in the JSON file `name`, if intermediate results are enabled."""
        if self.intermediate_results:
            # Since Python 3.11 all objects have __getstate__,
            # and the __getstate__ of a dict is None
            if isinstance(structured, dict):
                state = structured
            else:
                state = structured.__getstate__()

            with open(name, "w") as fd:
                json.dump(state, fd, indent=2, cls=ExtEncoder)

    def set_base_headers(self, hdr):
        """Set metadata in FITS headers.

        The keywords are:

        NUMXVER
            Version of numina.
        NUMRNAM
            Name of the class of the recipe.
        NUMRVER
            Version of the recipe, the attribute ``__version__`` of its
            class (1 by default) or the argument ``version``.
        NUMDRP
            Name of the DRP, the instrument.
        NUMDRPV
            Version of the DRP, by default the version of its package.
        NUMPIPE
            Name of the pipeline of the DRP.

        The DRP, its version and the pipeline are set when the recipe is
        created by the pipeline of a DRP. The decorator :func:`timeit`
        adds NUMUTC1 and NUMUTC2, the start and end of the run.
        """
        hdr["NUMXVER"] = (__version__, "Numina package version")
        hdr["NUMRNAM"] = (self.__class__.__name__, "Numina recipe name")
        hdr["NUMRVER"] = (self.__version__, "Numina recipe version")
        hdr["NUMDRP"] = (self.instrument, "Numina DRP name")
        hdr["NUMDRPV"] = (str(self.drp_version), "Numina DRP version")
        hdr["NUMPIPE"] = (self.pipeline, "Numina DRP pipeline")
        return hdr

    def build_recipe_input(self, ob, dal):
        """Build a RecipeInput object.

        The requirements are queried with `ob`, which is modified: a requirement
        of type ObservationResult adds to it the results of other OBs (ResultOf),
        and the tags are extracted from it. The caller uses these changes.
        """
        import numina.types.obsresult as obtype

        result = {}

        if isinstance(ob, ObservationResult):
            obsres = ob
        else:
            # A plain ObservingBlock: the ObservationResult shares its
            # attributes, so that the changes are visible in ob
            obsres = ObservationResult.__new__(ObservationResult)
            obsres.__dict__ = ob.__dict__

        # The requirement of the observation result is queried first,
        # as it can add frames used to extract the tags
        ob_query_field = None
        for key, req in self.requirements().items():
            if isinstance(req.type, obtype.ObservationResultType):
                ob_query_field = key
                obsres = req.query(dal, obsres, options=self.query_options.get(key))
                break

        qfields = self.tag_names()
        self.logger.debug("running recipe tagger with query fields: %s", qfields)
        if qfields:
            obsres.tags = extract_tags_from_obsres(obsres, qfields, self.datamodel, strict=True)
        else:
            obsres.tags = {}
        self.logger.debug("obsres tags are: %s", obsres.tags)

        for key, req in self.requirements().items():

            try:
                query_option = self.query_options.get(key)
                if key == ob_query_field:
                    result[key] = obsres
                else:
                    result[key] = req.query(dal, obsres, options=query_option)
            except NoResultFound as notfound:
                req.on_query_not_found(notfound)

        return self.create_input(**result)

    def extract_tags_from_ref(self, ref, tag_keys, base=None):
        """Return the tags `tag_keys` of the image `ref`, see ``extract_tags_from_img``."""
        return extract_tags_from_img(ref, tag_keys, self.datamodel, base=base)

    def get_filters(self, imgtypes, getters):
        """Select the getters that apply to the requirements of the recipe.

        Parameters
        ----------
        imgtypes : sequence
            For each element of `getters`, a type or None. The getter is
            used if the type of a requirement is an instance of the type,
            and always if None.
        getters : sequence
            Getters, or iterables of getters.

        Returns
        -------
        list
            The selected getters, the iterables are flattened.
        """
        import collections.abc

        used_getters = []
        for rtype, getter in zip(imgtypes, getters):
            self.logger.debug("get_filters, %s  %s", rtype, getter)
            if rtype is None:
                # Unconditional
                if isinstance(getter, collections.abc.Iterable):
                    used_getters.extend(getter)
                else:
                    used_getters.append(getter)
            else:
                # Search
                for key, val in self.RecipeInput.stored().items():
                    if isinstance(val.type, rtype):
                        if isinstance(getter, collections.abc.Iterable):
                            used_getters.extend(getter)
                        else:
                            used_getters.append(getter)
                        break
                else:
                    pass
        return used_getters

    def init_filters_generic(self, rinput, getters_seq, ins):
        """Build a ``SerialFlow`` of correctors for each sequence of getters.

        Each getter is called as ``getter(rinput, meta, ins, datamodel)``,
        where ``meta`` is the result of :meth:`gather_info`, and returns
        a corrector.
        """
        import numina.ext.gtc
        import numina.util.flow as flowmod

        # with BPM, bias, dark, flat and sky
        if numina.ext.gtc.check_gtc():
            self.logger.debug("running in GTC environment")
        else:
            self.logger.debug("running outside of GTC environment")

        meta = self.gather_info(rinput)
        self.logger.debug("obresult info")
        for entry in meta["obresult"]:
            self.logger.debug("frame info is %s", entry)

        reduction_flows = []
        for getters in getters_seq:
            correctors = [getter(rinput, meta, ins, self.datamodel) for getter in getters]
            reduction_flow = flowmod.SerialFlow(correctors)
            reduction_flows.append(reduction_flow)

        return reduction_flows

    def init_filters(self, rinput, ins=None):
        """Build the correction flows with the getters of :meth:`create_getters`.

        By default, `ins` is the configuration of the observation result.
        """
        # Defined to be overloaded
        if ins is None:
            ins = rinput.obresult.configuration

        getters_seq = self.create_getters()
        return self.init_filters_generic(rinput, getters_seq, ins)

    def create_getters(self):
        """Return the sequences of getters used by :meth:`init_filters`, none by default."""
        return []

    def gather_info(self, recipeinput):
        """Return the metadata of the images and observation results of `recipeinput`, by field."""
        # Requires collaboration from datamodel
        from numina.core import DataFrame

        klass = recipeinput.__class__
        metadata = {}
        for key in klass.stored():
            val = getattr(recipeinput, key)
            if isinstance(val, DataFrame):
                metadata[key] = self.datamodel.gather_info(val)
            # elif isinstance(val, ObservationResult):
            elif hasattr(val, "frames"):
                metadata[key] = self.datamodel.gather_info_oresult(val)
            else:
                pass
        return metadata


def _create_io_class(recipe_class, suffix, base, fields):
    """Create the input or result class of a recipe, with its fields"""
    if not fields:
        return base
    name = f"{recipe_class.__name__}{suffix}"
    attributes = dict(fields)
    attributes["__module__"] = recipe_class.__module__
    attributes["__qualname__"] = f"{recipe_class.__qualname__}.{name}"
    return generate_docs(type(name, (base,), attributes))


def timeit(method):
    """Decorator to measure the time used by the recipe"""

    import datetime

    @functools.wraps(method)
    def timed_method(self, rinput):

        time_start = datetime.datetime.now(datetime.UTC)
        result = method(self, rinput)
        time_end = datetime.datetime.now(datetime.UTC)
        result.time_it(time_start, time_end)
        self.logger.info("total time measured")
        return result

    return timed_method
