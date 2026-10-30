#
# Copyright 2011-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""DRP related classes"""

import copy
import warnings
import logging

import numina.core.query
import numina.util.objimport
import numina.util.parser
import numina.datamodel


class Pipeline:
    """Base class for pipelines.

    Parameters
    ----------
    instrument : str
        Name of the instrument.
    name : str
        Name of the pipeline.
    recipes : dict
        Description of the recipe of each observing mode, by mode key.
    version : int, optional
        Version of the pipeline.
    products : dict, optional
        Description of the product class of each mode, by mode key.
    provides : list of ProductEntry, optional
        The products provided by the observing modes.
    """

    def __init__(self, instrument, name, recipes, version=1, products=None, provides=None):
        self.instrument = instrument
        self.name = name
        self.recipes = recipes
        self.products = {} if products is None else products
        self.version = version
        # The InstrumentDRP of the pipeline, set by InstrumentDRP
        self.drp = None

        # Query by different keys
        self._provides_by_p = {}
        self._provides_by_r = {}

        provides_ = [] if provides is None else provides
        for k in provides_:
            self._provides_by_p[k.name] = k
            self._provides_by_r[k.mode] = k

    def _get_base_class(self, entry):
        """Import the class named in the ``class`` key of `entry`."""

        recipe_fqn = entry["class"]
        return numina.util.objimport.import_object(recipe_fqn)

    def _get_base_object(self, entry):
        """Build an object of the class of `entry`, with its ``args`` and ``kwargs``.

        The ``links`` of the entry are passed as the keyword argument
        ``query_options``, and its ``state``, if present, is set with
        ``__setstate__``.
        """

        Cls = self._get_base_class(entry)

        args = entry.get("args", ())
        kwargs = entry.get("kwargs", {})
        links = entry.get("links", {})
        if links:
            kwargs["query_options"] = links

        recipe = Cls.__new__(Cls, *args, **kwargs)
        recipe.__init__(*args, **kwargs)

        # Like pickle protocol
        if "state" in entry:
            recipe.__setstate__(entry["state"])

        return recipe

    def get_recipe_object(self, mode):
        """Load recipe object, according to observing mode"""

        if isinstance(mode, ObservingMode):
            key_mode = mode.key
        elif isinstance(mode, str):
            key_mode = mode
        else:
            key_mode = mode

        recipe_entry = self.recipes[key_mode]
        recipe = self._get_base_object(recipe_entry)

        # Init additional members
        recipe.mode = mode
        recipe.instrument = self.instrument
        recipe.pipeline = self.name
        if self.drp is not None:
            recipe.drp_version = self.drp.version

        return recipe

    load_recipe_object = get_recipe_object

    def load_product_class(self, mode):
        """Load recipe object, according to observing mode"""

        product_entry = self.products[mode]

        return self._get_base_class(product_entry)

    # Used by numina-plugin-db
    def load_product_from_name(self, label):
        """Build the product type named `label`, as ``Class(arg1=val1, ...)``.

        The class is the product class of the pipeline registered under
        the part of `label` before the parenthesis.
        """

        short, _ = numina.util.parser.split_type_name(label)
        klass = self.load_product_class(short)
        return klass.from_name(label)

    def who_provides(self, product_label):
        """Return the ProductEntry for some requirement"""

        entry = self._provides_by_p[product_label]
        return entry

    def provides(self, mode_label):
        """Return the ProductEntry for some mode"""

        entry = self._provides_by_r[mode_label]
        return entry


class InstrumentDRP:
    """Description of an Instrument Data Reduction Pipeline

    The DRPs are usually created from their ``drp.yaml`` with
    :func:`numina.core.pipelineload.drp_load`, in the function of their
    entry point in the group ``numina.pipeline.1``.

    Parameters
    ----------
    name : str
        Name of the instrument.
    configurations : dict
        The instrument configurations (profiles), the contents of their
        JSON files by file name.
    modes : dict of ObservingMode
        The observing modes, by key.
    pipelines : dict of Pipeline
        The pipelines, by name. Each pipeline maps the observing modes to
        their recipes.
    products : optional
        Not used.
    datamodel : type, optional
        The class of the data model of the instrument, a subclass of
        :class:`numina.datamodel.DataModel`. The DRP holds an instance.
        By default, the base class.
    version : str, optional
        Version of the DRP.
    default_requirements : dict, optional
        Default values of the requirements, by profile, pipeline and mode,
        see :meth:`default_requirements`.
    calibrations : list of dict, optional
        Datasets of calibrations of the instrument, with their name, url,
        known_hash and optionally description and directory, installed
        with ``numina inittree``.

    Attributes
    ----------
    checker : callable or None
        The function that checks the objects of the instrument, called
        as ``checker(obj, astype=None, level=None)``. It returns True or
        raises an exception if the object is not valid. The DRP sets it
        in the function of its entry point, and it is used by
        :data:`numina.core.config.check`.
    package : str
        The package of the DRP, set by
        :func:`~numina.core.pipelineload.drp_load`.
    profiles : str
        The module with the instrument configurations, set by
        :func:`~numina.core.pipelineload.drp_load`.

    """

    def __init__(
        self,
        name,
        configurations,
        modes,
        pipelines,
        products=None,
        datamodel=None,
        version="undefined",
        default_requirements=None,
        calibrations=None,
    ):
        self.name = name
        self.configurations = configurations
        self.modes = modes
        self.pipelines = pipelines
        if datamodel:
            self.datamodel = datamodel()
        else:
            self.datamodel = numina.datamodel.DataModel()
        self.version = version
        self.checker = None
        self.calibrations = [] if calibrations is None else calibrations
        self._def_reqs = {} if default_requirements is None else default_requirements
        if isinstance(pipelines, dict):
            for pipeline in pipelines.values():
                pipeline.drp = self

    def query_provides(self, product, pipeline="default", search=False):
        """Return the mode that provides a given product"""

        if search:
            return self.search_mode_provides(product)

        pipe = self.pipelines[pipeline]
        try:
            return pipe.who_provides(product)
        except KeyError:
            raise ValueError(f"no mode provides {product}")

    def search_mode_provides(self, product, pipeline="default"):
        """Search the mode that provides a given product"""

        pipeline = self.pipelines[pipeline]
        for obj, mode, field in self.iterate_mode_provides(self.modes, pipeline):
            # extract name from obj
            if obj.name() == product:
                return ProductEntry(obj.name(), mode.key, field)
        else:
            raise ValueError(f"no mode provides {product}")

    def iterate_mode_provides(self, modes, pipeline):
        """Return the mode that provides a given product"""

        for mode_key, mode in modes.items():
            try:
                recipe = pipeline.get_recipe_object(mode_key)
                for key, provide in recipe.products().items():
                    if provide.type.isproduct():
                        yield provide.type, mode, key
            except KeyError:
                warnings.warn(f"Mode {mode_key} has not recipe")

    def product_label(self, tipo):
        """The label of the product type `tipo`, its name"""
        return tipo.name()

    def select_profile(self, obresult):
        """Select instrument profile based on OB"""

        logger = logging.getLogger(__name__)
        logger.debug("calling default profile selector")
        # check configuration
        insconf = obresult.profile
        if insconf != "default" and insconf != "00000000-0000-0000-0000-000000000000":
            # Using profile as a UUID
            key = insconf
            date_obs = None
            keyname = "uuid"
        else:
            # get first possible image
            sample_frame = obresult.get_sample_frame()
            if sample_frame is None:
                key = obresult.instrument
                date_obs = None
                keyname = "name"
            else:
                img = sample_frame.open()
                try:
                    return self.select_profile_image(img)
                finally:
                    # Close the file only if it was opened here
                    if sample_frame.frame is None:
                        img.close()
        return key, date_obs, keyname

    def select_profile_image(self, img):
        """Select instrument profile based on FITS"""

        extr = self.datamodel.extractor_map["fits"]

        date_obs = extr.extract("observation_date", img)
        key = extr.extract("insconf", img)
        if key is not None:
            keyname = "uuid"
        else:
            key = extr.extract("instrument", img)
            keyname = "name"

        return key, date_obs, keyname

    def get_recipe_object(self, mode_name, pipeline_name="default"):
        """Build a recipe object from a given mode name"""
        active_mode = self.modes[mode_name]
        try:
            active_pipeline = self.pipelines[pipeline_name]
        except KeyError:
            msg = (
                f"pipeline '{pipeline_name}' not found in DRP '{self.name}', "
                f"available pipelines are {list(self.pipelines)}"
            )
            raise KeyError(msg) from None
        recipe = active_pipeline.get_recipe_object(active_mode)
        return recipe

    def default_requirements(self):
        """Default values of requirements, by profile, pipeline and mode"""
        # A copy, so that callers can modify it without changing the DRP
        return copy.deepcopy(self._def_reqs)


class ProductEntry:
    """Product provided by an observing mode of a pipeline.

    Parameters
    ----------
    name : str
        Name of the product type.
    mode : str
        Key of the observing mode that provides the product.
    field : str
        Name of the result of the recipe that holds the product.
    alias : str, optional
        Short name of the product. By default, the last component of
        `name` split by dots.
    """

    def __init__(self, name, mode, field, alias=None):
        self.name = name
        self.mode = mode
        self.field = field
        if alias is None:
            split_name = name.split(".")
            self.alias = split_name[-1]
        else:
            self.alias = alias

    def __repr__(self):
        msg = 'ProductEntry(name="{}", mode="{}", field="{}")'.format(self.name, self.mode, self.field)
        return msg


class ObservingMode:
    """Observing modes of an Instrument.

    Parameters
    ----------
    instrument : str, optional
        Name of the instrument.
    """

    def __init__(self, instrument=""):
        self.name = ""
        self.key = ""
        self.instrument = instrument
        self.summary = ""
        self.description = ""
        self.validator = None
        self.build_ob_options = None
        self.rawimage = None

    def validate(self, obsres):
        """Deprecated, it is not used by numina and will be removed

        The observation results are validated with the attribute validator.
        """
        warnings.warn(
            "ObservingMode.validate is deprecated, it is not used by numina and will be removed",
            DeprecationWarning,
            stacklevel=2,
        )
        return True

    def build_ob(self, partial_ob, backend, options=None):
        """Complete the observing block `partial_ob` and return it.

        If the options (`options`, or ``build_ob_options`` if not given)
        are a ``ResultOf``, the results of other observing blocks are
        searched with ``backend.search_result_relative`` and added to
        ``partial_ob.results``, by id. The method can be replaced with the
        key ``builder`` of the mode in ``drp.yaml``.
        """

        mod = options or self.build_ob_options

        if isinstance(mod, numina.core.query.ResultOf):
            result_type = mod.result_type
            name = "relative_result"
            val = backend.search_result_relative(name, result_type, partial_ob, result_desc=mod)
            for r in val:
                partial_ob.results[r.id] = r.content

        return partial_ob

    def __repr__(self):
        return f"ObservingMode(name={self.name}, instrument={self.instrument})"
