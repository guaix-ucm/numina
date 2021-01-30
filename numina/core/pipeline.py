#
# Copyright 2011-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""DRP related classes"""

import warnings
import logging

import numina.core.query
import numina.core.deptree
import numina.util.objimport
import numina.util.parser
import numina.datamodel


class Pipeline(object):
    """Base class for pipelines."""
    def __init__(self, instrument, name, recipes, version=1, products=None, provides=None):
        self.instrument = instrument
        self.name = name
        self.recipes = recipes
        self.products = {} if products is None else products
        self.version = version

        # Query by different keys
        self._provides_by_p = {}
        self._provides_by_r = {}

        provides_ = [] if provides is None else provides
        for k in provides_:
            self._provides_by_p[k.name] = k
            self._provides_by_r[k.mode] = k

    def get_recipe(self, mode):
        node = self.recipes[mode]
        return node['class']

    def _get_base_class(self, entry):

        recipe_fqn = entry['class']
        return numina.util.objimport.import_object(recipe_fqn)

    def _get_base_object(self, entry):

        Cls = self._get_base_class(entry)

        args = entry.get('args', ())
        kwargs = entry.get('kwargs', {})
        links = entry.get('links', {})
        if links:
            kwargs['query_options'] = links

        recipe = Cls.__new__(Cls, *args, **kwargs)
        recipe.__init__(*args, **kwargs)

        # Like pickle protocol
        if 'state' in entry:
            recipe.__setstate__(entry['state'])

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

        return recipe

    load_recipe_object = get_recipe_object

    def load_product_object(self, name):
        """Load product object, according to name"""

        product_entry = self.products[name]

        product = self._get_base_object(product_entry)

        return product

    def load_product_class(self, mode):
        """Load recipe object, according to observing mode"""

        product_entry = self.products[mode]

        return self._get_base_class(product_entry)

    def load_product_from_name(self, label):

        short, _ =   numina.util.parser.split_type_name(label)
        klass = self.load_product_class(short)
        return klass.from_name(label)

    def depsolve(self):
        """Load all recipes to search for products"""
        # load everything
        requires = {}
        provides = {}
        for mode, r in self.recipes.items():
            l = self.load_recipe_object(mode)

            for field, vv in l.requirements().items():
                if vv.type.isproduct():
                    name = vv.type.name()
                    pe = ProductEntry(name, mode, field)
                    requires[name] = pe

            for field, vv in l.products().items():
                if vv.type.isproduct():
                    name = vv.type.name()
                    pe = ProductEntry(name, mode, field)
                    provides[name] = pe

        return requires, provides

    def who_provides(self, product_label):
        """Return the ProductEntry for some requirement"""

        entry = self._provides_by_p[product_label]
        return entry

    def provides(self, mode_label):
        """Return the ProductEntry for some mode"""

        entry = self._provides_by_r[mode_label]
        return entry

    def _query_recipe(self, thismode, modes):
        if thismode in modes:
            thisnode = modes[thismode]
        else:
            thisnode = numina.core.deptree.DepNode(thismode)
            modes[thismode] = thisnode

        l = self.load_recipe_object(thismode)
        for field, vv in l.requirements().items():
            if vv.type.isproduct():

                result = self.who_provides(vv.type.name())
                # get first result
                if result:
                    good = result[0]
                    recurse = False
                    if good.mode not in modes:
                        # Add link
                        recurse = True
                        node = numina.core.deptree.DepNode(good.mode)
                        modes[good.mode] = node

                    node = modes[good.mode]
                    weight = 1
                    if vv.optional:
                        weight = 0
                    # print "link", thismode, "with", good.mode
                    newlink = numina.core.deptree.DepLink(node, weight=weight)
                    thisnode.links.append(newlink)
                    if recurse:
                        self._query_recipe(good.mode, modes)
                else:
                    # No recipe provides this product
                    pass
        return modes

    def query_recipe(self, mode):
        """Recursive query of all calibrations required by a mode"""
        allmodes = self._query_recipe(mode, {})
        return allmodes[mode]


class InstrumentDRP(object):
    """Description of an Instrument Data Reduction Pipeline

    Parameters
    ==========
       name : str
           Name of the instrument
       configurations : dict of InstrumentConfiguration
       modes : dict of ObservingModes
       pipeline : dict of Pipeline

    """
    def __init__(self, name, configurations, modes, pipelines, products=None, datamodel=None, version='undefined'):
        self.name = name
        self.configurations = configurations
        self.modes = modes
        self.pipelines = pipelines
        if datamodel:
            self.datamodel = datamodel()
        else:
            self.datamodel = numina.datamodel.DataModel()
        self.version = version

    def query_provides(self, product, pipeline='default', search=False):
        """Return the mode that provides a given product"""

        if search:
            return self.search_mode_provides(product)

        pipe = self.pipelines[pipeline]
        try:
            return pipe.who_provides(product)
        except KeyError:
            raise ValueError(f'no mode provides {product}')

    def search_mode_provides(self, product, pipeline='default'):
        """Search the mode that provides a given product"""

        pipeline = self.pipelines[pipeline]
        for obj, mode, field in self.iterate_mode_provides(self.modes, pipeline):
            # extract name from obj
            if obj.name() == product:
                return ProductEntry(obj.name(), mode.key, field)
        else:
            raise ValueError(f'no mode provides {product}')

    def iterate_mode_provides(self, modes, pipeline):
        """Return the mode that provides a given product"""

        for mode_key, mode in modes.items():
            try:
                recipe = pipeline.get_recipe_object(mode_key)
                for key, provide in recipe.products().items():
                    if provide.type.isproduct():
                        yield provide.type, mode, key
            except KeyError:
                warnings.warn(f'Mode {mode_key} has not recipe')

    def configuration_selector(self, obsres):
        warnings.warn("configuration_selector is deprecated, use 'select_configuration' instead",
                      DeprecationWarning, stacklevel=2)
        return self.select_configuration_old(obsres)

    def product_label(self, tipo):
        return tipo.name()

    def select_configuration_old(self, obresult):
        """Select instrument configuration based on OB"""

        logger = logging.getLogger(__name__)
        logger.debug('calling default configuration selector')

        # get first possible image
        ref_frame = obresult.get_sample_frame()
        ref = ref_frame.open()
        extr = self.datamodel.extractor_map['fits']
        if ref:
            # get INSCONF configuration
            result = extr.extract('insconf', ref)
            if result:
                # found the keyword, try to match
                logger.debug('found insconf config uuid=%s', result)
                # Use insconf as uuid key
                if result in self.configurations:
                    return self.configurations[result]
                else:
                    # Additional check for conf.name
                    for conf in self.configurations.values():
                        if conf.name == result:
                            return conf
                    else:
                        raise KeyError(f'insconf {result} does not match any config')

            # If not, try to match by DATE
            date_obs = extr.extract('observation_date', ref)
            for key, conf in self.configurations.items():
                if key == 'default':
                    # skip default
                    continue
                if conf.date_end is not None:
                    upper_t = date_obs < conf.date_end
                else:
                    upper_t = True
                if upper_t and (date_obs >= conf.date_start):
                    logger.debug('found date match, config uuid=%s', key)
                    return conf
        else:
            logger.debug('no match, using default configuration')
            return self.configurations['default']

    def select_configuration(self, obresult):
        return self.select_profile(obresult)

    def select_profile(self, obresult):
        """Select instrument profile based on OB"""

        logger = logging.getLogger(__name__)
        logger.debug('calling default profile selector')
        # check configuration
        insconf = obresult.configuration
        if insconf != 'default':
            key = insconf
            date_obs = None
            keyname = 'uuid'
        else:
            # get first possible image
            sample_frame = obresult.get_sample_frame()
            if sample_frame is None:
                key = obresult.instrument
                date_obs = None
                keyname = 'name'
            else:
                return self.select_profile_image(sample_frame.open())
        return key, date_obs, keyname

    def select_profile_image(self, img):
        """Select instrument profile based on FITS"""

        extr = self.datamodel.extractor_map['fits']

        date_obs = extr.extract('observation_date', img)
        key = extr.extract('insconf', img)
        if key is not None:
            keyname = 'uuid'
        else:
            key = extr.extract('instrument', img)
            keyname = 'name'

        return key, date_obs, keyname

    def get_recipe_object(self, mode_name, pipeline_name='default'):
        """Build a recipe object from a given mode name"""
        active_mode = self.modes[mode_name]
        active_pipeline = self.pipelines[pipeline_name]
        recipe = active_pipeline.get_recipe_object(active_mode)
        return recipe


class ProductEntry(object):
    def __init__(self, name, mode, field, alias=None):
        self.name = name
        self.mode = mode
        self.field = field
        if alias is None:
            split_name = name.split('.')
            self.alias = split_name[-1]
        else:
            self.alias = alias

    def __repr__(self):
        msg = 'ProductEntry(name="{}", mode="{}", field="{}")'.format(
            self.name, self.mode, self.field)
        return msg


class ObservingMode(object):
    """Observing modes of an Instrument."""
    def __init__(self, instrument=''):
        self.name = ''
        self.key = ''
        self.instrument = instrument
        self.summary = ''
        self.description = ''
        self.tagger = None
        self.validator = None
        self.build_ob_options = None
        self.rawimage = None

    def validate(self, obsres):
        return True

    def build_ob(self, partial_ob, backend, options=None):

        mod = options or self.build_ob_options

        if isinstance(mod, numina.core.query.ResultOf):
            result_type = mod.result_type
            name = 'relative_result'
            val = backend.search_result_relative(name, result_type, partial_ob, result_desc=mod)
            for r in val:
                partial_ob.results[r.id] = r.content

        return partial_ob

    def tag_ob(self, partial):
        if self.tagger is not None:
            warnings.warn("per mode taggers are deprecated, recipe requirements provide al required informatation",
                          DeprecationWarning, stacklevel=2)
            partial.tags = self.tagger(partial)
        return partial

    def __repr__(self):
        return f"ObservingMode(name={self.name})"
