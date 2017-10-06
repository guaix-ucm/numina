#
# Copyright 2011-2017 Universidad Complutense de Madrid
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

"""DRP related classes"""

import warnings

import numina.core.deptree
import numina.core.objimport
import numina.core.products
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
        return numina.core.objimport.import_object(recipe_fqn)

    def _get_base_object(self, entry):

        Cls = self._get_base_class(entry)

        args = entry.get('args', ())
        kwargs = entry.get('kwargs', {})

        # Like Pickle protocol
        recipe = Cls.__new__(Cls, *args)
        # Addition
        recipe.__init__(*args, **kwargs)

        # Like pickle protocol
        if 'state' in entry:
            recipe.__setstate__(entry['state'])

        return recipe

    def get_recipe_object(self, mode):
        """Load recipe object, according to observing mode"""

        recipe_entry = self.recipes[mode]
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
    """Description of an Instrument Data Reduction Pipeline"""
    def __init__(self, name, configurations, modes, pipelines, products=None, datamodel=None):
        self.name = name
        self.configurations = configurations
        self.selector = None
        self.modes = modes
        self.pipelines = pipelines
        if datamodel:
            self.datamodel = datamodel()
        else:
            self.datamodel = numina.datamodel.DataModel()

    def query_provides(self, product, pipeline='default', search=False):
        """Return the mode that provides a given product"""

        if search:
            return self.search_mode_provides(product)

        pipe = self.pipelines[pipeline]
        try:
            return pipe.who_provides(product)
        except KeyError:
            raise ValueError('no mode provides %s' % product)

    def search_mode_provides(self, product, pipeline='default'):
        """Search the mode that provides a given product"""

        pipeline = self.pipelines[pipeline]
        for obj, mode, field in self.iterate_mode_provides(self.modes, pipeline):
            # extract name from obj
            if obj.name() == product:
                return ProductEntry(obj.name(), mode.key, field)
        else:
            raise ValueError('no mode provides %s' % product)

    def iterate_mode_provides(self, modes, pipeline):
        """Return the mode that provides a given product"""

        for mode in modes:
            mode_key = mode.key
            try:
                recipe = pipeline.get_recipe_object(mode_key)
                for key, provide in recipe.products().items():
                    if isinstance(provide.type, numina.core.products.DataProductTag):
                        yield provide.type, mode, key
            except KeyError:
                warnings.warn('Mode {} has not recipe'.format(mode_key))

    def configuration_selector(self, obsres):
        if self.selector is not None:
            key = self.selector(obsres)
        else:
            key = 'default'
        return self.configurations[key]

    def product_label(self, tipo):
        return tipo.name()


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


class InstrumentConfiguration(object):

    def __init__(self, instrument):
        self.instrument = instrument
        self.name = 'config'
        self.uuid = ''
        self.data_start = 0
        self.data_end = 0
        self.components = {}

    def get(self, path, **kwds):
        # split key
        vals = path.split('.')
        component = vals[0]
        key = vals[1]
        conf = self.components[component]
        return conf.get(key, **kwds)


class ConfigurationEntry(object):
    def __init__(self, values, depends):
        self.values = values
        self.depends = depends

    def get(self, **kwds):
        result = self.values
        for dep in self.depends:
            key = kwds[dep]
            result = result[key]
        return result


class ComponentConfigurations(object):

    def __init__(self):
        self.name = 'config'
        self.uuid = ''
        self.data_start = 0
        self.data_end = 0
        self.component = 'component'
        self.configurations = {}

    def get(self, key, **kwds):
        conf = self.configurations[key]
        return conf.get(**kwds)


class ObservingMode(object):
    """Observing modes of an Instrument."""
    def __init__(self):
        self.name = ''
        self.uuid = ''
        self.key = ''
        self.url = ''
        self.instrument = ''
        self.summary = ''
        self.description = ''
        self.status = ''
        self.date = ''
        self.reference = ''
        self.tagger = None
        self.validator = None

    def validate(self):
        return True

    def __repr__(self):
        return "ObservingMode(name={})".format(self.name)