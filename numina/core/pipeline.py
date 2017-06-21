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

import numina.core.objimport
import numina.core.products
import numina.core.deptree

class Pipeline(object):
    """Base class for pipelines."""
    def __init__(self, instrument, name, recipes, version=1):
        self.instrument = instrument
        self.name = name
        self.recipes = recipes
        self.version = version

        self._cache = {}
        self._requires = {}
        self._provides = {}

    def get_recipe(self, mode):
        node = self.recipes[mode]
        return node['class']

    def get_recipe_object(self, mode):
        """Load recipe object, according to observing mode"""

        if mode in self._cache:
            return self._cache[mode]

        recipe_entry = self.recipes[mode]

        recipe_fqn = recipe_entry['class']
        args = recipe_entry.get('args', ())
        kwargs = recipe_entry.get('kwargs', {})
        Cls = numina.core.objimport.import_object(recipe_fqn)

        # Like Pickle protocol
        recipe = Cls.__new__(Cls, *args)
        # Addition
        recipe.__init__(*args, **kwargs)

        # Like pickle protocol
        if 'state' in recipe_entry:
            recipe.__setstate__(recipe_entry['state'])

        # Init additional members
        recipe.mode = mode
        recipe.instrument = self.instrument

        self._cache[mode] = recipe

        return recipe

    def init_depsolve(self):
        """Load all recipes to search for products"""
        # load everything
        for mode, r in self.recipes.items():
            l = self.get_recipe_object(mode)

            for field, vv in l.requirements().items():
                name = vv.type.name()
                pe = ProductEntry(name, mode, field, alias=None)
                #print("--->", field, vv.type, pe)

            for field, vv in l.products().items():
                name = vv.type.name()
                pe = ProductEntry(name, mode, field, alias=None)
                #print("+++>", field, vv.type, pe)
                self._provides[pe] = name

    def query_provides(self, obj):
        """Return the mode that provides some requirement"""
        key = obj.type.name()

        result = []
        for k in self._provides:
            if k.name == key:
                result.append(k)
        return result

    def _query_recipe(self, thismode, modes):
        if thismode in modes:
            thisnode = modes[thismode]
        else:
            thisnode = numina.core.deptree.DepNode(thismode)
            modes[thismode] = thisnode

        l = self.get_recipe_object(thismode)
        for field, vv in l.requirements().items():
            if vv.type.isproduct():

                result = self.query_provides(vv)
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
                    #print "link", thismode, "with", good.mode
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
    def __init__(self, name, configurations, modes, pipelines, products=None):
        self.name = name
        self.configurations = configurations
        self.selector = None
        self.modes = modes
        self.pipelines = pipelines
        self.products = products
        if products is None:
            self.products = {}

    def query_provides(self, product, search=False):
        """Return the mode that provides a given product"""

        try:
            return self.products[product]
        except KeyError:
            pass

        if search:
            return self.search_mode_provides(product)

        raise ValueError('no mode provides %s' % product)

    def search_mode_provides(self, product):
        """Search the mode that provides a given product"""

        for obj, mode, field in self.iterate_mode_provides():
            # extract name from obj
            name = obj.__class__.__name__
            if isinstance(obj, product):
                return ProductEntry(name, mode, field)
        else:
            raise ValueError('no mode provides %s' % product)

    def iterate_mode_provides(self):
        """Return the mode that provides a given product"""

        for mode in self.modes:
            mode_key = mode.key
            default_pipeline = self.pipelines['default']
            try:
                recipe = default_pipeline.get_recipe_object(mode_key)
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
        try:
            klass = tipo.__class__
            res = self.products[klass]
            return res.alias
        except KeyError:
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