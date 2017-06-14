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


class Pipeline(object):
    """Base class for pipelines."""
    def __init__(self, instrument, name, recipes, version=1):
        self.instrument = instrument
        self.name = name
        self.recipes = recipes
        self.version = version

    def get_recipe(self, mode):
        node = self.recipes[mode]
        return node['class']

    def get_recipe_object(self, mode):
        recipe_entry = self.recipes[mode]

        recipe_fqn = recipe_entry['class']
        args = recipe_entry.get('args', ())
        kwargs = recipe_entry.get('kwargs', {})
        Cls = numina.core.objimport.import_object(recipe_fqn)
        # Like Pickle protocol
        recipe = Cls.__new__(Cls, *args)
        recipe.__init__(*args, **kwargs)
        
        recipe.mode = mode
        recipe.instrument = self.instrument

        recipe.configure(**kwargs)

        # Like pickle protocol
        if 'state' in recipe_entry:
            recipe.__setstate__(recipe_entry['state'])

        return recipe


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
            self.products = []

    def query_provides(self, productname, search=False):
        """Return the mode that provides a given product"""

        for p in self.products:
            if p.name == productname:
                return p
        else:
            if search:
                return self.search_mode_provides(productname)

            raise ValueError('no mode provides %s' % productname)

    def search_mode_provides(self, productname):
        """Search the mode that provides a given product"""

        for obj, mode, field in self.iterate_mode_provides():
            # extract name from obj
            name = obj.__class__.__name__
            if name == productname:
                return ProductEntry(name, mode, field)
        else:
            raise ValueError('no mode provides %s' % productname)

    def iterate_mode_provides(self):
        """Return the mode that provides a given product"""

        for mode in self.modes:
            mode_key = mode.key
            default_pipeline = self.pipelines['default']
            try:
                fqn = default_pipeline.get_recipe(mode_key)
                recipe_class = numina.core.objimport.import_object(fqn)
                for key, provide in recipe_class.products().items():
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