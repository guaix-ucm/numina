#
# Copyright 2011-2016 Universidad Complutense de Madrid
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


class Pipeline(object):
    """Base class for pipelines."""
    def __init__(self, name, recipes, version=1):
        self.name = name
        self.recipes = recipes
        self.version = version

    def get_recipe(self, mode):
        return self.recipes[mode]


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

    def configuration_selector(self, obsres):
        if self.selector is not None:
            key = self.selector(obsres)
        else:
            key = 'default'
        return self.configurations[key]


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