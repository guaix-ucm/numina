#
# Copyright 2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Classes for static instrument configuration"""


import pkgutil
import json

from six import StringIO

import numina.util.objimport as obi


class BaseConfig(object):
    def __init__(self, name, uuid, date_start, date_end=None, description=""):
        self.name = name
        self.uuid = uuid
        self.date_start = date_start
        self.date_end = date_end
        self.description = description

    def __setstate__(self, state):
        self.name = state['name']
        self.uuid = state['uuid']
        self.date_start = state['date_start']
        self.date_end = state['date_end']
        self.description = state.get('description', '')


class InstrumentConfiguration(BaseConfig):
    def __init__(self, name, uuid, date_start, date_end=None, description=""):
        super(InstrumentConfiguration, self).__init__(name, uuid, date_start, date_end=date_end, description=description)
        self.components = {}
        self.instrument = self.name

    @classmethod
    def create_null(cls):
        return InstrumentConfiguration('EMPTY', '', date_start=0)


    def __setstate__(self, state):
        super(InstrumentConfiguration, self).__setstate__(state)
        self.instrument = self.name
        self.components = {}
        # FIXME: better logic
        p = state['loader']['path']
        for key, uuid in state['components'].items():
            resource_name = 'component-{}.json'.format(uuid)
            c = component_loader(p, resource_name)
            self.components[key] = c

    def get(self, path, **kwds):
        # split key
        vals = path.split('.')
        component = vals[0]
        key = vals[1]
        conf = self.components[component]
        return conf.get(key, **kwds)


class ComponentFacade(BaseConfig):
    def __init__(self, name, uuid, date_start, date_end=None, description=""):
        super(ComponentFacade, self).__init__(name, uuid, date_start, date_end=date_end, description=description)
        self.configurations = {}

    def __setstate__(self, state):
        super(ComponentFacade, self).__setstate__(state)
        self.configurations = {}

        configs = state.get('configurations', {})
        for key, val in configs.items():
            if 'uuid' in val:
                # remote component
                #
                # FIXME: better logic
                p = state['loader']['path']
                uuid = val['uuid']
                resource_name = 'component-{}.json'.format(uuid)
                resource = component_loader(p, resource_name)
                if key in resource.configurations:
                    self.configurations[key] = resource.configurations[key]
                else:
                    raise KeyError('{} in configuration {}'.format(key, uuid))
            else:
                self.configurations[key] = ConfigurationEntry(val['values'], val['depends'])

    def get(self, key, **kwds):
        conf = self.configurations[key]
        return conf.get(**kwds)


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


def instrument_loader(modpath, inspath):
    data = pkgutil.get_data(modpath, inspath)
    fp = StringIO(data.decode('utf-8'))
    state = json.load(fp)
    classname = state.get('fqn_class', 'numina.core.insconf.InstrumentConfiguration')
    #
    if 'loader' not in state:
        loader = dict(mode='package', path=modpath)
        state['loader'] = loader
    #
    return config_loader_state(state, classname)


def component_loader(modpath, compath):
    data = pkgutil.get_data(modpath, compath)
    fp = StringIO(data.decode('utf-8'))
    state = json.load(fp)
    classname = state.get('fqn_class', 'numina.core.insconf.ComponentFacade')
    #
    if 'loader' not in state:
        loader = dict(mode='package', path=modpath)
        state['loader'] = loader
    #
    return config_loader_state(state, classname)


def config_loader_state(state, classname):

    LoadClass = obi.import_object(classname)
    ins = LoadClass.__new__(LoadClass)
    ins.__setstate__(state)
    return ins

