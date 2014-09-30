#
# Copyright 2011-2014 Universidad Complutense de Madrid
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

'''DRP loader.'''

import logging
import pkgutil
import importlib

import pkg_resources
import yaml
import uuid

_logger = logging.getLogger('numina')


def import_object(path):
    '''Import an object given its fully qualified name.'''
    spl = path.split('.')
    cls = spl[-1]
    mods = '.'.join(spl[:-1])
    mm = importlib.import_module(mods)
    Cls = getattr(mm, cls)
    return Cls


class Pipeline(object):
    '''Base class for pipelines.'''
    def __init__(self, name, recipes, version=1):
        self.name = name
        self.recipes = recipes
        self.version = version

    def get_recipe(self, mode):
        return self.recipes[mode]


class InstrumentConfiguration(object):
    '''Configuration of an Instrument.'''
    def __init__(self, values):
        self.values = values


class Instrument(object):
    '''Description of an Instrument.'''
    def __init__(self, name, configurations, modes, pipelines):
        self.name = name
        self.configurations = configurations
        self.modes = modes
        self.pipelines = pipelines


class ObservingMode(object):
    '''Observing modes of an Instrument.'''
    def __init__(self):
        self.name = ''
        self.uuid = ''
        self.key = ''
        self.url = ''
        self.instrument = ''
        self.summary = ''
        self.description = ''
        self.recipe = ''
        self.recipe_class = None
        self.status = ''
        self.date = ''
        self.reference = ''


def om_repr(dumper, data):
    return dumper.represent_mapping('!om', data.__dict__)


def om_cons(loader, node):
    om = ObservingMode()
    value = loader.construct_mapping(node)
    om.__dict__ = value
    om.uuid = uuid.UUID(om.uuid)
    return om

yaml.add_representer(ObservingMode, om_repr)
yaml.add_constructor('!om', om_cons)


class LoadableDRP(object):
    '''Container for the loaded DRP.'''
    def __init__(self, instruments):
        self.instruments = instruments


def init_drp_system_s(namespace):
    '''Load all available DRPs in package 'namespace'.'''

    drp = {}

    for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__,
                                                    namespace.__name__ + '.'):
        try:
            loader = imp.find_module(name)
            mod = loader.load_module(name)
            mod_plugin = getattr(mod, '__numina_drp__', None)
            if mod_plugin:
                drp.update(mod_plugin.instruments)
            else:
                _logger.warning('Module %s does not contain a valid DRP', mod)
        except StandardError as error:
            _logger.warning('Problem importing %s, '
                            'error of type %s with message "%s"',
                            name, type(error), error
                            )

    return drp


def init_drp_system():
    '''Load all available DRPs in 'numina.pipeline' entry_point.'''

    drp = {}

    for entry in pkg_resources.iter_entry_points(group='numina.pipeline.1'):
        drp_loader = entry.load()
        mod = drp_loader()
        if mod:
            drp.update(mod.instruments)
        else:
            _logger.warning('Module %s does not contain a valid DRP', mod)

    return drp


def init_backends(namespace, backend='default'):
    '''Load all available DRPs in package 'namespace'.'''

    backends = []

    for imp, name, _is_pkg in pkgutil.walk_packages(namespace.__path__,
                                                    namespace.__name__ + '.'):
        try:
            loader = imp.find_module(name)
            mod = loader.load_module(name)
            mod_plugin = getattr(mod, '__numina_store__', None)

            if mod_plugin:
                _logger.debug(
                    'Loading backends from  %s.__numina_store__', name
                    )
                module = mod_plugin.get(backend, None)
                if module:
                    # Is module explicit or guessed
                    backends.append((module, True))
                else:
                    _logger.debug(
                        'No value for backend %s in %s',
                        backend, name
                        )
            else:
                _logger.debug('%s.__numina_store__ is undefined', name)
                # Guessing a module name as a fallback
                base = name.split('.')[-1]
                module = '%s.store' % base
                _logger.debug('Guess that a %s module is defined', module)
                # Module not explicit, guessed
                backends.append((module, False))

        except StandardError as error:
            _logger.warning(
                'Problem importing %s, error of'
                ' type %s with message "%s"',
                name, type(error), error
                )

        for module, defined in backends:
            try:
                importlib.import_module(module)
            except TypeError:
                _logger.warning(
                    'TypeError exception importing'
                    ' %s module, ignoring',
                    module
                    )
            except ImportError:
                if defined:
                    # Bother users only if they can fix the problem
                    _logger.warning(
                        'ImportError exception '
                        'importing %s module, ignoring',
                        module
                        )


def drp_load(package, resource):
    '''Load the DRPS from a resource file.'''
    ins_all = {}
    for yld in yaml.load_all(pkgutil.get_data(package, resource)):
        ins = load_instrument(yld)
        ins_all[ins.name] = ins

    return LoadableDRP(ins_all)


def load_modes(node):
    modes = list()
    for child in node:
        modes.append(load_mode(child))
    return modes


def load_mode(node):
    m = ObservingMode()
    m.__dict__ = node
    return m


def load_pipelines(node):
    keys = ['default']
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r in pipelines node', key)
    pipelines = dict()
    for key in node:
        pipelines[key] = load_pipeline(key, node[key])
    return pipelines


def load_confs(node):
    keys = ['default']
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r in configurations node', key)
    confs = dict()
    for key in node:
        confs[key] = load_conf(node[key])
    return confs


def load_pipeline(name, node):
    keys = ['recipes', 'version']
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r inside pipeline node', key)
    recipes = node['recipes']
    version = node['version']
    return Pipeline(name, recipes, version)


def load_conf(node):
    keys = []
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r inside configuration node', key)

    return InstrumentConfiguration(node)


def load_instrument(node):
    # Verify keys...
    keys = ['name', 'configurations', 'modes', 'pipelines']

    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r in root node', key)

    # name = node['name']
    pipe_node = node['pipelines']
    mode_node = node['modes']
    conf_node = node['configurations']

    trans = {'name': node['name']}
    trans['pipelines'] = load_pipelines(pipe_node)
    trans['modes'] = load_modes(mode_node)
    trans['configurations'] = load_confs(conf_node)

    return Instrument(**trans)


def print_i(ins):
    print ins.name
    print_c(ins.configurations)
    print_m(ins.modes)
    print_p(ins.pipelines)


def print_p(pipelines):
    print 'Pipelines'
    for p, n in pipelines.items():
        print ' pipeline', p
        print '   version', n.version
        print '   recipes'
        for m, r in n.recipes.items():
            print '    ', m, '->', r


def print_c(confs):
    print 'Configurations'
    for c in confs:
        print ' conf', c, confs[c].values


def print_m(modes):
    print 'Modes'
    for c in modes:
        print ' mode', c.key


def init_backends(backend='default'):
    '''Load storage modes'.'''

    backends = []

    for entry in pkg_resources.iter_entry_points(group='numina.storage.1'):
        store = entry.load()
        backends.append((store, True))

    return backends
