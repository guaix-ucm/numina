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

"""Build a LoadableDRP from a yaml file"""

import pkgutil
import yaml
import os

from six import StringIO

from .objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import InstrumentDRP
from .pipeline import InstrumentConfiguration
from .pipeline import ProductEntry
from .taggers import get_tags_from_full_ob


def check_section(node, section, keys=None):
    """Validate keys in a section"""
    if keys:
        for key in keys:
            if key not in node:
                raise ValueError('Missing key %r inside %r node' % (key, section))


def drp_load(package, resource, confclass=None):
    """Load the DRPS from a resource file."""
    data = pkgutil.get_data(package, resource)
    return drp_load_data(package, data, confclass=confclass)


def drp_load_data(package, data, confclass=None):
    """Load the DRPS from data."""
    drpdict = yaml.load(data)
    ins = load_instrument(package, drpdict, confclass=confclass)
    return ins


def load_modes(node):
    """Load all observing modes"""
    return [load_mode(child) for child in node]


def load_mode(node):
    """Load one observing mdode"""
    obs_mode = ObservingMode()
    obs_mode.__dict__.update(node)

    # handle validator
    load_mode_validator(obs_mode, node)
    
    # handle tagger:
    ntagger = node.get('tagger')

    if ntagger is None:
        pass
    elif isinstance(ntagger, list):

        def full_tagger(obsres):
            return get_tags_from_full_ob(obsres, reqtags=ntagger)

        obs_mode.tagger = full_tagger
    elif isinstance(ntagger, str):
        # load function
        obs_mode.tagger = import_object(ntagger)

    else:
        raise TypeError('tagger must be None, a list or a string')

    return obs_mode


def load_mode_validator(obs_mode, node):
    """Load observing mode validator"""

    nval = node.get('validator')

    if nval is None:
        pass
    elif isinstance(nval, str):
        # load function
        obs_mode.validator = import_object(nval)
    else:
        raise TypeError('validator must be None or a string')

    return obs_mode


def load_pipelines(node):
    keys = ['default']
    check_section(node, 'pipelines', keys=keys)

    pipelines = {}
    for key in node:
        pipelines[key] = load_pipeline(key, node[key])
    return pipelines


def load_confs(package, node, confclass=None):
    keys = ['values']
    check_section(node, 'configurations', keys=keys)

    path = node.get('path')
    if path:
        modpath = path
    else:
        modpath = "%s.instrument.configs" % package

    if confclass is None:
        loader = DefaultLoader(modpath=modpath)

        def confclass(uuid):
            return build_instrument_config(uuid, loader)

    default_entry = node.get('default')
    tagger = node.get('tagger')
    if tagger:
        ins_tagger = import_object(tagger)
    else:
        ins_tagger = lambda obsres: 'default'

    values = node['values']
    confs = {}
    for uuid in values:
        confs[uuid] = confclass(uuid)
    if default_entry:
        confs['default'] = confs[default_entry]
    else:
        if 'default' not in confs:
            # Choose the first if is not already defined
            confs['default'] = confs[values[0]]
    return confs, ins_tagger


def load_pipeline(name, node):

    keys = ['recipes', 'version']
    check_section(node, 'pipeline', keys=keys)

    recipes = node['recipes']
    version = node['version']
    return Pipeline(name, recipes, version)


def load_prods(node, allmodes):
    result = []
    for entry in node:
        name = entry['name']
        alias = entry.get('alias')
        mode_name = entry['mode']
        field = entry['field']
        for mode in allmodes:
            if mode.key == mode_name:
                prod = ProductEntry(name, mode, field, alias=alias)
                result.append(prod)
                break
        else:
            # Undefined mode
            pass

    return result


def load_instrument(package, node, confclass=None):
    # Verify keys...
    keys = ['name', 'configurations', 'modes', 'pipelines']
    check_section(node, 'root', keys=keys)

    # name = node['name']
    pipe_node = node['pipelines']
    mode_node = node['modes']
    conf_node = node['configurations']
    prod_node = node.get('products', [])

    trans = {'name': node['name']}
    trans['pipelines'] = load_pipelines(pipe_node)
    trans['modes'] = load_modes(mode_node)
    confs, selector = load_confs(package, conf_node, confclass=confclass)
    trans['configurations'] = confs
    trans['products'] = load_prods(prod_node, trans['modes'])
    ins = InstrumentDRP(**trans)
    ins.selector = selector
    return ins


class PathLoader(object):
    def __init__(self, inspath, compath):
        self.inspath = inspath
        self.compath = compath

    def build_component_fp(self, key):
        fname = 'component-%s.json' % key
        fcomp = open(os.path.join(self.compath, fname))
        return fcomp

    def build_instrument_fp(self, key):
        fname = 'instrument-%s.json' % key
        fcomp = open(os.path.join(self.inspath, fname))
        return fcomp


class DefaultLoader(object):
    def __init__(self, modpath):
        self.modpath = modpath

    def build_component_fp(self, key):
        fname = 'component-%s.json' % key
        return self.build_type_fp(fname)

    def build_instrument_fp(self, key):
        fname = 'instrument-%s.json' % key
        return self.build_type_fp(fname)

    def build_type_fp(self, fname):
        data = pkgutil.get_data(self.modpath, fname)
        fcomp = StringIO(data.decode('utf-8'))
        return fcomp


def build_instrument_config(uuid, loader):

    fp = loader.build_instrument_fp(uuid)

    mm = load_instrument_configuration_from_file(fp, loader=loader)
    return mm


def load_ce_from_file(fp):
    import json
    from .pipeline import ConfigurationEntry
    contents = json.load(fp)

    if contents['type'] != 'configuration':
        raise ValueError('type is not configuration')

    key = contents['name']
    confs = contents['configurations']
    val = confs[key]
    mm = ConfigurationEntry(val['values'], val['depends'])
    return mm


def load_cc_from_file(fp, loader):
    from .pipeline import ComponentConfigurations, ConfigurationEntry
    import json
    contents = json.load(fp)
    mm = ComponentConfigurations()
    if contents['type'] != 'component':
        raise ValueError('type is not component')
    mm.component = contents['name']
    mm.name = contents['description']
    mm.uuid = contents['uuid']
    mm.data_start = 0
    mm.data_end = 0
    for key, val in contents['configurations'].items():
        if 'uuid' in val:
            # remote component
            fp = loader.build_component_fp(val['uuid'])
            mm.configurations[key] = load_ce_from_file(fp)
        else:
            mm.configurations[key] = ConfigurationEntry(val['values'], val['depends'])
    return mm


def load_instrument_configuration_from_file(fp, loader):
    import json

    contents = json.load(fp)
    if contents['type'] != 'instrument':
        raise ValueError('type is not instrument')

    mm = InstrumentConfiguration.__new__(InstrumentConfiguration)

    mm.instrument = contents['name']
    mm.name = contents['description']
    mm.uuid = contents['uuid']
    mm.data_start = 0
    mm.data_end = 0
    mm.components = {}
    for cname, cuuid in contents['components'].items():
        fcomp = loader.build_component_fp(cuuid)
        rr = load_cc_from_file(fcomp, loader=loader)
        mm.components[cname] = rr
    return mm
