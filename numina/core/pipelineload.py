#
# Copyright 2011-2018 Universidad Complutense de Madrid
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

import six
from six import StringIO

from .objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import InstrumentDRP
from .pipeline import InstrumentConfiguration
from .pipeline import ProductEntry
from .query import ResultOf
from .taggers import get_tags_from_full_ob
import numina.util.convert as convert


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

    # handle builder
    load_mode_builder(obs_mode, node)

    # handle tagger:
    load_mode_tagger(obs_mode, node)

    return obs_mode


def load_mode_tagger(obs_mode, node):
    """Load observing mode OB tagger"""

    # handle tagger:
    ntagger = node.get('tagger')

    if ntagger is None:
        pass
    elif isinstance(ntagger, list):

        def full_tagger(obsres):
            return get_tags_from_full_ob(obsres, reqtags=ntagger)

        obs_mode.tagger = full_tagger
    elif isinstance(ntagger, six.string_types):
        # load function
        obs_mode.tagger = import_object(ntagger)
    else:
        raise TypeError('tagger must be None, a list or a string')

    return obs_mode


def load_mode_builder(obs_mode, node):
    """Load observing mode OB builder"""

    # Check 'builder' and 'builder_options'
    nval1 = node.get('builder')

    if nval1 is not None:
        if isinstance(nval1, str):
            # override method
            newmethod = import_object(nval1)
            obs_mode.build_ob = newmethod.__get__(obs_mode)
        else:
            raise TypeError('builder must be None or a string')
    else:
        nval2 = node.get('builder_options')

        if nval2 is not None:
            if isinstance(nval2, list):
                for opt_dict in nval2:

                    if 'result_of' in opt_dict:
                        fields = opt_dict['result_of']
                        obs_mode.build_ob_options = ResultOf(**fields)
                        break
            else:
                raise TypeError('builder_options must be None or a list')

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


def load_pipelines(instrument, node):
    keys = ['default']
    check_section(node, 'pipelines', keys=keys)

    pipelines = {}
    for key in node:
        pipelines[key] = load_pipeline(instrument, key, node[key])
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
        ins_tagger = None

    values = node['values']
    confs = {}
    if values:
        for uuid in values:
            confs[uuid] = confclass(uuid)
    else:
        confs['default'] = InstrumentConfiguration('EMPTY')
        default_entry = 'default'

    if default_entry:
        confs['default'] = confs[default_entry]
    else:
        if 'default' not in confs:
            # Choose the first if is not already defined
            confs['default'] = confs[values[0]]
    return confs, ins_tagger


def load_pipeline(instrument, name, node):

    keys = ['recipes', 'version']
    check_section(node, 'pipeline', keys=keys)

    recipes = load_base("recipes", node['recipes'])
    if 'products' in node:
        products = load_base("products", node['products'])
    else:
        products = {}
    if 'provides' in node:
        provides = load_prods(node['provides'], recipes.keys())
    else:
        provides = []
    version = node['version']
    return Pipeline(instrument, name, recipes, version=version,
                    products=products, provides=provides)


def load_recipe(name, node):

    recipe = {'class': ''}

    keys =  ['class']
    if isinstance(node, dict):
        check_section(node, name, keys=keys)
        recipe = node
    else:
        recipe['class'] = node
    if 'args' in recipe:
        recipe['args'] = tuple(recipe['args'])
    return recipe


def load_base(name, node):

    #keys = ['recipes', 'version']
    #check_section(node, 'pipeline', keys=keys)
    recipes = {}
    for key in node:
        recipes[key] = load_recipe(key, node[key])
    return recipes


def load_prods(node, allmodes):
    result = []
    for entry in node:
        name = entry['name']
        mode_name = entry['mode']
        field = entry['field']
        for mode_key in allmodes:
            if mode_key == mode_name:
                prod = ProductEntry(name, mode_key, field)
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

    trans = {'name': node['name']}
    if 'datamodel' in node:
        trans['datamodel'] = import_object(node['datamodel'])
    else:
        trans['datamodel'] = None
    trans['pipelines'] = load_pipelines(node['name'], pipe_node)
    trans['modes'] = load_modes(mode_node)
    confs, custom_selector = load_confs(package, conf_node, confclass=confclass)
    trans['configurations'] = confs
    ins = InstrumentDRP(**trans)
    # add bound method
    if custom_selector:
        ins.select_configuration = custom_selector.__get__(ins)
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
    mm.date_start = convert.convert_date(contents['date_start'])
    mm.date_end = convert.convert_date(contents['date_end'])
    mm.components = {}
    for cname, cuuid in contents['components'].items():
        fcomp = loader.build_component_fp(cuuid)
        rr = load_cc_from_file(fcomp, loader=loader)
        mm.components[cname] = rr
    return mm
