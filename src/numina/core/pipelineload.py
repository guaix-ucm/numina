#
# Copyright 2011-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Build a LoadableDRP from a yaml file"""

import pkgutil
import importlib
from io import StringIO

import yaml

from numina.util.objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import InstrumentDRP
# from .instrument.insconf import InstrumentConfiguration, instrument_loader
from .pipeline import ProductEntry
from .query import ResultOf
from .taggers import get_tags_from_full_ob


def check_section(node, section, keys=None):
    """Validate keys in a section"""
    if keys:
        for key in keys:
            if key not in node:
                raise ValueError(f'Missing key {key!r} inside {section!r} node')


def drp_load(package, resource, confclass=None):
    """Load the DRPS from a resource file."""
    data = pkgutil.get_data(package, resource)
    return drp_load_data(package, data, confclass=confclass)


def drp_load_data(package, data, confclass=None):
    """Load the DRPS from data."""
    drpdict = yaml.safe_load(data)
    ins = load_instrument(package, drpdict, confclass=confclass)
    if ins.version == 'undefined':
        pkg = importlib.import_module(package)
        ins.version = getattr(pkg, '__version__', 'undefined')
    return ins


def load_modes(node, confclass=None):
    """Load all observing modes"""
    if isinstance(node, list):
        values = [load_mode(child, confclass) for child in node]
        keys = [mode.key for mode in values]
        return dict(zip(keys, values))
    elif isinstance(node, dict):
        values = {key: load_mode(child) for key, child in node}
        return values
    else:
        raise NotImplementedError


def load_mode(node, confclass=None):
    """Load one observing mdode"""
    obs_mode = ObservingMode()
    if confclass is not None:
        node = confclass.mode_loader(node)
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
    elif isinstance(ntagger, str):
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
    import numina.instrument.assembly as asbl
    keys = []
    check_section(node, 'configurations', keys=keys)

    path = node.get('path')
    if path:
        modpath = path
    else:
        modpath = f"{package}.instrument.configs"

    if confclass is None:
        _loader = DefaultLoader(modpath=modpath)

    tagger = node.get('tagger')
    if tagger:
        ins_tagger = import_object(tagger)
    else:
        ins_tagger = None

    confs = asbl.load_paths_store([modpath])
    return confs, ins_tagger, modpath


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

    keys = ['class']
    if isinstance(node, dict):
        check_section(node, name, keys=keys)
        recipe = node
    else:
        recipe['class'] = node
    if 'args' in recipe:
        recipe['args'] = tuple(recipe['args'])
    if 'links' in recipe:
        recipe['links'] = load_link(recipe['links'])
    return recipe


def load_link(node):
    """Build a ResultOf query from dict"""
    result = {}
    for key, opt_dict in node.items():

        if 'result_of' in opt_dict:
            fields = opt_dict['result_of']
            result[key] = ResultOf(**fields)
        else:
            pass
    return result


def load_base(name, node):

    # keys = ['recipes', 'version']
    # check_section(node, 'pipeline', keys=keys)
    recipes = {}
    for key in node:
        recipes[key] = load_recipe(key, node[key])
    # print(recipes)
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

    trans = {'name': node['name'], 'version': 'undefined'}
    if 'datamodel' in node:
        trans['datamodel'] = import_object(node['datamodel'])
    else:
        trans['datamodel'] = None
    if 'version' in node:
        trans['version'] = node['version']
    trans['pipelines'] = load_pipelines(node['name'], pipe_node)
    trans['modes'] = load_modes(mode_node, confclass)
    confs, custom_selector, modpath = load_confs(package, conf_node, confclass=confclass)
    # trans['configurations'] = confs
    trans['configurations'] = confs
    ins = InstrumentDRP(**trans)
    # idiom to add a bound method
    if custom_selector:
        ins.select_configuration = custom_selector.__get__(ins)
    # Add package name
    ins.package = package
    # Add profile path
    ins.profiles = modpath
    return ins


class DefaultLoader(object):
    def __init__(self, modpath):
        self.modpath = modpath

    def build_component_fp(self, key):
        fname = f'component-{key}.json'
        return self.build_type_fp(fname)

    def build_instrument_fp(self, key):
        fname = f'instrument-{key}.json'
        return self.build_type_fp(fname)

    def build_type_fp(self, fname):
        data = pkgutil.get_data(self.modpath, fname)
        fcomp = StringIO(data.decode('utf-8'))
        return fcomp
