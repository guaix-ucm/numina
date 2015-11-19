#
# Copyright 2011-2015 Universidad Complutense de Madrid
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

'''Build a LoadableDRP from a yaml file'''

import pkgutil
import yaml

from .objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import InstrumentDRP
from .pipeline import InstrumentConfiguration
from .taggers import get_tags_from_full_ob


def drp_load(package, resource):
    """Load the DRPS from a resource file."""
    data = pkgutil.get_data(package, resource)
    return drp_load_data(data)


def drp_load_data(data):
    """Load the DRPS from data."""
    drpdict = yaml.load(data)
    ins = load_instrument(drpdict)
    return ins


def load_modes(node):
    modes = list()
    for child in node:
        modes.append(load_mode(child))
    return modes


def load_mode(node):
    obs_mode = ObservingMode()
    # handle tagger:
    obs_mode.__dict__.update(node)

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


def load_pipelines(node):
    keys = ['default']
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r in pipelines node', key)
    pipelines = {}
    for key in node:
        pipelines[key] = load_pipeline(key, node[key])
    return pipelines


def load_confs(node):
    keys = ['default']
    for key in keys:
        if key not in node:
            raise ValueError('Missing key %r in configurations node', key)
    confs = {}
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
    prod_node = node.get('products', [])

    trans = {'name': node['name']}
    trans['pipelines'] = load_pipelines(pipe_node)
    trans['modes'] = load_modes(mode_node)
    trans['configurations'] = load_confs(conf_node)
    trans['products'] = prod_node
    return InstrumentDRP(**trans)
