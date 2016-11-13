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

from .objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import InstrumentDRP
from .pipeline import InstrumentConfiguration
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
    return drp_load_data(data, confclass=confclass)


def drp_load_data(data, confclass=None):
    """Load the DRPS from data."""
    drpdict = yaml.load(data)
    ins = load_instrument(drpdict, confclass=confclass)
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


def load_confs(node, confclass=None):
    keys = ['values']
    check_section(node, 'configurations', keys=keys)

    if confclass is None:
        confclass = InstrumentConfiguration

    default_entry = node.get('default')
    tagger = node.get('tagger')
    if tagger:
        ins_tagger = import_object(tagger)
    else:
        ins_tagger = lambda obsres: 'default'

    values = node['values']
    confs = {}
    for uuid in values:
        confs[uuid] = confclass(uuid, uuid)
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


def load_instrument(node, confclass=None):
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
    confs, selector = load_confs(conf_node, confclass=confclass)
    trans['configurations'] = confs
    trans['products'] = prod_node
    ins = InstrumentDRP(**trans)
    ins.selector = selector
    return ins
