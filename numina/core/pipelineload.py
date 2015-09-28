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

'''Build a LoadableDRP from a yaml file'''

import pkgutil
import uuid
import six
import yaml

from .objimport import import_object
from .pipeline import ObservingMode
from .pipeline import Pipeline
from .pipeline import LoadableDRP
from .pipeline import Instrument
from .pipeline import InstrumentConfiguration


def om_repr(dumper, data):
    return dumper.represent_mapping('!om', data.__dict__)


def om_cons(loader, node):
    om = ObservingMode()
    value = loader.construct_mapping(node)
    om.__dict__.update(value)
    om.uuid = uuid.UUID(om.uuid)
    return om

yaml.add_representer(ObservingMode, om_repr)
yaml.add_constructor('!om', om_cons)


def drp_load(package, resource):
    """Load the DRPS from a resource file."""
    data = pkgutil.get_data(package, resource)
    return drp_load_data(data)



def drp_load_data(data):
    """Load the DRPS from data."""
    ins_all = {}
    for yld in yaml.load_all(data):
        ins = load_instrument(yld)
        ins_all[ins.name] = ins

    return LoadableDRP(ins_all)


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
    return Instrument(**trans)


def print_i(ins):
    six.print_(ins.name)
    print_c(ins.configurations)
    print_m(ins.modes)
    print_p(ins.pipelines)


def print_p(pipelines):
    six.print_('Pipelines')
    for p, n in pipelines.items():
        six.print_(' pipeline', p)
        six.print_('   version', n.version)
        six.print_('   recipes')
        for m, r in n.recipes.items():
            six.print_('    ', m, '->', r)


def print_c(confs):
    six.print_('Configurations')
    for c in confs:
        six.print_(' conf', c, confs[c].values)


def print_m(modes):
    six.print_('Modes')
    for c in modes:
        six.print_(' mode', c.key)


def get_tags_from_full_ob(ob, reqtags=None):
    # each instrument should have one
    # perhaps each mode...
    files = ob.images
    cfiles = ob.children
    alltags = {}

    if reqtags is None:
        reqtags = []

    # Init alltags...
    # Open first image
    if files:
        for fname in files[:1]:
            with fname.open() as fd:
                header = fd[0].header
                for t in reqtags:
                    alltags[t] = header[t]
    else:

        for prod in cfiles[:1]:
            prodtags = prod.tags
            for t in reqtags:
                alltags[t] = prodtags[t]

    for fname in files:
        with fname.open() as fd:
            header = fd[0].header

            for t in reqtags:
                if alltags[t] != header[t]:
                    msg = 'wrong tag %s in file %s' % (t, fname)
                    raise ValueError(msg)

    for prod in cfiles:
        prodtags = prod.tags
        for t in reqtags:
            if alltags[t] != prodtags[t]:
                msg = 'wrong tag %s in product %s' % (t, prod)
                raise ValueError(msg)

    return alltags


def tagger_empty(obsres):
    return get_tags_from_full_ob(obsres, reqtags=[])

def tagger_vph(obsres):
    return get_tags_from_full_ob(obsres, reqtags=['vph'])

