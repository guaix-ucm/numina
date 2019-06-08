#
# Copyright 2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import os.path
import json
import collections

import numina.util.objimport

import pkg_resources

# Try to use isoparse ISO-8601, if not available
# use generic parser
import dateutil.parser

try:
    isoparse = dateutil.parser.isoparse
except AttributeError:
    isoparse = dateutil.parser.parse

import numina.instrument.configorigin as cf
# from numina.core.instrument.configorigin import ElementOrigin
# import numina.core.instrument.generic as repre


def get_default_class(etype):
    import numina.instrument.generic as repre
    if etype == 'instrument':
        return repre.InstrumentGeneric
    elif etype == 'component':
        return repre.ComponentGeneric
    elif etype == 'setup':
        return repre.SetupGeneric
    elif etype == 'properties':
        return repre.PropertiesGeneric
    else:
        raise ValueError('no class for {}'.format(etype))


ComponentCollection = collections.namedtuple('ComponentCollection', 'dirname paths')


def load_resources_pkg(pkgname, configs):

    valid_paths = []

    for res in pkg_resources.resource_listdir(pkgname, configs):
        if res.endswith('.json'):
            valid_paths.append(res)
    fname = pkg_resources.resource_filename(pkgname, configs)
    return ComponentCollection(fname, valid_paths)


def load_resources_dir(dirname):

    valid_paths = []
    raise NotImplementedError
    # return ComponentCollection(dirname, valid_paths)


def load_comp_store(comp_collection):

    comp_store = {}
    for p in comp_collection.paths:
        with open(os.path.join(comp_collection.dirname, p)) as fd:
            cont = json.load(fd)
            cont['origin'] = cf.ElementOrigin.create_from_dict(cont)
            comp_store[p] = cont
    return comp_store


def load_panoply_store(sys_drps=None, defpath=None):
    if defpath is None:
        file_paths = []
    else:
        file_paths = [defpath]

    pkg_paths = []
    # DRP paths
    if sys_drps is not None:
        for drp in sys_drps.drps.values():
            pkg_paths.append(drp.profiles)

    return load_paths_store(pkg_paths, file_paths)


def load_paths_store(pkg_paths=None, file_paths=None):
    import os
    comp_store = {}

    elements = []

    # Prepare file paths
    if file_paths is None:
        file_paths = []
    if pkg_paths is None:
        pkg_paths = []

    for fpath in file_paths:
        elements.append((fpath, os.listdir, (fpath, )))

    # DRP paths
    for ppath in pkg_paths:
        splitp = ppath.split('.')
        pkgname = '.'.join(splitp[:-1])
        configs = splitp[-1]
        dirname = pkg_resources.resource_filename(pkgname, configs)
        elements.append((dirname, pkg_resources.resource_listdir, (pkgname, configs)))

    # Load everything
    for dirname, func_resource, args in elements:
        for res in func_resource(*args):
            if res.endswith('.json'):
                with open(os.path.join(dirname, res)) as fd:
                    cont = json.load(fd)

                    cont['origin'] = cf.ElementOrigin.create_from_dict(cont)
                    comp_store[res] = cont

    return comp_store


def find_instrument(comp_store, name, date):
    return find_element(comp_store, 'instrument', name, date)


def find_element(comp_store, etype, keyval, date, by_key='name'):

    if isinstance(date, str):
        datet = isoparse(date)
    else:
        datet = date

    for key, val in comp_store.items():
        if (keyval == val[by_key]) and (val['type'] == etype):
            if val['origin'].is_valid_date(datet):
                return val
            else:
                # print('date not valid', datet, val['origin'].date_start, val['origin'].date_end)
                pass
    else:
        raise ValueError("Not found {} {}={} for date={}".format(etype, by_key, keyval, date))


def assembly_instrument(comp_store, keyval, date, by_key='name'):
    return assembly_element(comp_store, 'instrument', keyval, date, by_key=by_key)


def assembly_element(comp_store, etype, tmpl, date, dest=None, by_key='name'):
    import numina.instrument.generic as repre

    somed = find_element(comp_store, etype, tmpl, date, by_key=by_key)

    if dest is None:
        dest_name = somed['name']
    else:
        dest_name = dest

    if 'class' in somed:
        class_name = somed['class']
        Klss = numina.util.objimport.import_object(class_name)
    else:
        Klss = get_default_class(etype)

    # Load setup objects
    setup_objects = {}
    for c in somed.get('setup', []):
        cid = c.get("id")
        if 'name' in c:
            res = assembly_element(comp_store, 'setup', c['name'], date, dest=cid)
            setup_objects[res.name] = res
        elif 'uuid' in c:
            res = assembly_element(comp_store, 'setup', c['uuid'], date, dest=cid,
                                   by_key='uuid')
            setup_objects[res.name] = res
        else:
            raise ValueError('error in setup: {}'.format(c))

    # Load config objects
    prop_objects = {}
    confs = somed.get('properties', [])
    for entry in confs:
        # Keep old version
        if isinstance(entry, str):
            key = entry
            base = confs[key]
            values = base['values']
            depends = base['depends']
            confe = repre.PropertyEntry(values, depends)
            prop_objects[entry] = confe
        else:
            key = entry['id']
            if 'values' in entry:
                values = entry['values']
                depends = entry['depends']
                confe = repre.PropertyEntry(values, depends)
            elif 'name' in entry:
                res = assembly_element(comp_store, 'properties', entry['name'], date)
                confe = res.properties[key]
            elif 'uuid' in entry:

                res = assembly_element(comp_store, 'properties', entry['uuid'], date,
                                       by_key='uuid')
                confe = res.properties[key]
            else:
                raise ValueError('error in properties: {}'.format(entry))

        prop_objects[key] = confe

    if hasattr(Klss, 'init_args'):
        b_args, b_kwds = Klss.init_args(dest_name, setup_objects)
    else:
        b_kwds = {}
        if etype in ['component', 'instrument', 'properties']:
            b_kwds['properties'] = prop_objects
        b_kwds.update(setup_objects)
        b_args = (dest_name,)

    iml = Klss(*b_args, **b_kwds)

    theorigin = somed['origin']
    iml.set_origin(theorigin)

    # For setup classes
    if 'values' in somed:
        iml.values = somed['values']

    # Add components when device is initialised
    for c in somed.get('components', []):
        # Use 'id' as name, if present
        cid = c.get("id")
        if 'name' in c:
            res = assembly_element(comp_store, 'component', c['name'], date, dest=cid)
        elif 'uuid' in c:
            res = assembly_element(comp_store, 'component', c['uuid'], date, dest=cid,
                                   by_key='uuid')
        else:
            raise ValueError('error in components: {}'.format(c))

        res.set_parent(iml)

    return iml
