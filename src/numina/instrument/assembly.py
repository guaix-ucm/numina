#
# Copyright 2019-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import collections
import itertools
import json
import pathlib

import attrs
from dateutil.parser import isoparse
import importlib_resources

import numina.util.objimport
import numina.instrument.configorigin as cf
import numina.instrument.generic
# from numina.core.instrument.configorigin import ElementOrigin
# import numina.core.instrument.generic as repre


def get_default_class(etype):
    """
    Transform the name of the component into its class

    Parameters
    ----------
    etype : {'instrument', 'component', 'setup'. 'properties'}
        Named type of component

    Returns
    -------
    ComponentGeneric or InstrumentGeneric or SetupGeneric or PropertiesGeneric

    """
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
        raise ValueError(f'no class for {etype}')


_default_class = {}
_default_class['instrument'] = numina.instrument.generic.InstrumentGeneric
_default_class['component'] = numina.instrument.generic.InstrumentGeneric
_default_class['setup'] = numina.instrument.generic.InstrumentGeneric
_default_class['properties'] = numina.instrument.generic.InstrumentGeneric

@attrs.define
class ComponentCollection:
    dirname = attrs.field()
    paths: attrs.field()


def load_resources_pkg(pkgname: str, configs: str) -> ComponentCollection:
    """
    Gather the path of the components

    Parameters
    ----------
    pkgname : str
    configs : str

    Returns
    -------
    ComponentCollection
        Description of the instrument components
    """

    valid_paths = []

    for res in importlib_resources.files('{}.{}'.format(pkgname, configs)).iterdir():
        if importlib_resources.as_file(res).suffix == '.json':
            valid_paths.append(res.name)

    dirpath = importlib_resources.files(pkgname) / configs
    return ComponentCollection(dirpath, valid_paths)


def load_resources_dir(dirname):
    raise NotImplementedError
    # return ComponentCollection(dirname, valid_paths)


def load_comp_store(comp_collection: ComponentCollection) -> dict:
    """

    Parameters
    ----------
    comp_collection : ComponentCollection

    Returns
    -------
    dict
    """
    comp_store = {}
    for entry in comp_collection.dirname.iterfiles():
        if entry.name in comp_collection.paths:
            with open(entry) as fd:
                cont = json.load(fd)
                cont['origin'] = cf.ElementOrigin.create_from_dict(cont)
                comp_store[entry.name] = cont
    return comp_store


def load_panoply_store(sys_drps=None, defpath=None) -> dict:
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


def load_paths_store(pkg_paths=None, file_paths=None) -> dict:
    """

    Parameters
    ----------
    pkg_paths
    file_paths

    Returns
    -------
    dict
    """

    comp_store = {}
    # Prepare file paths
    if file_paths is None:
        file_paths = []
    if pkg_paths is None:
        pkg_paths = []

    paths1 = [pathlib.Path(fpath) for fpath in file_paths]
    paths2 = [importlib_resources.files(ppath) for ppath in pkg_paths]

    for path in itertools.chain(paths1, paths2):
        for obj in path.iterdir():
            if obj.suffix == '.json':
                with open(obj) as fd:
                    cont = json.load(fd)
                    cont['origin'] = cf.ElementOrigin.create_from_dict(cont)
                    comp_store[obj.name] = cont

    return comp_store


def find_instrument(comp_store, name, date):
    """
    Find instrument in the component collection

    Raises
    ------
    ValueError
        If there is no instrument
    """
    return find_element(comp_store, 'instrument', name, date)


def find_element(comp_store, etype, keyval, date, by_key='name'):
    """
    Find component in the component collection

    Raises
    ------
    ValueError
        If there is no component
    """
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
        raise ValueError(
            f"Not found {etype} {by_key}={keyval} for date={date}")


def assembly_instrument(comp_store, keyval, date, by_key='name'):
    """
    Assembly an instrument configuration object.

    Create an instrument object from a store of configurations using
    either the UUID of the configuration or the date of the configuration

    Parameters
    ----------
    comp_store : dict
    keyval : str
    date : str or datetime
    by_key : str, optional

    Returns
    -------
    InstrumentGeneric
        a instrument configuration
    """
    return assembly_element(comp_store, 'instrument', keyval, date, by_key=by_key)


def assembly_element(comp_store, etype, tmpl, date, dest=None, by_key='name'):
    """

    Parameters
    ----------
    comp_store : dict
    etype
    tmpl
    date : str or datetime
    dest
    by_key : str, optional

    Returns
    -------
    ComponentGeneric or InstrumentGeneric or SetupGeneric or PropertiesGeneric

    Raises
    ------
    ValueError is there is a problem during element construction
    """
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
            res = assembly_element(comp_store, 'setup',
                                   c['name'], date, dest=cid)
            setup_objects[res.name] = res
        elif 'uuid' in c:
            res = assembly_element(comp_store, 'setup', c['uuid'], date, dest=cid,
                                   by_key='uuid')
            setup_objects[res.name] = res
        else:
            raise ValueError(f'error in setup: {c}')

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
                res = assembly_element(
                    comp_store, 'properties', entry['name'], date)
                confe = res.properties[key]
            elif 'uuid' in entry:

                res = assembly_element(comp_store, 'properties', entry['uuid'], date,
                                       by_key='uuid')
                confe = res.properties[key]
            else:
                raise ValueError(f'error in properties: {entry}')

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
            res = assembly_element(
                comp_store, 'component', c['name'], date, dest=cid)
        elif 'uuid' in c:
            res = assembly_element(comp_store, 'component', c['uuid'], date, dest=cid,
                                   by_key='uuid')
        else:
            raise ValueError(f'error in components: {c}')

        res.set_parent(iml)

    return iml
