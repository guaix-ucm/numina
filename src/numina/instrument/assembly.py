#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from datetime import datetime
import itertools
import json
import pathlib
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

import attrs
from dateutil.parser import isoparse
import importlib_resources
import numina.util.objimport

from .elements import SetupBlock, PropertiesBlock, ElementEnum
from .generic import CG
from .configorigin import ElementOrigin
from .property import (
    PropertyEntry,
    PropertyProxy,
    PropertyModOneOf,
    PropertyModLimits,
    PropertyBase,
)
from ..keydef import KeyDefinition


@attrs.define
class ComponentCollection:
    dirname = attrs.field()
    paths = attrs.field()


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
                cont["origin"] = ElementOrigin.from_dict(cont)
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
            if obj.suffix == ".json":
                with open(obj) as fd:
                    cont = json.load(fd)
                    cont["origin"] = ElementOrigin.from_dict(cont)
                    comp_store[obj.name] = cont

    return comp_store


def find_instrument(
    comp_store, keyval: str, date: str | datetime, by_key="name"
) -> dict[str, Any]:
    """
    Find instrument in the component collection

    Raises
    ------
    ValueError
        If there is no instrument
    """
    return find_element(
        comp_store, ElementEnum.ELEM_INSTRUMENT, keyval, date, by_key=by_key
    )


def find_element(
    comp_store, etype: ElementEnum, keyval: str, date: str | datetime, by_key="name"
) -> dict[str, Any]:
    """
    Find component of the given type in the component collection

    Raises
    ------
    ValueError
        If there is no component
    """
    if isinstance(date, str):
        datet = isoparse(date)
    else:
        datet = date

    element_name = ElementEnum.to_str(etype)

    for key, val in comp_store.items():
        if (keyval == val[by_key]) and (val["type"] == element_name):
            if val["origin"].is_valid_date(datet):
                return val
            else:
                # print('date not valid', datet, val['origin'].date_start, val['origin'].date_end)
                pass
    else:
        raise ValueError(f"Not found {element_name} {by_key}={keyval} for date={date}")


def assembly_instrument(
    comp_store, keyval: str, date: str | datetime, by_key: str = "name"
) -> CG:
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
        an instrument configuration
    """
    return assembly_element(
        comp_store, ElementEnum.ELEM_INSTRUMENT, keyval, date, by_key=by_key
    )


def assembly_component(
    comp_store, keyval: str, date: str | datetime, dest=None, by_key="name"
) -> CG:
    """
    Assembly an instrument configuration object.

    Create an instrument object from a store of configurations using
    either the UUID of the configuration or the date of the configuration

    Parameters
    ----------
    comp_store : dict
    keyval : str
    date : str or datetime
    dest: str, optional
    by_key : str, optional

    Returns
    -------
    ComponentGeneric
        an instrument configuration
    """
    return assembly_element(
        comp_store, ElementEnum.ELEM_COMPONENT, keyval, date, dest=dest, by_key=by_key
    )


def assembly_property(
    comp_store, keyval: str, date: str | datetime, dest=None, by_key="name"
) -> PropertiesBlock:
    return assembly_element(
        comp_store, ElementEnum.ELEM_PROPERTIES, keyval, date, dest=dest, by_key=by_key
    )


def assembly_setup(
    comp_store, keyval: str, date: str | datetime, dest: str = None, by_key="name"
) -> SetupBlock:
    return assembly_element(
        comp_store, ElementEnum.ELEM_SETUP, keyval, date, dest=dest, by_key=by_key
    )


def assembly_element(
    comp_store,
    etype: ElementEnum,
    tmpl: str,
    date: str | datetime,
    dest: str | None = None,
    by_key="name",
):
    """
    Assembly a given element in the object.
    Parameters
    ----------
    comp_store : dict
    etype: ElementEnum
    tmpl
    date : str or datetime
    dest
    by_key : str, optional

    Returns
    -------
    ComponentGeneric or SetupGeneric or PropertiesGeneric

    Raises
    ------
    ValueError is there is a problem during element construction
    """

    somed = find_element(comp_store, etype, tmpl, date, by_key=by_key)

    if dest is None:
        dest_name = somed["name"]
    else:
        dest_name = dest
    # print('dest', dest_name, somed['name'], somed['uuid'])
    if "class" in somed:
        class_name = somed["class"]
        clss = numina.util.objimport.import_object(class_name)
    else:
        class_name = somed["name"]
        clss = ElementEnum.to_class(etype)

    setup_block = somed.get("setup", [])
    setup_obj = process_setup(comp_store, somed["name"], setup_block, date)

    prop_block = somed.get("properties", [])
    prop_objects = process_properties(comp_store, prop_block, date)
    # print('prop', prop_objects)

    iml = clss.from_component(
        class_name,
        dest_name,
        origin=somed["origin"],
        properties=prop_objects,
        setup=setup_obj,
    )
    # if len(setup_obj.values) > 0:
    #    print(setup_block, Klss)
    comp_block = somed.get("components", [])
    comp_objects = process_components(comp_store, comp_block, date)

    # print('comp', comp_objects)
    for comp in comp_objects:
        comp.set_parent(iml)

    return iml


def process_setup(
    comp_store,
    setup_id: str,
    setup_block: "Iterable[dict[str, Any]]",
    date: str | datetime,
) -> SetupBlock:
    setup_objects = {}
    for entry in setup_block:
        match entry:
            case {"values": values}:
                cid = entry.get("id", "values")
                setup_objects[cid] = values
            case {"name": name, "id": cid}:
                res = assembly_setup(comp_store, name, date, dest=cid, by_key="name")
                setup_objects[res.name] = res
            case {"uuid": name, "id": cid}:
                res = assembly_setup(comp_store, name, date, dest=cid, by_key="uuid")
                setup_objects[res.name] = res

            case _:
                raise ValueError(f"error in setup: {entry}")

    setup_obj = SetupBlock(setup_id)
    for key, val in setup_objects.items():
        if isinstance(val, SetupBlock):
            setup_obj.values.update(val.values)
        else:
            setup_obj.values[key] = val

    return setup_obj


def process_properties(
    comp_store, prop_block, date: str | datetime
) -> dict[str, PropertyBase]:
    prop_objects: dict[str, PropertyBase] = {}
    for entry in prop_block:
        key = entry["id"]
        match entry:
            case {"values": values, "depends": depends}:
                confe = PropertyEntry(values, depends)
            case {"one_of": one_of, **rest}:
                if len(one_of) < 1:
                    raise ValueError(f'"one_of must have length >= 1: {entry}')
                default = rest.get("default", one_of[0])
                # FIXME: default is stored twice
                mapping = process_key_def(rest, default=default)
                confe = PropertyModOneOf(one_of, default=default, mapping=mapping)
            case {"limits": limits, **rest}:
                default = rest.get("default", 0)
                # FIXME: default is stored twice
                mapping = process_key_def(rest, default=default)
                confe = PropertyModLimits(limits, default=default, mapping=mapping)
            case {"references": ref_name}:
                confe = PropertyProxy(ref_name)
            case {"name": e_name}:
                res = assembly_property(comp_store, e_name, date)
                confe = res.properties[key]
            case {"uuid": e_uuid}:
                res = assembly_property(comp_store, e_uuid, date, by_key="uuid")
                confe = res.properties[key]
            case _:
                raise ValueError(f"error in properties: {entry}")

        prop_objects[key] = confe
    return prop_objects


def process_components(comp_store, comp_block, date: str | datetime) -> list[CG]:
    # Add components when device is initialised
    # print('load components')
    comp_objects = []
    for entry in comp_block:
        # Use 'id' as name, if present
        cid = entry.get("id")
        if "name" in entry:
            key = "name"
        elif "uuid" in entry:
            key = "uuid"
        else:
            raise ValueError(f"error in components: {entry}")

        res = assembly_component(comp_store, entry[key], date, dest=cid, by_key=key)
        comp_objects.append(res)
    return comp_objects


def process_key_def(rest: dict[str, Any], default) -> KeyDefinition | None:
    key = rest.get("key")
    ext = rest.get("ext")
    if key is not None:
        return KeyDefinition(key, ext=ext, default=default)
    return None
