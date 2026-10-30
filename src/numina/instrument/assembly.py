#
# Copyright 2019-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from datetime import datetime
import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from dateutil.parser import isoparse
import numina.util.objimport

from .collection import load_paths_store
from .elements import SetupBlock, PropertiesBlock, ElementEnum
from .generic import CG
from .property import (
    PropertyEntry,
    PropertyProxy,
    PropertyModOneOf,
    PropertyModLimits,
    PropertyBase,
)
from ..keydef import KeyDefinition

_logger = logging.getLogger(__name__)


def load_panoply_store(sys_drps=None, defpath=None) -> dict:
    """Load the instrument configurations of the DRPs and of an additional directory.

    Parameters
    ----------
    sys_drps : optional
        Loaded DRPs, an object with a ``drps`` mapping. The configurations are read
        from the ``profiles`` package of each DRP.
    defpath : str or path-like, optional
        Directory with additional configuration files.

    Returns
    -------
    dict
        The configurations, by file name, as returned by ``load_paths_store``.
    """
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


def find_instrument(comp_store, keyval: str, date: str | datetime, by_key="name") -> dict[str, Any]:
    """
    Find instrument in the component collection

    Raises
    ------
    ValueError
        If there is no instrument
    """
    return find_element(comp_store, ElementEnum.ELEM_INSTRUMENT, keyval, date, by_key=by_key)


def find_element(comp_store, etype: ElementEnum, keyval: str, date: str | datetime, by_key="name") -> dict[str, Any]:
    """
    Find component of the given type in the component collection

    If date is None and there are several components,
    the one with the most recent start date is used.
    If there are several components valid for the date,
    the first one in the collection is used, with a warning.

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

    candidates = [
        val
        for val in comp_store.values()
        if (keyval == val[by_key]) and (val["type"] == element_name) and val["origin"].is_valid_date(datet)
    ]
    if not candidates:
        raise ValueError(f"Not found {element_name} {by_key}={keyval} for date={date}")

    if datet is None and len(candidates) > 1:
        # Without date, all the components are valid, use the most recent
        selected = max(candidates, key=lambda val: val["origin"].date_start or datetime.min)
        _logger.warning(
            "no date to select the configuration of %s %s=%s, using the most recent: uuid=%s",
            element_name,
            by_key,
            keyval,
            selected["origin"].uuid,
        )
        return selected

    selected = candidates[0]
    if len(candidates) > 1:
        _logger.warning(
            "%d configurations of %s %s=%s are valid for date=%s (uuids %s), using uuid=%s",
            len(candidates),
            element_name,
            by_key,
            keyval,
            date,
            ", ".join(str(val["origin"].uuid) for val in candidates),
            selected["origin"].uuid,
        )
    return selected


def assembly_instrument(comp_store, keyval: str, date: str | datetime, by_key: str = "name") -> CG:
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
    return assembly_element(comp_store, ElementEnum.ELEM_INSTRUMENT, keyval, date, by_key=by_key)


def configure_obresult(obsres, drp, comp_store=None) -> CG:
    """Select, assembly and configure the instrument configuration of obsres

    The configuration is selected by drp.select_profile, with the profile
    of obsres, if it is a UUID, or with the first frame of obsres. It is
    configured with the first frame of obsres. The configuration is
    stored in obsres.configuration, and its UUID in obsres.profile.

    Parameters
    ----------
    obsres : ObservationResult
    drp : InstrumentDRP
        The DRP of the instrument of obsres
    comp_store : dict, optional
        The components of the configurations, by default those of the
        profiles of drp

    Returns
    -------
    InstrumentGeneric
        the instrument configuration

    Raises
    ------
    ValueError
        If the configuration is not for the instrument of obsres
    """
    if comp_store is None:
        comp_store = load_paths_store([drp.profiles])

    key, date_obs, keyname = drp.select_profile(obsres)
    configuration = assembly_instrument(comp_store, key, date_obs, by_key=keyname)
    if configuration.name != obsres.instrument:
        msg = (
            f"instrument configuration uuid={configuration.origin.uuid} is for "
            f"instrument '{configuration.name}', not '{obsres.instrument}'"
        )
        raise ValueError(msg)
    obsres.configuration = configuration
    obsres.profile = str(configuration.origin.uuid)
    _logger.debug("instrument profile is %s", obsres.profile)

    sample_frame = obsres.get_sample_frame()
    if sample_frame is not None:
        _logger.debug("configuring instrument model with image from obsres")
        img = sample_frame.open()
        try:
            configuration.configure_with_image(img)
        finally:
            # Close the file only if it was opened here
            if sample_frame.frame is None:
                img.close()
    else:
        _logger.debug("no configuring instrument model")
    return configuration


def assembly_component(comp_store, keyval: str, date: str | datetime, dest=None, by_key="name") -> CG:
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
    return assembly_element(comp_store, ElementEnum.ELEM_COMPONENT, keyval, date, dest=dest, by_key=by_key)


def assembly_property(comp_store, keyval: str, date: str | datetime, dest=None, by_key="name") -> PropertiesBlock:
    """Assemble the properties element whose ``by_key`` is ``keyval``, valid for ``date``."""
    return assembly_element(comp_store, ElementEnum.ELEM_PROPERTIES, keyval, date, dest=dest, by_key=by_key)


def assembly_setup(comp_store, keyval: str, date: str | datetime, dest: str = None, by_key="name") -> SetupBlock:
    """Assemble the setup element whose ``by_key`` is ``keyval``, valid for ``date``."""
    return assembly_element(comp_store, ElementEnum.ELEM_SETUP, keyval, date, dest=dest, by_key=by_key)


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
    """Build the setup block ``setup_id`` from the entries of a ``setup`` section.

    An entry with ``values`` stores them under its ``id`` (``"values"`` by default). An entry
    with ``name`` or ``uuid`` references a setup element, whose values are merged in the block.

    Raises
    ------
    ValueError
        If an entry has none of the expected forms.
    """
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


def process_properties(comp_store, prop_block, date: str | datetime) -> dict[str, PropertyBase]:
    """Build the property descriptors of the entries of a ``properties`` section.

    The keys of each entry select the descriptor: ``values`` and ``depends`` give a
    ``PropertyEntry``, ``one_of`` a ``PropertyModOneOf``, ``limits`` a ``PropertyModLimits``
    and ``references`` a ``PropertyProxy``. An entry with ``name`` or ``uuid`` takes the
    property with the same ``id`` from that properties element.

    Returns
    -------
    dict
        The descriptors, by the ``id`` of each entry.

    Raises
    ------
    ValueError
        If an entry has none of the expected forms, or its ``one_of`` list is empty.
    """
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
    """Assemble the components of the entries of a ``components`` section.

    Each entry references a component element by ``name`` or ``uuid``. Its ``id``, if present,
    is the name of the device.

    Raises
    ------
    ValueError
        If an entry has neither ``name`` nor ``uuid``.
    """
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
    """Return the ``KeyDefinition`` of the ``key`` and ``ext`` of a property entry, None without ``key``."""
    key = rest.get("key")
    ext = rest.get("ext")
    if key is not None:
        return KeyDefinition(key, ext=ext, default=default)
    return None
