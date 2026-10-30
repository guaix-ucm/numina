#
# Copyright 2016-2035 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import typing

if typing.TYPE_CHECKING:
    from .configorigin import ElementOrigin

from .state import State


class DeviceBase:
    """Base class of the devices, that form a tree of named devices.

    Parameters
    ----------
    name
        Name of the device, its key in the children of its parent.
    origin
        Origin of the configuration of the device, that gives its ``uuid``.
    parent
        Parent device.
    """

    def __init__(
        self,
        name: str,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
    ):
        self.name = name
        self.parent = None
        self.origin = None
        self.uuid = None
        self.children: dict[str, "DeviceBase"] = {}
        self.set_origin(origin)
        self.set_parent(parent)
        self.state = State()
        # FIXME: this can be removed
        self.is_configured = False

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent: "DeviceBase | None" = None,
        properties=None,
        setup=None,
    ) -> "DeviceBase":
        """Create the device from an element of the instrument configurations.

        This is the constructor used by ``numina.instrument.assembly``. ``name`` is the
        ``class`` of the element, or its name if it has none, and ``comp_id`` the name of
        the device. This implementation ignores ``name``, ``properties`` and ``setup``.
        """
        obj = cls.__new__(cls)
        obj.__init__(comp_id, origin, parent)
        return obj

    def set_origin(self, origin: "ElementOrigin") -> None:
        """Set the origin of the configuration of the device, and the ``uuid`` from it."""
        self.origin = origin
        if self.origin is not None:
            self.uuid = self.origin.uuid

    def set_parent(self, new_parent: "DeviceBase") -> None:
        """Remove the device from the children of its parent and add it to those of ``new_parent``.

        Raises
        ------
        ValueError
            If ``new_parent`` already has a child with the name of the device.
        """
        if self.parent:
            del self.parent.children[self.name]
        if new_parent:
            if self.name in new_parent.children:
                raise ValueError(f"{self.name} already registered with new_parent")
            else:
                new_parent.children[self.name] = self
                self.parent = new_parent

    def get_device(self, path: str) -> "DeviceBase":
        """Return the descendant device at ``path``, a dot-separated sequence of names.

        A first name equal to the name of this device is skipped.

        Raises
        ------
        KeyError
            If a device in the path does not exist.
        """
        vals = path.split(".")
        return self.get_device_seq(vals)

    def get_device_seq(self, path: list[str]) -> "DeviceBase":
        """Return the descendant device at ``path``, a sequence of names (see ``get_device``)."""

        if len(path) == 0:
            return self

        base = path[0]
        rpath = path[1:]
        if base == self.name:
            # Skip first term if its myself
            return self.get_device_seq(rpath)

        comp = self.children[base]
        if rpath:
            return comp.get_device_seq(rpath)
        else:
            return comp

    @staticmethod
    def split_path(path: str) -> tuple[list[str], str]:
        """Split a dot-separated ``path`` into the list of device names and the property name."""
        keys = path.split(".")
        device_path = keys[:-1]
        property_name = keys[-1]
        return device_path, property_name

    def get_property(self, path: str):
        """Return the value of the property at ``path``, the device path followed by the property name."""
        device_path, prop_name = self.split_path(path)
        dev = self.get_device_seq(device_path)
        val = getattr(dev, prop_name)
        return val

    def get_value(self, path: str, **state):
        """Return the value of the property at ``path``, ignoring ``state``."""
        # Ignoring state
        return self.get_property(path)

    def configure_me(self, info: dict) -> None:
        """Set the attributes of the device from the mapping ``info``, in the configuring status."""
        with self.state.managed_configure():
            for key in info:
                setattr(self, key, info[key])

    def configure(self, info: dict) -> None:
        """Configure the devices in ``info``, a mapping of device paths to mappings of attributes."""
        for key, value in info.items():
            node = self.get_device(key)
            if node:
                node.configure_me(value)

    def configure_me_with_header(self, hdr) -> None:
        """Configure the device with the FITS header ``hdr``, nothing to do by default."""
        with self.state.managed_configure():
            # To be implemented...
            pass

    def configure_with_header(self, hdr) -> None:
        """Configure the device and its descendants with the FITS header ``hdr``."""
        self.configure_me_with_header(hdr)
        for ch in self.children.values():
            ch.configure_with_header(hdr)

    def configure_me_with_image(self, image) -> None:
        """Configure the device with the header of the primary HDU of ``image``."""
        # To be implemented...
        self.configure_me_with_header(image[0].header)

    def configure_with_image(self, image):
        """Configure the device and its descendants with the FITS ``image``."""
        self.configure_me_with_image(image)
        for ch in self.children.values():
            ch.configure_with_image(image)

    def depends_on(self):
        """Compute the dependencies for me and my children"""
        total_depends = set([])

        total_depends.update(self.my_depends())

        for ch in self.children.values():
            ch_depends = ch.depends_on()
            total_depends.update(ch_depends)
        return total_depends

    def my_depends(self) -> set[str]:
        """Return the dependencies of the device, without those of its children, none by default."""
        return set([])

    def get_property_names(self) -> set[str]:
        """Return the names of the properties included by ``get_properties``, none by default."""
        return set([])

    def init_config_info(self) -> dict[str, typing.Any]:
        """Return the initial dictionary of ``get_properties``, empty by default."""
        return dict()

    def config_info(self):
        """Return the properties of the device and its descendants, by device path (see ``visit``)."""
        return visit(self)

    def end_config_info(self, result) -> dict[str, typing.Any]:
        """Add the names of the children, if any, to ``result`` and return it."""
        if self.children:
            result["children"] = list(self.children.keys())
        return result

    def get_properties(self, init=True) -> dict[str, typing.Any]:
        """Return the properties of the device, by name.

        If ``init`` is True, the dictionary starts with the contents of ``init_config_info``.
        """
        if init:
            result = self.init_config_info()
        else:
            result = dict()

        for key in self.get_property_names():
            result[key] = getattr(self, key)
        return result


def visit(node: DeviceBase, root: str = "", result: dict | None = None, sep: str = "."):
    """Visit recursively all subdevices of node and return their configurations"""

    result = {} if result is None else result

    if root != "":
        node_name = root + sep + node.name
    else:
        node_name = node.name

    result[node_name] = node.get_properties()
    sub_meta = result
    for child in node.children.values():
        visit(child, root=node_name, result=sub_meta)
    return result
