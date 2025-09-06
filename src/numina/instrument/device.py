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
        obj = cls.__new__(cls)
        obj.__init__(comp_id, origin, parent)
        return obj

    def set_origin(self, origin: "ElementOrigin") -> None:
        self.origin = origin
        if self.origin is not None:
            self.uuid = self.origin.uuid

    def set_parent(self, new_parent: "DeviceBase") -> None:
        if self.parent:
            del self.parent.children[self.name]
        if new_parent:
            if self.name in new_parent.children:
                raise ValueError(f"{self.name} already registered with new_parent")
            else:
                new_parent.children[self.name] = self
                self.parent = new_parent

    def get_device(self, path: str) -> "DeviceBase":
        vals = path.split(".")
        return self.get_device_seq(vals)

    def get_device_seq(self, path: list[str]) -> "DeviceBase":

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
        keys = path.split(".")
        device_path = keys[:-1]
        property_name = keys[-1]
        return device_path, property_name

    def get_property(self, path: str):
        device_path, prop_name = self.split_path(path)
        dev = self.get_device_seq(device_path)
        val = getattr(dev, prop_name)
        return val

    def get_value(self, path: str, **state):
        # Ignoring state
        return self.get_property(path)

    def configure_me(self, info: dict) -> None:
        with self.state.managed_configure():
            for key in info:
                setattr(self, key, info[key])

    def configure(self, info: dict) -> None:
        for key, value in info.items():
            node = self.get_device(key)
            if node:
                node.configure_me(value)

    def configure_me_with_header(self, hdr) -> None:
        with self.state.managed_configure():
            # To be implemented...
            pass

    def configure_with_header(self, hdr) -> None:
        self.configure_me_with_header(hdr)
        for ch in self.children.values():
            ch.configure_with_header(hdr)

    def configure_me_with_image(self, image) -> None:
        # To be implemented...
        self.configure_me_with_header(image[0].header)

    def configure_with_image(self, image):
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
        return set([])

    def get_property_names(self) -> set[str]:
        return set([])

    def init_config_info(self) -> dict[str, typing.Any]:
        return dict()

    def config_info(self):
        return visit(self)

    def end_config_info(self, result) -> dict[str, typing.Any]:
        if self.children:
            result["children"] = list(self.children.keys())
        return result

    def get_properties(self, init=True) -> dict[str, typing.Any]:
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
