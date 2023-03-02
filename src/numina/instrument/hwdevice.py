#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""A representation of the a hardware device"""

import inspect

from .device import DeviceBase


class HWDevice(DeviceBase):
    def __init__(self, name, origin=None, parent=None):
        super(HWDevice, self).__init__(
            name, origin=origin, parent=parent
        )

    def config_info(self):
        return visit(self)

    def get_properties(self):
        meta = self.init_config_info()
        for key, prop in inspect.getmembers(self.__class__):
            if isinstance(prop, property):
                try:
                    meta[key] = getattr(self, key).value
                except:
                    meta[key] = getattr(self, key)
        return meta

    def init_config_info(self):
        return dict(name=self.name)

    def end_config_info(self, meta):
        if self.children:
            meta['children'] = self.children.keys()
        return meta

    def configure_me(self, value):
        for key in value:
            setattr(self, key, value[key])

    def configure(self, info):
        for key, value in info.items():
            node = self.get_device(key)
            if node:
                node.configure_me(value)


def visit(node, root='', meta=None):
    sep = '.'
    if meta is None:
        meta = {}

    if node.name is None:
        base = 'unknown'
    else:
        base = node.name

    if root != '':
        node_name = root + sep + base
    else:
        node_name = base

    meta[node_name] = node.get_properties()
    submeta = meta
    for child in node.children.values():
        visit(child, root=node_name, meta=submeta)
    return meta
