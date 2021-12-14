#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""A representation of the different devices of an instrument"""


class DeviceBase(object):
    def __init__(self, name, origin=None, parent=None):
        self.name = name
        self.parent = None
        self.origin = None
        self.children = {}
        self.set_origin(origin)
        self.set_parent(parent)
        # TODO. Implement a state machine
        self.is_configured = False
        # TODO: Implement equality

    def set_origin(self, origin):
        self.origin = origin

    def set_parent(self, newparent):
        if self.parent:
            del self.parent.children[self.name]
        if newparent:
            if self.name in newparent.children:
                raise ValueError(f'{self.name} already registered with newparent')
            else:
                newparent.children[self.name] = self
                self.parent = newparent

    def get_device(self, path):
        vals = path.split('.')
        return self.get_device_seq(vals)

    def get_device_seq(self, path):

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

    def get_property(self, path):
        keys = path.split('.')
        devce_path = keys[:-1]
        prop = keys[-1]
        dev = self.get_device_seq(devce_path)

        val = getattr(dev, prop)

        return val

    def get_value(self, path, **state):
        # Ignoring state
        return self.get_property(path)

    def configure_me_with_header(self, hdr):
        # To be implemented...
        self.is_configured = True

    def configure_with_header(self, hdr):
        self.configure_me_with_header(hdr)
        for ch in self.children.values():
            ch.configure_with_header(hdr)

    def configure_me_with_image(self, image):
        self.is_configured = True
        self.configure_me_with_header(image[0].header)

    def configure_with_image(self, image):
        self.configure_me_with_image(image)
        for ch in self.children.values():
            ch.configure_with_image(image)
