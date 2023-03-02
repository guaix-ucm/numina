#
# Copyright 2016-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from .device import DeviceBase


class ElementBase(object):
    """Base class for objects in component collection"""
    def __init__(self, name, origin=None):
        self.name = name
        self.origin = origin

    def set_origin(self, origin):
        self.origin = origin


class PropertiesGeneric(ElementBase):
    """Class representing a properties component"""
    def __init__(self, name, properties=None, origin=None):
        super(PropertiesGeneric, self).__init__(name, origin=origin)

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        self.children = {}


class ComponentGeneric(DeviceBase):
    """Class representing a device component"""
    def __init__(self, name, properties=None, origin=None, parent=None):
        super(ComponentGeneric, self).__init__(
            name, origin=origin, parent=parent
        )

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        #
        self._internal_state = {}

        for prop in self.properties.values():
            deflts = prop.defaults()
            for dk, dv in deflts.items():
                self._internal_state[dk] = dv

    def get_value(self, path, **state):
        keys = path.split('.')
        devce_path = keys[:-1]
        prop = keys[-1]

        dev = self.get_device_seq(devce_path)
        if prop in dev.properties:
            propentry = dev.properties[prop]
            return propentry.get(**state)
        else:
            return getattr(dev, prop)

    def __getattr__(self, item):
        if item == 'uuid':
            if self.origin is not None:
                return self.origin.uuid
            
        if item in self.properties:
            prop_entry = self.properties[item]
            value = prop_entry.get(**self._internal_state)
            return value
        else:
            raise AttributeError(f"component has no attribute '{item}'")

    def depends_on(self):
        """Compute the dependencies for me and my children"""
        mydepends = set([])

        mydepends.update(self.my_depends())

        for ch in self.children.values():
            if hasattr(ch, 'depends_on'):
                chdepends = ch.depends_on()
                mydepends.update(chdepends)
        return mydepends

    def configure_me_with_header(self, hdr):
        for key in self.my_depends():
            self._internal_state[key] = hdr[key]

    def my_depends(self):
        """Compute the dependencies for me"""
        mydepends = set([])
        for prop in self.properties.values():
            mydepends.update(prop.depends)
        return mydepends


class InstrumentGeneric(ComponentGeneric):
    """Class representing a instrument component"""
    pass


class SetupGeneric(ElementBase):
    """Class representing a setup component"""
    def __init__(self, name, origin=None):
        super(SetupGeneric, self).__init__(name, origin=origin)
        self.values = {}


class PropertyEntry(object):
    def __init__(self, values, depends):
        self.values = values
        self.depends = depends

    def get(self, **kwds):
        result = self.values
        for dep in self.depends:
            key = kwds[dep]
            result = result[key]
        return result

    def defaults(self):
        val = {}
        result = self.values
        for dep in self.depends:
            # First key, sorted needed in Python < 3.6
            key = sorted(result.keys())[0]
            val[dep] = key
            result = result[key]
        return val


ConfigurationEntry = PropertyEntry
