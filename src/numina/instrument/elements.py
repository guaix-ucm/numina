#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import enum
from typing import TypeAlias, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Self
    from .configorigin import ElementOrigin

from .generic import ComponentGeneric, InstrumentGeneric


class ElementBase:
    """Base class for objects in component collection"""

    def __init__(self, name: str, origin: "ElementOrigin | None" = None):
        self.name = name
        self.origin = origin

    def set_origin(self, origin):
        """Set the origin of the configuration of the element."""
        self.origin = origin

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent=None,
        properties=None,
        setup=None,
    ) -> "Self":
        """Create the element named ``comp_id`` with ``origin``, ignoring the other arguments."""
        obj = cls.__new__(cls)
        obj.__init__(comp_id, origin)
        return obj


class PropertiesBlock(ElementBase):
    """Class representing a properties component

    Parameters
    ----------
    name
        Name of the element.
    properties
        Property descriptors, by name.
    origin
        Origin of the configuration.
    """

    def __init__(
        self,
        name: str,
        properties: dict | None = None,
        origin: "ElementOrigin | None" = None,
    ):
        super().__init__(name, origin=origin)

        self.properties = {} if properties is None else properties
        self.children = {}

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent=None,
        properties=None,
        setup=None,
    ) -> "Self":
        """Create the block named ``comp_id`` with ``properties`` and ``origin``."""
        obj = cls.__new__(cls)
        obj.__init__(comp_id, properties, origin)
        return obj


class SetupBlock(ElementBase):
    """Class representing a setup component"""

    def __init__(self, name: str, origin: "ElementOrigin | None" = None):
        super().__init__(name, origin=origin)
        self.values = {}

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent=None,
        properties=None,
        setup=None,
    ) -> "Self":
        """Create the block named ``comp_id``, with the values of the block ``setup``."""
        obj = super().from_component(name, comp_id, origin, parent, properties, setup)
        if setup is not None:
            obj.values = setup.values
        return obj


class ComponentBlock(ElementBase):
    """Class representing a device component"""

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent=None,
        properties=None,
        setup=None,
    ) -> "ComponentGeneric":
        """Create the device with ``ComponentGeneric.from_component``."""
        return ComponentGeneric.from_component(name, comp_id, origin, parent, properties, setup)


class InstrumentBlock(ElementBase):
    """Class representing a device component"""

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent=None,
        properties=None,
        setup=None,
    ) -> "InstrumentGeneric":
        """Create the instrument with ``InstrumentGeneric.from_component``."""
        return InstrumentGeneric.from_component(name, comp_id, origin, parent, properties, setup)


BlockType: TypeAlias = Type[InstrumentBlock] | Type[ComponentBlock] | Type[SetupBlock] | Type[PropertiesBlock]


class ElementEnum(enum.Enum):
    """Types of the elements of the instrument configurations."""

    ELEM_INSTRUMENT = 0
    ELEM_COMPONENT = 1
    ELEM_SETUP = 2
    ELEM_PROPERTIES = 3

    @staticmethod
    def to_class(block: "ElementEnum") -> BlockType:
        """Return the block class of the element type ``block``."""
        match block:
            case ElementEnum.ELEM_INSTRUMENT:
                return InstrumentBlock
            case ElementEnum.ELEM_COMPONENT:
                return ComponentBlock
            case ElementEnum.ELEM_SETUP:
                return SetupBlock
            case ElementEnum.ELEM_PROPERTIES:
                return PropertiesBlock

    @staticmethod
    def from_str(element_name: str) -> "ElementEnum":
        """Return the element type named ``element_name``, the ``type`` of a configuration file.

        Raises
        ------
        ValueError
            If the name is not the name of an element type.
        """
        match element_name:
            case "instrument":
                return ElementEnum.ELEM_INSTRUMENT
            case "component":
                return ElementEnum.ELEM_COMPONENT
            case "properties":
                return ElementEnum.ELEM_PROPERTIES
            case "setup":
                return ElementEnum.ELEM_SETUP
            case _:
                raise ValueError(f"no class for {element_name}")

    @staticmethod
    def to_str(block: "ElementEnum") -> str:
        """Return the name of the element type ``block``, the inverse of ``from_str``."""
        match block:
            case ElementEnum.ELEM_INSTRUMENT:
                return "instrument"
            case ElementEnum.ELEM_COMPONENT:
                return "component"
            case ElementEnum.ELEM_SETUP:
                return "setup"
            case ElementEnum.ELEM_PROPERTIES:
                return "properties"
