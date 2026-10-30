#

from typing import TypeVar, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .configorigin import ElementOrigin
    from .property import PropertyBase
    from typing import Self

from .device import DeviceBase


class ComponentGeneric(DeviceBase):
    """Class representing a device component from a file"""

    _property_names = []
    _property_readonly = []
    _property_configurable = []
    _mappings = {}

    def __init__(
        self,
        name: str,
        origin: "ElementOrigin | None" = None,
        parent: DeviceBase | None = None,
    ):
        super().__init__(name, origin=origin, parent=parent)
        # This attribute is used to store the values of the properties
        self._internal_state = {}

        for prop_name in self._property_names:
            deflts = self.__class__.__dict__[prop_name].defaults()

            for dk, dv in deflts.items():
                self._internal_state[dk] = dv

    @classmethod
    def from_component(
        cls,
        name: str,
        comp_id: str,
        origin: "ElementOrigin | None" = None,
        parent: DeviceBase | None = None,
        properties=None,
        setup=None,
    ) -> "Self":
        """Create the device named ``comp_id`` as an instance of a new subclass with ``properties``.

        The subclass, named after the class and ``name``, is built with ``make_component_class``.
        ``setup`` is ignored.
        """
        class_name = "{base_name}_{name}".format(base_name=cls.__name__, name=name)
        ncls = make_component_class(class_name, cls, properties=properties)
        obj = ncls.__new__(ncls)
        obj.__init__(comp_id, origin, parent)
        return obj

    def get_value(self, path: str, **state):
        """Return the value of the property at ``path``, with its device configured temporarily with ``state``."""
        device_path, prop_name = self.split_path(path)

        dev = self.get_device_seq(device_path)
        # save state and restore state
        current_state = dev.config_info()
        dev.configure_me(state)
        value = getattr(dev, prop_name)
        dev.configure(current_state)
        return value

    def configure_me_with_header(self, hdr):
        """Set the properties mapped to header keywords with their values in ``hdr``, unless they are None."""
        for el, key in self._mappings.items():
            val = key.get_value(hdr)
            if val is not None:
                setattr(self, el, val)

    def my_depends(self) -> set[str]:
        """Compute the dependencies for me"""
        my_deps = set([])
        for prop_name in self._property_names:
            depends = self.__class__.__dict__[prop_name].depends
            my_deps.update(depends)
        return my_deps

    def get_property_names(self):
        """Return the names of the configurable properties."""
        return self._property_configurable


class InstrumentGeneric(ComponentGeneric):
    """Instrument assembled from a configuration, the same as ``ComponentGeneric``."""

    pass


CG = TypeVar("CG", bound=ComponentGeneric)


def make_component_class(
    name: str,
    baseclass: Type[ComponentGeneric],
    properties: "dict[str, PropertyBase] | None" = None,
) -> Type[ComponentGeneric]:
    """Create a subclass of ``baseclass`` named ``name``, with the property descriptors ``properties``.

    The properties are set as class attributes, and their names are recorded as read-only or
    configurable. Their header keyword definitions, if any, are recorded in ``_mappings``.
    """
    # Inject properties as descriptors in the class instance
    cls = type(name, (baseclass,), {})

    cls._property_readonly = []
    cls._property_configurable = []
    cls._property_names = []
    cls._mappings = {}

    if properties is not None:
        cls._property_names = list(properties.keys())
        for p_name, prop in properties.items():
            if prop.is_readonly:
                cls._property_readonly.append(p_name)
            else:
                cls._property_configurable.append(p_name)
            if prop.mapping is not None:
                cls._mappings[p_name] = prop.mapping
            # This method is not called automatically here
            prop.__set_name__(cls, p_name)
            setattr(cls, p_name, prop)
    return cls
