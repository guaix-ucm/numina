from ..generic import ComponentGeneric, make_component_class, InstrumentGeneric
from ..configorigin import ElementOrigin
from ..property import PropertyEntry, PropertyModLimits


def test_uuid1():

    comp = ComponentGeneric("testcomp")
    assert comp.uuid is None


def test_uuid2():

    comp = ComponentGeneric("testcomp")
    uuid_str = "8e87de0f-2584-4618-9bdb-76c6420f26cf"
    origin = ElementOrigin("testconf", uuid_str)
    comp.set_origin(origin)

    assert str(comp.uuid) == uuid_str
    assert str(comp.origin.uuid) == uuid_str


def test_component_1():

    props = {"p1": PropertyEntry(values=1, depends=[])}
    comp1 = ComponentGeneric.from_component("TEST1", "testcomp1", properties=props)
    assert hasattr(comp1, "p1")
    comp2 = ComponentGeneric("testcomp2")
    assert not hasattr(comp2, "p1")


def test_component_2():
    props = {"p1": PropertyEntry(values=1, depends=[])}

    cls = make_component_class("B", ComponentGeneric, props)

    obj = cls("b")  # noqa
    assert hasattr(obj, "p1")
    assert isinstance(obj, ComponentGeneric)


def test_component_3():
    props = {"p1": PropertyEntry(values=1, depends=[])}

    obj = ComponentGeneric.from_component("B", "b", properties=props)
    assert isinstance(obj, ComponentGeneric)


def test_component_4():
    """Each object has a value of the property"""
    props = {"p1": PropertyModLimits(limits=[-1, 3], default=0)}

    cls = make_component_class("B", ComponentGeneric, props)

    obj1 = cls("b")  # noqa
    obj2 = cls("b")  # noqa

    # defaults before setting any value
    assert obj1.p1 == 0
    assert obj2.p1 == 0

    obj1.p1 = 3
    obj2.p1 = 2

    assert obj1.p1 == 3
    assert obj2.p1 == 2


def test_instrument_1():
    props = {"p1": PropertyEntry(values=1, depends=[])}
    comp1 = InstrumentGeneric.from_component("TEST1", "testcomp1", properties=props)
    assert hasattr(comp1, "p1")
    comp2 = InstrumentGeneric("testcomp2")
    assert not hasattr(comp2, "p1")


def test_instrument_2():
    props = {"p1": PropertyEntry(values=1, depends=[])}

    cls = make_component_class("B", InstrumentGeneric, props)

    obj = cls("b")  # noqa
    assert hasattr(obj, "p1")
    assert isinstance(obj, InstrumentGeneric)


def test_instrument_3():
    props = {"p1": PropertyEntry(values=1, depends=[])}

    obj = InstrumentGeneric.from_component("B", "b", properties=props)
    assert isinstance(obj, InstrumentGeneric)


def test_instrument_4():
    """Each object has a value of the property"""
    props = {"p1": PropertyModLimits(limits=[-1, 3], default=0)}

    cls = make_component_class("B", InstrumentGeneric, props)

    obj1 = cls("b")  # noqa
    obj2 = cls("b")  # noqa

    # defaults before setting any value
    assert obj1.p1 == 0
    assert obj2.p1 == 0

    obj1.p1 = 3
    obj2.p1 = 2

    assert obj1.p1 == 3
    assert obj2.p1 == 2
