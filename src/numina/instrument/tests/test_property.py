import pytest


from ..generic import InstrumentGeneric
from ..property import PropertyEntry, PropertyModOneOf, PropertyModLimits


@pytest.mark.parametrize("mode, value", [("A", 1), ("B", 2)])
def test_property_entry(mode, value):
    values = {"A": 1, "B": 2}
    depends = ["mode"]
    prop = PropertyEntry(values, depends)
    state = {"mode": mode}
    res = prop.get(**state)
    assert res == value


def test_property_entry_set():
    """Setting this type of property does nothing"""
    values = {"A": 1, "B": 2}
    depends = ["mode"]
    prop1 = PropertyEntry(values, depends)
    prop2 = PropertyModOneOf(["A", "B"], default="A")
    props = {"p1": prop1, "mode": prop2}
    obj = InstrumentGeneric.from_component("B", "b", properties=props)
    assert obj.p1 == 1
    obj.p1 = 2
    assert obj.p1 == 1


def test_property_limit_except():
    with pytest.raises(ValueError):
        prop1 = PropertyModLimits([3])  # noqa


@pytest.mark.parametrize("value, inside", [(-2, False), (0, True), (2, False)])
def test_property_limit_check(value, inside):
    prop1 = PropertyModLimits([-1, 1])
    assert prop1.value_check(value) == inside
