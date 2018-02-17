
import pytest


from numina.util.parser import parse_arg_line
from numina.util.parser import split_type_name


def test_split_type_name():
    label = 'TraceMap(a=1, b="2")'

    pre, post = split_type_name(label)

    assert pre == 'TraceMap'
    assert post == 'a=1, b="2"'


def test_parse_arg():

    assert parse_arg_line("a=1, b='2', c=True") == {"a": 1, "b": "2", "c": True}


def test_parse_arg_empty():

    assert parse_arg_line("   ") == {}


def test_parse_arg_malformed():

    with pytest.raises(ValueError):
        parse_arg_line("abcbcb")