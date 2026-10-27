import operator

import pytest

import numina.core.tagexpr as tagexpr


def test_tag_expresion1():

    mode_tags = {"vph": "MR-I", "insmode": "LCB"}

    insmode = tagexpr.TagRepr("insmode")
    vph = tagexpr.TagRepr("vph")
    speclamp = tagexpr.TagRepr("speclamp")
    p_ = tagexpr.Placeholder

    expr1 = vph == p_("vph")
    expr2 = (vph == "LR-I") & (insmode == "LCB") & (insmode != "MOS")
    expr3 = (vph == "LR-I") & (insmode == "LCB") & (speclamp == p_("speclamp"))

    assert isinstance(expr1, tagexpr.Expression)
    assert isinstance(expr2, tagexpr.Expression)
    assert isinstance(expr3, tagexpr.Expression)

    expr11 = expr1.fill_placeholders(**mode_tags)
    assert len(expr11.places()) == 0

    expr21 = expr2.fill_placeholders(**mode_tags)
    assert len(expr21.places()) == 0

    # with pytest.raises(KeyError):
    #    expr3.fill_placeholders(**mode_tags)


def test_tag_expresion_filter1():

    insmode = tagexpr.TagRepr("insmode")
    vph = tagexpr.TagRepr("vph")
    p_ = tagexpr.Placeholder

    expr1 = (vph == "LR-I") & (insmode == p_("insmode")) & (insmode != "MOS")

    res = [n for n in tagexpr.filter_tree(lambda node: True, expr1)]

    assert len(res) == 11

    assert isinstance(res[0], tagexpr.TagRepr)
    assert res[0].name == "vph"
    assert isinstance(res[1], tagexpr.ConstExpr)
    assert res[1].value == "LR-I"
    assert isinstance(res[2], tagexpr.PredEq)
    assert isinstance(res[3], tagexpr.TagRepr)
    assert res[3].name == "insmode"
    assert isinstance(res[4], tagexpr.Placeholder)
    assert res[4].name == "insmode"
    assert isinstance(res[5], tagexpr.PredEq)
    assert isinstance(res[6], tagexpr.PredAnd)
    assert isinstance(res[7], tagexpr.TagRepr)
    assert res[7].name == "insmode"
    assert isinstance(res[8], tagexpr.ConstExpr)
    assert res[8].value == "MOS"
    assert isinstance(res[9], tagexpr.PredNe)
    assert isinstance(res[10], tagexpr.PredAnd)

    expr2 = expr1.fill_placeholders(insmode="LCB")
    assert expr2.eval(insmode="LCB", vph="LR-I") is True


def test_tag_expresion_filter2():

    insmode = tagexpr.TagRepr("insmode")
    vph = tagexpr.TagRepr("vph")
    temp = tagexpr.TagRepr("temp")
    p_ = tagexpr.Placeholder

    expr1 = (vph == p_("vph")) & (insmode == p_("insmode")) & (temp >= p_("temp"))
    expr2 = expr1.fill_placeholders(insmode="LCB", vph="LR-I", temp=22.4)

    assert expr2.eval(insmode="LCB", vph="LR-I", temp=22.4)

    assert expr2.eval(insmode="LCB", vph="LR-I", temp=24.4)

    assert expr2.eval(insmode="LCB", vph="LR-I", temp=20.0) is False

    assert expr2.eval(insmode="MOS", vph="LR-I", temp=23.0) is False


def test_operators():
    vph = tagexpr.TagRepr("vph")
    cases = [
        (vph > 2, tagexpr.PredGt, 3, 2),
        (vph >= 2, tagexpr.PredGe, 2, 1),
        (vph < 2, tagexpr.PredLt, 1, 2),
        (vph <= 2, tagexpr.PredLe, 2, 3),
        (vph != 2, tagexpr.PredNe, 1, 2),
    ]
    for expr, cls, true_value, false_value in cases:
        assert isinstance(expr, cls)
        assert expr.eval(vph=true_value) is True
        assert expr.eval(vph=false_value) is False


def test_or():
    vph = tagexpr.TagRepr("vph")
    expr = (vph == "LR-U") | (vph == "LR-B")
    assert isinstance(expr, tagexpr.PredOr)
    assert str(expr) == "((TagRepr(vph) == ConstExpr(LR-U)) OR (TagRepr(vph) == ConstExpr(LR-B)))"
    assert expr.eval(vph="LR-B") is True
    assert expr.eval(vph="LR-I") is False
    assert expr.fields() == {"vph"}


def test_bool_operands():
    expr = tagexpr.TagRepr("vph") == "LR-U"
    assert (expr & True) is expr
    assert (True & expr) is expr
    assert (expr & False) is False
    assert (expr | True) is True
    assert (True | expr) is True


def test_or_false():
    expr = tagexpr.TagRepr("vph") == "LR-U"
    assert (expr | False) is expr
    assert (False | expr) is expr


def test_not_supported_operands():
    expr = tagexpr.TagRepr("vph") == "LR-U"
    with pytest.raises(TypeError):
        expr & 3
    with pytest.raises(TypeError):
        expr | "a"


def test_not():
    expr = ~(tagexpr.TagRepr("vph") == "LR-U")
    assert isinstance(expr, tagexpr.PredNot)
    assert expr.eval(vph="LR-U") is False
    assert expr.eval(vph="LR-B") is True


def test_not_fill_placeholders():
    vph = tagexpr.TagRepr("vph")
    expr = ~(vph == tagexpr.Placeholder("vph"))
    filled = expr.fill_placeholders(vph="LR-U")
    assert filled.places() == set()
    assert filled.eval(vph="LR-U") is False


def test_tags_and_copy():
    vph = tagexpr.TagRepr("vph", metadata={"type": str})
    expr = vph == tagexpr.Placeholder("vph")
    assert expr.tags() == {"vph"}
    assert not expr.is_terminal()
    assert vph.is_terminal()
    assert vph.metadata == {"type": str}
    copy = expr.copy()
    assert copy is not expr
    assert str(copy) == str(expr) == "(TagRepr(vph) == Placeholder(vph))"
    assert tagexpr.Expression.map_tree(lambda node, children: len(children), expr) == 2


def test_eval_missing_tag():
    vph = tagexpr.TagRepr("vph")
    assert vph.eval(other=1) is vph


def test_eval_placeholder():
    """A expression with placeholders is incomplete, its value is not a bool"""
    expr = tagexpr.TagRepr("vph") == tagexpr.Placeholder("vph")
    assert expr.eval(vph="LR-U") is not True


def test_clone_not_implemented():
    with pytest.raises(NotImplementedError):
        tagexpr.Expression().clone([])


def test_adapter():
    expr = tagexpr.TagRepr("vph") == "LR-U"
    constraint = tagexpr.adapter(expr)
    assert constraint.key == "vph"
    assert constraint.value == "LR-U"
    assert constraint.type is str
    assert constraint.oper is operator.eq
    assert repr(constraint).startswith("ConstraintAdapter(key=vph, value=LR-U")
    assert tagexpr.adapter(tagexpr.TagRepr("vph")) is None


class Attr:
    def __init__(self, name):
        self.name = name
        self.type = str
        self.description = f"The {name}"


def test_query_expr_from_attr():
    assert tagexpr.query_expr_from_attr([]) is tagexpr.ConstExprTrue

    expr = tagexpr.query_expr_from_attr([Attr("vph"), Attr("insmode")])
    assert expr.fields() == {"vph", "insmode"}
    assert expr.places() == {"vph", "insmode"}
    filled = expr.fill_placeholders(vph="LR-U", insmode="LCB")
    assert filled.eval(vph="LR-U", insmode="LCB") is True
    assert filled.eval(vph="LR-U", insmode="MOS") is False
