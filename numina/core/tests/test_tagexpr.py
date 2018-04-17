
import pytest
import numina.core.tagexpr as tagexpr


def test_tagexpresion1():

    mode_tags = {'vph': 'MR-I', 'insmode': 'LCB'}

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

    expr11 = expr1.fill_placeholders(mode_tags)
    assert len(expr11.places()) == 0

    expr21 = expr2.fill_placeholders(mode_tags)
    assert len(expr21.places()) == 0

    with pytest.raises(KeyError):
        expr3.fill_placeholders(mode_tags)


def test_tagexpresion_filter1():

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
