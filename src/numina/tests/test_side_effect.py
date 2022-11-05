
from .seffect import side_effect, record_call, FuncCall


def test_side_effect_1():

    somevals = []

    def sidefunc(pos):
        somevals.append(pos)

    @side_effect(sidefunc)
    def testfunc(pos):
        return pos

    assert testfunc(1) == 1

    assert somevals == [1]


def test_side_effect_except():

    somevals = []

    def sidefunc(pos):
        a = 1 / 0
        somevals.append(a)

    @side_effect(sidefunc)
    def testfunc(pos):
        return pos

    assert testfunc(1) == 1
    # Exception is ignored, no side effect
    assert somevals == []


def test_record_call():

    @record_call
    def testfunc(pos):
        return pos

    assert testfunc.side_effect.args == ()
    assert testfunc.side_effect.kwargs == {}
    assert testfunc.side_effect.called == False
    
    assert testfunc(1) == 1
    assert isinstance(testfunc.side_effect, FuncCall)

    assert testfunc.side_effect.args == (1,)
    assert testfunc.side_effect.kwargs == {}
    assert testfunc.side_effect.called == True

    testfunc.side_effect.clear()

    assert testfunc.side_effect.args == ()
    assert testfunc.side_effect.kwargs == {}
    assert testfunc.side_effect.called == False