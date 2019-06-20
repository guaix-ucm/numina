
import pytest

from numina.tests.seffect import record_call

from ..wheel import Wheel, Carrousel


class Disp(object):
    def __init__(self, name):
        self.name = name


@pytest.fixture
def wheel_dev():
    wheel = Wheel(3)
    for idx in range(3):
        wheel.put_in_pos(Disp(idx), idx)
    return wheel


@pytest.fixture
def carrousel_dev():
    wheel = Carrousel(3)
    for idx in range(3):
        wheel.put_in_pos(Disp(idx), idx)
    return wheel


def test_wheel(wheel_dev):
    curr = wheel_dev.current()
    assert isinstance(curr, Disp)
    assert curr.name == 0
    assert wheel_dev.pos() == 0


def test_carrousel(carrousel_dev):
    curr = carrousel_dev.current()
    assert isinstance(curr, Disp)
    assert curr.name == 0
    assert carrousel_dev.pos() == 0


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_move_wheel(carrousel_dev, pos):
    carrousel_dev.move_to(pos)

    curr = carrousel_dev.current()
    assert isinstance(curr, Disp)
    assert curr.name == pos
    assert carrousel_dev.pos() == pos


def test_move_wheel3(carrousel_dev):

    with pytest.raises(ValueError):
        carrousel_dev.move_to(8)


@pytest.mark.parametrize("pos", [1, 2])
def test_move_signals(carrousel_dev, pos):

    @record_call
    def changed_cb(pos):
        pass

    @record_call
    def moved_cb(pos):
        pass

    # Connect signals
    carrousel_dev.changed.connect(changed_cb)
    carrousel_dev.moved.connect(moved_cb)

    carrousel_dev.move_to(0)
    # Reset capture
    changed_cb.side_effect.clear()
    moved_cb.side_effect.clear()

    carrousel_dev.move_to(0)

    # Check call values
    assert moved_cb.side_effect.args == (0,)
    assert changed_cb.side_effect.called == False

    # Clear last call values
    changed_cb.side_effect.clear()
    moved_cb.side_effect.clear()

    carrousel_dev.move_to(pos)

    assert moved_cb.side_effect.called
    assert moved_cb.side_effect.args == (pos,)

    assert changed_cb.side_effect.called
    assert changed_cb.side_effect.args == (pos,)


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_select(carrousel_dev, pos):
    carrousel_dev.select(pos)
    assert carrousel_dev.pos() == pos
    curr = carrousel_dev.current()
    assert isinstance(curr, Disp)
    assert curr.name == pos


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_turn(wheel_dev, pos):
    wheel_dev.select(pos)

    @record_call
    def changed_cb(pos):
        pass

    @record_call
    def moved_cb(pos):
        pass

    # Connect signals
    wheel_dev.changed.connect(changed_cb)
    wheel_dev.moved.connect(moved_cb)

    steps = 6
    step = 0
    while step < steps:
        wheel_dev.turn()
        target = (pos + 1 + step) % 3
        step += 1
        assert wheel_dev.pos() == target
        assert moved_cb.side_effect.called
        assert moved_cb.side_effect.args == (target,)
        assert changed_cb.side_effect.called
        assert changed_cb.side_effect.args == (target,)

        changed_cb.side_effect.clear()
        moved_cb.side_effect.clear()


@pytest.mark.parametrize("pos", [1, 2])
def test_move_signals(carrousel_dev, pos):

    @record_call
    def changed_cb(pos):
        pass

    @record_call
    def moved_cb(pos):
        pass

    # Connect signals
    carrousel_dev.changed.connect(changed_cb)
    carrousel_dev.moved.connect(moved_cb)

    carrousel_dev.move_to(0)
    # Reset capture
    changed_cb.side_effect.clear()
    moved_cb.side_effect.clear()

    carrousel_dev.move_to(0)

    # Check call values
    assert moved_cb.side_effect.called
    assert moved_cb.side_effect.args == (0, )
    assert changed_cb.side_effect.called == False
    # assert moved_cb.callpos == 0

    # Clear last call values
    changed_cb.side_effect.clear()
    moved_cb.side_effect.clear()

    carrousel_dev.move_to(pos)

    assert moved_cb.side_effect.called
    assert moved_cb.side_effect.args == (pos,)

    assert changed_cb.side_effect.called
    assert changed_cb.side_effect.args == (pos,)
