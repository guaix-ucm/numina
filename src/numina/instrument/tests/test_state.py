import pytest
from numina.tests.seffect import record_call

from ..state import State, Status, Transition


def test_state1():
    state = State()
    assert state.current == Status.STATUS_ACTIVE


def test_configuring():
    state = State()
    state.transition(Transition.TRANSITION_CONFIGURE)
    assert state.current == Status.STATUS_CONFIGURING
    state.transition(Transition.TRANSITION_END_CONFIGURE)
    assert state.current == Status.STATUS_ACTIVE


def test_configuring2():
    state = State()
    with state.managed_configure():
        try:
            raise ValueError()
        except ValueError:
            state.transition(Transition.TRANSITION_ERROR)

    assert state.current == Status.STATUS_FAILED


def test_configuring3():
    state = State()
    with state.managed_configure():
        assert state.current == Status.STATUS_CONFIGURING

    assert state.current == Status.STATUS_ACTIVE


def test_reset1():

    @record_call
    def enter_failed_cb():
        pass

    @record_call
    def exit_failed_cb():
        pass

    @record_call
    def enter_reset_cb():
        pass

    state = State()
    state.enter_failed.connect(enter_failed_cb)
    state.exit_failed.connect(exit_failed_cb)
    state.enter_reset.connect(enter_reset_cb)
    state.transition(Transition.TRANSITION_ERROR)

    assert state.current == Status.STATUS_FAILED
    assert enter_failed_cb.side_effect.called is True
    assert exit_failed_cb.side_effect.called is False
    assert enter_reset_cb.side_effect.called is False

    for t_name in [
        Transition.TRANSITION_CONFIGURE,
        Transition.TRANSITION_END_CONFIGURE,
        Transition.TRANSITION_END_RESET,
    ]:
        enter_failed_cb.side_effect.clear()
        exit_failed_cb.side_effect.clear()
        enter_reset_cb.side_effect.clear()
        state.transition(t_name)
        assert state.current == Status.STATUS_FAILED
        assert enter_failed_cb.side_effect.called is False
        assert exit_failed_cb.side_effect.called is False
        assert enter_reset_cb.side_effect.called is False

    enter_failed_cb.side_effect.clear()
    exit_failed_cb.side_effect.clear()
    enter_reset_cb.side_effect.clear()
    state.transition(Transition.TRANSITION_RESET)
    assert state.current == Status.STATUS_RESETTING
    assert enter_failed_cb.side_effect.called is False
    assert exit_failed_cb.side_effect.called is True
    assert enter_reset_cb.side_effect.called is True


def create_state():
    created_signals = {}
    for n in [
        "enter_failed",
        "exit_failed",
        "enter_reset",
        "exit_reset",
        "enter_active",
        "enter_configure",
    ]:
        created_signals[n] = record_call(lambda: None)

    state = State()
    for key in created_signals:
        getattr(state, key).connect(created_signals[key])

    return state, created_signals


@pytest.mark.parametrize(
    "t_name",
    [
        Transition.TRANSITION_CONFIGURE,
        Transition.TRANSITION_END_CONFIGURE,
    ],
)
def test_reset_fail(t_name):

    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_RESET)

    for n in created_signals:
        created_signals[n].side_effect.clear()

    state.transition(t_name)
    assert state.current == Status.STATUS_FAILED

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    called_signals["enter_failed"] = True
    called_signals["exit_reset"] = True
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]


def test_reset_active():

    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_RESET)

    for n in created_signals:
        created_signals[n].side_effect.clear()

    state.transition(Transition.TRANSITION_END_RESET)
    assert state.current == Status.STATUS_ACTIVE

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    called_signals["enter_active"] = True
    called_signals["exit_reset"] = True
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]


def test_reset_reset():

    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_RESET)

    for n in created_signals:
        created_signals[n].side_effect.clear()

    state.transition(Transition.TRANSITION_RESET)
    assert state.current == Status.STATUS_RESETTING

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]


def test_active_reset0():
    """From active to reset"""
    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_RESET)

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    called_signals["enter_reset"] = True

    assert state.current == Status.STATUS_RESETTING
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]


def test_active_config0():
    """From active to configure"""
    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_CONFIGURE)

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    called_signals["enter_configure"] = True

    assert state.current == Status.STATUS_CONFIGURING
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]


def test_config_config():

    state, created_signals = create_state()

    state.transition(Transition.TRANSITION_CONFIGURE)

    for n in created_signals:
        created_signals[n].side_effect.clear()

    state.transition(Transition.TRANSITION_CONFIGURE)
    assert state.current == Status.STATUS_CONFIGURING

    called_signals = {}
    for n in created_signals:
        called_signals[n] = False
    for n in called_signals:
        assert created_signals[n].side_effect.called == called_signals[n]
