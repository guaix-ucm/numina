
from ..signal import Signal


def cb1(arg1):
    return arg1


def cb2(arg2):
    return 0


def test_signal_cb():
    sig = Signal()
    cid1 = sig.connect(cb1)
    cid2 = sig.connect(cb2)

    res = sig.emit(12)
    assert res == [(cid1, 12), (cid2, 0)]


def test_signal_delete():
    sig = Signal()

    cid1 = sig.connect(cb1)
    cid2 = sig.connect(cb2)

    sig.delete(cid1)
    assert len(sig.callbacks) == 1

    res = sig.emit(12)
    assert res == [(cid2, 0)]
