
import pytest

from ..shutter import Shutter


class Testigo(object):

    __test__ = False  # avoid pytest collection

    def __init__(self):
        self.clear()

    def func(self, pos):
        self.called = True
        self.callpos = pos

    def clear(self):
        self.callpos = None
        self.called = False


@pytest.fixture
def shutter_dev():
    wheel = Shutter()
    return wheel


def test_shutter1(shutter_dev):
    curr = shutter_dev.current()
    assert shutter_dev.pos() == 1
    assert curr.name == 'OPEN'


def test_shutter_open(shutter_dev):
    shutter_dev.open()
    curr = shutter_dev.current()
    assert shutter_dev.pos() == 1
    assert curr.name == 'OPEN'


def test_shutter_stop(shutter_dev):
    shutter_dev.close()
    curr = shutter_dev.current()
    assert shutter_dev.pos() == 0
    assert curr.name == 'STOP'
