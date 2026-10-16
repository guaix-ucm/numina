import pytest

from . import testdrp


@pytest.fixture(autouse=True)
def _empty_seen():
    """Empty the values recorded by the recipes of testdrp before each test"""
    testdrp.SEEN.clear()
    testdrp.SEEN_BY_MODE.clear()


@pytest.fixture
def seen():
    """The values received by the recipes of testdrp, in this test"""
    return testdrp.SEEN


@pytest.fixture
def seen_by_mode():
    """The values received by the recipes of testdrp, by mode, in this test"""
    return testdrp.SEEN_BY_MODE
