"""Pools of processes that log through the loggers of the main process"""

import logging
import multiprocessing

import pytest

from numina.util.parallel import logging_pool

LOGGER = "test_parallel_pool.worker"


def work(value):
    """Log a message of each level, and return the value doubled"""
    logger = logging.getLogger(LOGGER)
    logger.debug("debug message of %d", value)
    logger.info("info message of %d", value)
    return 2 * value


METHODS = [m for m in ["fork", "forkserver", "spawn"] if m in multiprocessing.get_all_start_methods()]


@pytest.fixture
def logger():
    logger = logging.getLogger(LOGGER)
    yield logger
    logger.setLevel(logging.NOTSET)


@pytest.mark.parametrize("method", METHODS)
def test_messages_of_the_processes(caplog, logger, method):
    """The messages of the processes reach the handlers of the main process"""
    logger.setLevel(logging.DEBUG)
    with logging_pool(2, context=multiprocessing.get_context(method)) as pool:
        assert pool.map(work, [1, 2, 3]) == [2, 4, 6]
    messages = sorted(r.getMessage() for r in caplog.records if r.name == LOGGER)
    assert messages == sorted(f"{level} message of {v}" for v in [1, 2, 3] for level in ["debug", "info"])


@pytest.mark.parametrize("method", METHODS)
def test_levels_of_the_main_process(caplog, logger, method):
    """The processes have the levels of the loggers of the main process"""
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.DEBUG)
    with logging_pool(2, context=multiprocessing.get_context(method)) as pool:
        pool.map(work, [1, 2])
    messages = sorted(r.getMessage() for r in caplog.records if r.name == LOGGER)
    assert messages == ["info message of 1", "info message of 2"]


def initialize(value):
    logging.getLogger(LOGGER).info("initialized with %s", value)


def test_initializer(caplog, logger):
    logger.setLevel(logging.INFO)
    with logging_pool(1, initializer=initialize, initargs=("x",)) as pool:
        pool.map(work, [1])
    assert "initialized with x" in [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize("method", [m for m in ["forkserver", "spawn"] if m in METHODS])
def test_plain_pool_loses_messages(caplog, logger, method):
    """Without logging_pool, the messages of the processes do not reach the main process"""
    logger.setLevel(logging.DEBUG)
    with multiprocessing.get_context(method).Pool(2) as pool:
        pool.map(work, [1, 2])
    assert [r for r in caplog.records if r.name == LOGGER] == []


def import_astropy(value):
    """Import a package that creates its logger with its own class"""
    import astropy.io.fits  # noqa: F401

    return value


@pytest.mark.parametrize("method", [m for m in ["forkserver", "spawn"] if m in METHODS])
def test_packages_with_own_logger(method):
    """The processes can import astropy, whose logger has a level in the main process"""
    import astropy  # noqa: F401

    assert logging.getLogger("astropy").level != logging.NOTSET
    with logging_pool(1, context=multiprocessing.get_context(method)) as pool:
        # it hung when the logger of astropy was created before importing it
        assert pool.map_async(import_astropy, [1]).get(timeout=60) == [1]
