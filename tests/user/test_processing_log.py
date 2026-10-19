"""The log of a reduction, processing.log, with numina run"""

import logging
import warnings

import pytest

from numina.core import BaseRecipe
from numina.core.requirements import ObservationResultRequirement

from tests.support.drpdef import make_drp

from .testdrp import create_basedir, make_config, run_obs


class WarningRecipe(BaseRecipe):
    """Issues a warning, and logs a message"""

    obresult = ObservationResultRequirement()

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        self.logger.info("message of the recipe")
        warnings.warn("a warning of the recipe", RuntimeWarning)
        return self.create_result()


def work_in_pool(value):
    logging.getLogger("numina.test_pool_recipe").info("message of the process for %d", value)
    return value


class PoolRecipe(BaseRecipe):
    """Logs from the processes of a pool"""

    obresult = ObservationResultRequirement()

    def __init__(self, *args, **kwargs):
        super().__init__(version=1)

    def run(self, recipe_input):
        from numina.util.parallel import logging_pool

        with logging_pool(2) as pool:
            pool.map(work_in_pool, [1, 2])
        return self.create_result()


DRP_LOG = make_drp({"fail": "numina.core.utils.AlwaysFailRecipe", "pool": PoolRecipe, "warn": WarningRecipe})


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    return create_basedir(drpmocker, tmp_path, monkeypatch, drp=DRP_LOG)


def processing_log(basedir, obsid):
    return (basedir / f"obsid{obsid}_results" / "processing.log").read_text()


def test_error_of_the_recipe(basedir):
    """The error and its traceback are in the log"""
    with pytest.raises(TypeError, match="This Recipe always fails"):
        run_obs(make_config(basedir), {"id": 2, "mode": "fail", "instrument": "TEST1", "images": ["image1.fits"]})
    log = processing_log(basedir, 2)
    assert "the recipe failed, TypeError: This Recipe always fails" in log
    assert "Traceback (most recent call last)" in log


@pytest.mark.filterwarnings("default:a warning of the recipe")
def test_warnings_of_the_recipe(basedir):
    run_obs(make_config(basedir), {"id": 3, "mode": "warn", "instrument": "TEST1", "images": ["image1.fits"]})
    log = processing_log(basedir, 3)
    assert "message of the recipe" in log
    assert "RuntimeWarning: a warning of the recipe" in log


@pytest.mark.filterwarnings("default:a warning of the recipe")
def test_logging_state_is_restored(basedir):
    """After the reduction, the warnings are not captured and the levels are restored"""
    numina_level = logging.getLogger("numina").level
    run_obs(make_config(basedir), {"id": 3, "mode": "warn", "instrument": "TEST1", "images": ["image1.fits"]})
    assert logging.getLogger("numina").level == numina_level
    assert getattr(logging, "_warnings_showwarning", None) is None
    assert not logging.getLogger("py.warnings").handlers


def test_messages_of_a_pool_of_processes(basedir):
    """The messages of the processes of logging_pool are in the log"""
    run_obs(make_config(basedir), {"id": 4, "mode": "pool", "instrument": "TEST1", "images": ["image1.fits"]})
    log = processing_log(basedir, 4)
    assert "message of the process for 1" in log
    assert "message of the process for 2" in log
