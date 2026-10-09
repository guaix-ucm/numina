"""create_datamanager with the test DRPs of numina.testing.drps

Regression tests of two problems found when combining
create_datamanager with the drpmocker fixture.
"""

import pkgutil

import pytest

import numina.drps
from numina.drps.drpsystem import DrpSystem
from numina.user.helpers import create_datamanager

# uuid of the profile in numina/testing/drps/configs/instrument-test1.json
TEST1_PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"


def load_test1(drpmocker):
    drpmocker.add_drp("TEST1", pkgutil.get_data("numina.testing.drps", "drptest1.yaml"))


@pytest.fixture
def stale_drp_cache(monkeypatch):
    """The cache of get_system_drps is filled before the DRPs are mocked

    This happens when another test calls get_system_drps first.
    """
    monkeypatch.setattr(numina.drps, "_system_drps", DrpSystem())


def test_datamanager_drp_without_recipes_package(drpmocker, run_config):
    """A DRP without <package>.recipes/configs.yaml has no default requirements"""
    # The package of TEST1 is 'numina', and 'numina.recipes' does not exist
    load_test1(drpmocker)

    datamanager = create_datamanager(run_config, None)

    assert "TEST1" in datamanager.backend.drps.query_all()
    assert TEST1_PROFILE in {conf["uuid"] for conf in datamanager.backend.components.values()}
    assert datamanager.backend.req_table == {}


def test_datamanager_drpmocker_with_stale_cache(stale_drp_cache, drpmocker, run_config):
    """drpmocker must work even if the cache of get_system_drps was already filled"""
    load_test1(drpmocker)

    datamanager = create_datamanager(run_config, None)

    assert "TEST1" in datamanager.backend.drps.query_all()
