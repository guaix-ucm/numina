import pytest

pytest_plugins = ["numina.testing.plugins", "tests.support.plugins_pattri"]


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run also the slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow test, skipped without --runslow")


def pytest_collection_modifyitems(config, items):
    """Skip the slow tests without --runslow"""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run it")
    for item in items:
        if item.get_closest_marker("slow"):
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def calibindex_cache(tmp_path_factory, monkeypatch):
    """The cache of the indexes of calibrations, in a temporary directory, not in the cache of the user"""
    cache_dir = tmp_path_factory.mktemp("calibindex")
    monkeypatch.setattr("numina.dal.calibindex.default_cache_dir", lambda: str(cache_dir))
    return cache_dir
