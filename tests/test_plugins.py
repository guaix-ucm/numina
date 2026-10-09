"""Configuration of the result_compare plugin"""

import os

import pytest

from numina.testing.plugins import pytest_configure
from numina.testing.pytest_resultcmp import ResultCompPlugin


class FakePluginManager:
    def __init__(self):
        self.plugins = []

    def register(self, plugin):
        self.plugins.append(plugin)


class FakeConfig:
    def __init__(self, **options):
        self.options = options
        self.pluginmanager = FakePluginManager()
        self.ini = {"markers": []}

    def getoption(self, name, default=None):
        return self.options.get(name, default)

    def getini(self, name):
        return self.ini[name]


def configure(**options):
    config = FakeConfig(**options)
    pytest_configure(config)
    (plugin,) = config.pluginmanager.plugins
    assert isinstance(plugin, ResultCompPlugin)
    return plugin


def test_generate_path_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin = configure(**{"--resultcmp-generate-path": "gen"})

    assert plugin.enabled
    assert plugin.generate_dir == os.path.join(str(tmp_path), "gen")
    assert plugin.reference_dir is None


def test_reference_path_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plugin = configure(**{"--resultcmp": True, "--resultcmp-reference-path": "ref"})

    assert plugin.enabled
    assert plugin.reference_dir == os.path.join(str(tmp_path), "ref")
    assert plugin.generate_dir is None


def test_generate_path_overrides_reference_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    options = {"--resultcmp-generate-path": "gen", "--resultcmp-reference-path": "ref"}
    with pytest.warns(UserWarning, match="Ignoring --resultcmp-reference-path"):
        plugin = configure(**options)

    assert plugin.generate_dir == os.path.join(str(tmp_path), "gen")


def test_disabled():
    plugin = configure()

    assert not plugin.enabled
    assert plugin.reference_dir is None
    assert plugin.generate_dir is None
