import copy
import os.path
import pkgutil

import pytest
import yaml

import numina.core.pipelineload as pload
from .. import helpers
from ..helpers import WorkEnvironment, add_missing_entries, load_observations

# uuid of the profile in numina/drps/tests/configs/instrument-test1.json
TEST1_PROFILE = "225fcaf2-7f6f-49cc-972a-70fd0aee8e96"

DRP_DEFAULTS = {
    "version": 1,
    "requirements": {
        "TEST1": {
            TEST1_PROFILE: {
                "default": {
                    "bias": [
                        {"name": "nlines", "tags": {"vph": "LR-U"}, "content": [25, 25]},
                        {"name": "nlines", "tags": {"vph": "LR-V"}, "content": [15, 5]},
                    ]
                }
            }
        }
    },
}

CONTROL_REQUIREMENTS = {
    "TEST1": {
        TEST1_PROFILE: {
            "default": {
                "bias": [
                    {"name": "nlines", "tags": {"vph": "LR-U"}, "content": [99]},
                    {"name": "nlines", "tags": {"vph": "LR-B"}, "content": [7]},
                ]
            }
        }
    }
}


def test_work1(tmpdir):
    """Test default definitions"""
    base = "base"
    basedir = str(tmpdir.dirpath(base))
    data = "data"
    workdir = "a"
    resultsdir = "b"
    work = WorkEnvironment(data, basedir, workdir, resultsdir)
    work.sane_work()

    assert work.workdir == os.path.join(basedir, workdir)
    assert work.basedir == basedir
    assert work.resultsdir == os.path.join(basedir, resultsdir)

    index_base = "index.pkl"
    assert work.index_file == os.path.join(work.workdir, index_base)

    assert os.path.isdir(work.workdir)
    assert os.path.isdir(work.resultsdir)
    assert os.path.isdir(work.basedir)
    assert os.path.isfile(work.index_file)


def test_add_missing_entries():
    entries = [{"name": "nlines", "tags": {"vph": "LR-U"}, "content": [99]}]
    defaults = [
        {"name": "nlines", "tags": {"vph": "LR-U"}, "content": [25, 25]},
        {"name": "nlines", "tags": {"vph": "LR-V"}, "content": [15, 5]},
    ]
    result = add_missing_entries(entries, defaults)
    assert result is entries
    assert entries == [
        {"name": "nlines", "tags": {"vph": "LR-U"}, "content": [99]},
        {"name": "nlines", "tags": {"vph": "LR-V"}, "content": [15, 5]},
    ]


@pytest.fixture
def test1_drp(drpmocker):
    """The DRP TEST1 of numina.drps.tests, with DRP_DEFAULTS as default requirements"""
    drpdata = yaml.safe_load(pkgutil.get_data("numina.drps.tests", "drptest1.yaml"))

    def loader():
        # A copy, DRP_DEFAULTS is used to check the results
        defaults = copy.deepcopy(DRP_DEFAULTS)
        return pload.load_instrument("numina", drpdata, default_requirements=defaults)

    drpmocker.add_drp("TEST1", loader)


def assert_drp_defaults_unchanged(datamanager):
    """The defaults of the DRP are not modified by create_datamanager"""
    drp = datamanager.backend.drps.query_by_name("TEST1")
    assert drp.default_requirements() == DRP_DEFAULTS


@pytest.mark.parametrize("version", [1, 2])
def test_create_datamanager_keeps_control_requirements(test1_drp, run_config, tmp_path, version):
    """The values in the control file are not replaced by the defaults of the DRP"""
    if version == 1:
        control = {"version": 1, "requirements": CONTROL_REQUIREMENTS}
    else:
        control = {"version": 2, "database": {"requirements": CONTROL_REQUIREMENTS}}
    reqfile = tmp_path / "control.yaml"
    reqfile.write_text(yaml.safe_dump(control))

    datamanager = helpers.create_datamanager(run_config, str(reqfile))

    entries = datamanager.backend.req_table["TEST1"][TEST1_PROFILE]["default"]["bias"]
    by_tags = {entry["tags"]["vph"]: entry["content"] for entry in entries}
    # from the control file, LR-U overrides the default of the DRP
    assert by_tags["LR-U"] == [99]
    assert by_tags["LR-B"] == [7]
    # from the defaults of the DRP
    assert by_tags["LR-V"] == [15, 5]
    assert len(entries) == 3
    assert_drp_defaults_unchanged(datamanager)


def test_create_datamanager_drp_defaults(test1_drp, run_config):
    """Without control file, the defaults of the DRP are used"""
    datamanager = helpers.create_datamanager(run_config, None)

    entries = datamanager.backend.req_table["TEST1"][TEST1_PROFILE]["default"]["bias"]
    assert entries == DRP_DEFAULTS["requirements"]["TEST1"][TEST1_PROFILE]["default"]["bias"]
    # the real profiles of TEST1 are loaded
    assert TEST1_PROFILE in {conf["uuid"] for conf in datamanager.backend.components.values()}
    assert_drp_defaults_unchanged(datamanager)


def test_load_observations(tmp_path):
    obfile = tmp_path / "obsdata.yaml"
    obfile.write_text(
        yaml.safe_dump_all(
            [
                {"id": 1, "mode": "bias", "instrument": "FAKE"},
                {"id": 2, "mode": "bias", "instrument": "FAKE", "enabled": False},
            ]
        )
    )
    sessions, loaded_obs = load_observations([str(obfile)])

    assert sessions == [
        [
            {"id": 1, "enabled": True},
            {"id": 2, "enabled": False},
        ]
    ]
    assert [ob["id"] for ob in loaded_obs] == [1, 2]


def test_load_observations_session(tmp_path):
    sessfile = tmp_path / "session.yaml"
    sessfile.write_text(yaml.safe_dump({"session": [{"id": 1, "enabled": True}, {"id": 2, "enabled": False}]}))
    sessions, loaded_obs = load_observations([str(sessfile)], is_session=True)

    assert sessions == [[{"id": 1, "enabled": True}, {"id": 2, "enabled": False}]]
    assert loaded_obs == []
