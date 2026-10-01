"""Templates of the names of the directories and files of numina run"""

import json
import pkgutil

import pytest
import yaml

from numina.dal.utils import TEMPLATE_FIELDS, check_template, fill_template
from ..helpers import create_datamanager


@pytest.mark.parametrize("template", ["obsid{obsid}_{taskid}_work", "result.json", "{taskid}"])
def test_check_template_valid(template):
    assert check_template(template) == template


@pytest.mark.parametrize("template", ["obsid{obid}_work", "obsid{}_work", "{obsid.real}"])
def test_check_template_invalid(template):
    with pytest.raises(ValueError, match="valid fields are"):
        check_template(template)


def test_fill_template():
    assert TEMPLATE_FIELDS == ("obsid", "taskid")
    assert fill_template("obsid{obsid}_{taskid}_work", obsid=4, taskid=12) == "obsid4_12_work"


@pytest.fixture
def test1_drp(drpmocker):
    drpmocker.add_drp("TEST1", pkgutil.get_data("numina.drps.tests", "drptest1.yaml"))


def test_datamanager_invalid_template(test1_drp, run_config):
    run_config["tool.run"]["resultdir_tmpl"] = "results_{obid}"

    with pytest.raises(ValueError, match="results_{obid}"):
        create_datamanager(run_config, None)


def test_format1_results_found_with_templates(test1_drp, run_config, tmp_path):
    """In format 1, the results are searched with the templates used to store them"""
    run_config["tool.run"]["resultdir_tmpl"] = "res_{obsid}"
    run_config["tool.run"]["resultfile_tmpl"] = "out_{taskid}.json"
    datamanager = create_datamanager(run_config, None)
    # The mode 'image' of TEST1 has no tagger, that is deprecated
    datamanager.backend.add_obs([{"id": 7, "mode": "image", "instrument": "TEST1", "images": []}])

    # The directory where the results of the OB are stored
    task = datamanager.backend.new_task("reduce", {"oblock_id": 7})
    workenv = datamanager.create_workenv(task)
    assert workenv.resultsdir_rel == "res_7"

    # A result stored there is found by other OBs
    resultsdir = tmp_path / "res_7"
    resultsdir.mkdir()
    (resultsdir / "out_7.json").write_text(json.dumps({"values": {"value": 42}}))

    stored = datamanager.backend.search_result_id(7, None, "value")
    assert stored.content == 42


def test_format2_keeps_datamanager_templates(test1_drp, run_config, tmp_path):
    """In format 2, the templates include the taskid, the configuration is not used"""
    run_config["tool.run"]["resultdir_tmpl"] = "res_{obsid}"
    reqfile = tmp_path / "control.yaml"
    reqfile.write_text(yaml.safe_dump({"version": 2, "database": {}}))

    datamanager = create_datamanager(run_config, str(reqfile))

    assert datamanager.workdir_tmpl == "obsid{obsid}_{taskid}_work"
    assert datamanager.resultdir_tmpl == "obsid{obsid}_{taskid}_result"
