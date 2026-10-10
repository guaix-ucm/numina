"""Reduction requests: how to reduce the observing blocks, in a file or in Session.run"""

import pytest
import yaml

from numina.schemas import SchemaValidationError
from numina.user.cli import base_config, main
from numina.user.helpers import ReductionRequests
from numina.user.session import Session

from . import test_run_registry as trr

BIAS = trr.BIAS
IMAGE = {"id": 10, "mode": "image", "instrument": "TEST1", "images": ["image2.fits"]}
PARAM = {"id": 30, "mode": "param", "instrument": "TEST1"}
PARAM_OB = {"id": 31, "mode": "param", "instrument": "TEST1", "requirements": {"value": 3}}


@pytest.fixture
def basedir(drpmocker, tmp_path, monkeypatch):
    basedir = trr.create_basedir(drpmocker, tmp_path, monkeypatch)
    (basedir / "obs.yaml").write_text(yaml.safe_dump_all([BIAS, IMAGE, PARAM, PARAM_OB]))
    # without the configuration files of the user
    monkeypatch.setattr("numina.user.cli.load_config", base_config)
    return basedir


def write_requests(path, *docs):
    path.write_text(yaml.safe_dump_all(docs))
    return str(path)


def numina(capsys, *args):
    status = main(["--disable-plugins", *args])
    captured = capsys.readouterr()
    return status, captured.err


def test_load_requests(tmp_path):
    filename = write_requests(
        tmp_path / "req.yaml",
        {"id": 30, "requirements": {"value": 5}, "pipeline": "default"},
        {"requirements": {"value": 1}, "mode": "param"},
    )
    requests = ReductionRequests.load(filename)
    assert requests.ids() == [30]
    assert requests.arguments(30) == {"requirements": {"value": 5}, "pipeline": "default"}
    # the id of the command line is a string
    assert requests.arguments("30") == {"requirements": {"value": 5}, "pipeline": "default"}
    # the default request
    assert requests.arguments(31) == {"requirements": {"value": 1}, "as_mode": "param"}


def test_load_requests_without_default(tmp_path):
    requests = ReductionRequests.load(write_requests(tmp_path / "req.yaml", {"id": 30}))
    assert requests.arguments(31) == {}


@pytest.mark.parametrize(
    "docs, error, match",
    [
        ([{"id": 1}, {"id": 1}], ValueError, "two requests for oblock_id=1"),
        ([{}, {"pipeline": "default"}], ValueError, "two requests without id"),
        ([{"id": 1, "frames": ["a.fits"]}], SchemaValidationError, "invalid request file"),
        ([{"id": 1, "requirements": []}], SchemaValidationError, "invalid request file"),
    ],
)
def test_load_invalid_requests(tmp_path, docs, error, match):
    filename = write_requests(tmp_path / "req.yaml", *docs)
    with pytest.raises(error, match=match):
        ReductionRequests.load(filename)


def test_db_run_request(basedir, capsys):
    numina(capsys, "db", "ingest", "--db", "numina-db.json", "obs.yaml")
    req = write_requests(basedir / "req.yaml", {"id": 30, "requirements": {"value": 5}})
    # the ids are those of the request file
    status, _ = numina(capsys, "db", "run", "--db", "numina-db.json", "--request", req)
    assert status == 0
    assert trr.SEEN == [5]


def test_db_run_request_and_command_line(basedir, capsys):
    """The values of the command line override those of the request"""
    numina(capsys, "db", "ingest", "--db", "numina-db.json", "obs.yaml")
    req = write_requests(basedir / "req.yaml", {"id": 30, "requirements": {"value": 5}})
    numina(capsys, "db", "run", "--db", "numina-db.json", "--request", req, "30", "--parameter-value=7")
    assert trr.SEEN == [7]


def test_request_overrides_ob(basedir, capsys):
    numina(capsys, "db", "ingest", "--db", "numina-db.json", "obs.yaml")
    # the default request applies to the blocks without their own request
    req = write_requests(basedir / "req.yaml", {"requirements": {"value": 5}})
    numina(capsys, "db", "run", "--db", "numina-db.json", "--request", req, "30", "31")
    assert trr.SEEN == [5, 5]
    # without request, the value of the OB
    numina(capsys, "db", "run", "--db", "numina-db.json", "31")
    assert trr.SEEN == [5, 5, 3]


def test_db_run_without_ids(basedir, capsys):
    numina(capsys, "db", "ingest", "--db", "numina-db.json", "obs.yaml")
    req = write_requests(basedir / "req.yaml", {"requirements": {"value": 5}})
    status, err = numina(capsys, "db", "run", "--db", "numina-db.json", "--request", req)
    assert status == 2
    assert "no observing blocks" in err


def test_numina_run_request(basedir, capsys):
    req = write_requests(basedir / "req.yaml", {"id": 30, "requirements": {"value": 5}})
    (basedir / "param.yaml").write_text(yaml.safe_dump(PARAM))
    status, _ = numina(capsys, "run", "--request", req, "param.yaml")
    assert trr.SEEN == [5]


def test_session_request(basedir):
    session = Session(basedir=basedir, config=base_config())
    session.add_observations("obs.yaml")
    session.run(30, request={"requirements": {"value": 5}})
    write_requests(basedir / "req.yaml", {"id": 31, "requirements": {"value": 6}})
    session.run(31, request="req.yaml")
    # the keyword arguments override the request
    session.run(31, request="req.yaml", requirements={"value": 8})
    assert trr.SEEN == [5, 6, 8]


def test_session_invalid_request(basedir):
    session = Session(basedir=basedir, config=base_config())
    session.add_observations("obs.yaml")
    with pytest.raises(SchemaValidationError):
        session.run(30, request={"value": 5})
