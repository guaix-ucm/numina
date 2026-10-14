import pytest

import numina.user.clirun
from numina.exceptions import ValidationError
from numina.user.cli import main


@pytest.fixture
def invalid_run(monkeypatch):
    def mode_run(args, extra_args, config):
        raise ValidationError("invalid RecipeInput: obresult: invalid raw images")

    monkeypatch.setattr(numina.user.clirun, "mode_run_obsmode", mode_run)


def test_validation_error(invalid_run, capsys):
    """A validation error is logged, without traceback"""
    assert main(["--disable-plugins", "run", "obsresult.yaml"]) == 1
    err = capsys.readouterr().err
    assert "ERROR: invalid RecipeInput: obresult: invalid raw images" in err
    assert "Traceback" not in err


def test_schema_validation_error(tmp_path, monkeypatch, capsys):
    """An invalid control file is logged, without traceback"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "obsresult.yaml").write_text("id: 1\ninstrument: TEST1\nmode: bias\n")
    (tmp_path / "control.yaml").write_text("version: 1\nproducts:\n  TEST1:\n    - {id: 1, type: T, tags: {}}\n")
    assert main(["--disable-plugins", "run", "obsresult.yaml", "-r", "control.yaml"]) == 1
    err = capsys.readouterr().err
    assert "ERROR: control.yaml: invalid control file, at products -> TEST1" in err
    assert "Traceback" not in err


def test_validation_error_debug(invalid_run):
    """With --debug, the exception is raised"""
    with pytest.raises(ValidationError, match="invalid raw images"):
        main(["--disable-plugins", "-d", "run", "obsresult.yaml"])
