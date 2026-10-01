"""Arguments not known by the parser of the numina command"""

import logging.config

import pytest

import numina.user.clirun
from ..cli import main, process_unknown_arguments


def test_parameters_are_accepted():
    extra = process_unknown_arguments(["--parameter-a=1", "--parameter-b=x=y"])
    assert extra.extra_control == {"a": "1", "b": "x=y"}
    assert extra.rejected == []


@pytest.mark.parametrize("argument", ["--parameter-a", "--parameter-=1", "--workdir", "obs.yaml"])
def test_other_arguments_are_rejected(argument):
    extra = process_unknown_arguments([argument])
    assert extra.extra_control == {}
    assert extra.rejected == [argument]


def test_main_rejects_unknown_option(capsys):
    """An option removed from numina run fails, instead of being ignored"""
    with pytest.raises(SystemExit) as exc:
        main(["--disable-plugins", "run", "--workdir", "/tmp/test1", "obs.yaml"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --workdir" in capsys.readouterr().err


def test_main_passes_parameters(monkeypatch):
    calls = []
    monkeypatch.setattr(numina.user.clirun, "mode_run_obsmode", lambda *args: calls.append(args))
    # main configures logging, that changes the loggers of the other tests
    monkeypatch.setattr(logging.config, "dictConfig", lambda conf: None)

    main(["--disable-plugins", "run", "--parameter-value=3", "obs.yaml"])

    assert len(calls) == 1
    args, extra_args, config = calls[0]
    assert args.obsresult == ["obs.yaml"]
    assert extra_args.extra_control == {"value": "3"}
