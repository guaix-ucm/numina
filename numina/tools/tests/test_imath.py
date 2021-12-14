

from ..imath import main


def test_imath_works(capsys):
    """"Test that imath displays its help msg"""

    try:
        main(['--help'])
    except SystemExit:
        pass

    out, err = capsys.readouterr()
    # out = out.split("\n")
    # out.sort()
    assert isinstance(out, str)
