from ..cli import main


def test_identify_run1(capsys):
    expected = ["", "IDENTIFY"]
    main(['identify', 'r000001.fits'])
    out, err = capsys.readouterr()
    out = out.split("\n")
    out.sort()
    expected.sort()
    assert out == expected
