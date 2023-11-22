
import pytest

from ..cli import main


def test_identify_run1(capsys):
    with pytest.raises(FileNotFoundError):
        main(['identify', 'r000001.fits'])
