"""The CLI starts faster if the heavy modules are imported only when needed"""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["matplotlib", "scipy.interpolate", "scipy.ndimage"])
def test_cli_does_not_import(module):
    code = f"import sys, numina.user.cli, numina.user.clirun; print({module!r} in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize(
    "module",
    [
        "numina.array.wavecalib.arccalibration",
        "numina.array.wavecalib.crosscorrelation",
        "numina.array.wavecalib.core",
    ],
)
def test_wavecalib_does_not_import_matplotlib(module):
    code = f"import sys, {module}; print('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_matplotlib_qt_plt():
    import matplotlib.pyplot

    from numina.array.display import matplotlib_qt
    from numina.array.display.matplotlib_qt import plt, set_window_geometry

    assert plt is matplotlib.pyplot
    assert matplotlib_qt.plt is plt
    set_window_geometry(None)
    with pytest.raises(AttributeError):
        matplotlib_qt.other
