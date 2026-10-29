"""The functions of numina-wavecalib moved from __main__ to numina.array.wavecalib.core"""

import pytest

import numina.array.wavecalib.__main__ as wavecalib_main
import numina.array.wavecalib.core as core


@pytest.mark.parametrize(
    "name",
    ["collapsed_spectrum", "read_wv_master_from_array", "read_wv_master_file", "find_fxpeaks", "wvcal_spectrum"],
)
def test_moved_functions_deprecated(name):
    with pytest.warns(DeprecationWarning, match=f"use numina.array.wavecalib.core.{name}"):
        function = getattr(wavecalib_main, name)
    assert function is getattr(core, name)


def test_other_names():
    with pytest.raises(AttributeError):
        wavecalib_main.other
