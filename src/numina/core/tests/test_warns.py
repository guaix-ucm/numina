
import pytest


def test_warns_qc():

    with pytest.warns(DeprecationWarning):
        import numina.core.qc  # noqa: F401


def test_warns_products():

    with pytest.warns(DeprecationWarning):
        import numina.core.products  # noqa: F401
