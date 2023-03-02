
import pytest


def test_warns_qc():

    with pytest.warns(DeprecationWarning):
        import numina.core.qc


def test_warns_products():

    with pytest.warns(DeprecationWarning):
        import numina.core.products