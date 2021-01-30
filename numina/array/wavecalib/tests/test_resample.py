from ..resample import rebin

import numpy as np


def test_rebin():
    a = np.ones((20,40))
    res = np.ones((5, 10))
    assert np.allclose(rebin(a, 5, 10), res)
