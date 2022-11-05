#
# Copyright 2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest


def _no_bench_func(method, *args, **kwds):
    return method(*args, **kwds)


@pytest.fixture
def benchmark():
    """Mock function used if pytest-benchmark is not installed.
    
    For actual benchmarking install pytest-benchmark
    """
    return _no_bench_func
   
