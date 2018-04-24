#
# Copyright 2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


def test_compat_datatype():

    import numina.core.types.datatype as dcompat
    import numina.types.datatype as mod

    assert mod is dcompat
