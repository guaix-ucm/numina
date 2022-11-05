#
# Copyright 2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


class Namespace(object):
    """Namespace class, like argparse.Namespace

    >>> nm = Namespace(a=1, b="field")
    >>> nm.a == 1
    >>> nm.b == "field"
    
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
