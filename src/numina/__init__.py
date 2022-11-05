#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Numina data processing system."""


import logging

from .version import version

__version__ = version

# Top level NullHandler
logging.getLogger("numina").addHandler(logging.NullHandler())
