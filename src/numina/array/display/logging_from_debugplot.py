#
# Copyright 2018-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import logging


def logging_from_debugplot(debugplot):
    """Set debugging level based on debugplot value.

    Parameters
    ----------
    debugplot : int
            Debugging level for messages and plots. For details see
            'numina.array.display.pause_debugplot.py'.
    """

    if isinstance(debugplot, int):
        if abs(debugplot) >= 10:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
    else:
        raise ValueError("Unexpected debugplot=" + str(debugplot))
