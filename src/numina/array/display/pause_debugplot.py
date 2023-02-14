#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import sys

from numina.array.display.matplotlib_qt import plt

DEBUGPLOT_CODES = (0, -1, 1, -2, 2, -10, 10, -11, 11, -12, 12,
                   -21, 21, -22, 22)


def pause_debugplot(debugplot, optional_prompt=None, pltshow=False,
                    tight_layout=True):
    """Ask the user to press RETURN to continue after plotting.

    Parameters
    ----------
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
        21 : debug, extra plots without pauses
        22 : debug, extra plots with pause
        NOTE: negative values are also valid and indicate that a call
        to plt.close() is also performed
    optional_prompt : string
        Optional prompt.
    pltshow : bool
        If True, a call to plt.show() is also performed.
    tight_layout : bool
        If True, and pltshow=True, a call to plt.tight_layout() is
        also performed.

    """

    if debugplot not in DEBUGPLOT_CODES:
        raise ValueError('Invalid debugplot value:', debugplot)

    if debugplot < 0:
        debugplot_ = -debugplot
        pltclose = True
    else:
        debugplot_ = debugplot
        pltclose = False

    if pltshow:
        if debugplot_ in [1, 2, 11, 12, 21, 22]:
            if tight_layout:
                plt.tight_layout()
            if debugplot_ in [1, 11, 21]:
                plt.show(block=False)
                plt.pause(0.2)
            elif debugplot_ in [2, 12, 22]:
                print('Press "q" to continue...', end='')
                sys.stdout.flush()
                plt.show()
                print('')
    else:
        if debugplot_ in [2, 12, 22]:
            if optional_prompt is None:
                print('Press <RETURN> to continue...', end='')
            else:
                print(optional_prompt, end='')
            sys.stdout.flush()
            cdummy = sys.stdin.readline().strip()

    if debugplot_ in [1, 2, 11, 12, 21, 22] and pltclose:
        plt.close()
