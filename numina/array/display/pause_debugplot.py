#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt


def pause_debugplot(debugplot, optional_prompt=None, pltshow=False):
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
    optional_prompt : string
        Optional prompt.
    pltshow : bool
        If True, a call to plt.show() is also performed.

    """

    if debugplot in [1, 2, 11, 12, 21, 22] and pltshow:
        plt.show(block=False)
        plt.pause(0.001)

    if debugplot in [2, 12, 22]:
        try:
            if optional_prompt is not None:
                input(optional_prompt)
            else:
                input("\nPress RETURN to continue...")
        except SyntaxError:
            pass

        print(' ')
