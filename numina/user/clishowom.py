#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

'''User command line interface of Numina.'''

from __future__ import print_function


def show_observingmodes(args):
    for theins in args.drps.values():
        if not args.instrument or (args.instrument == theins.name):
            for mode in theins.modes:
                if not args.name or (mode.key in args.name):
                    print_obsmode(mode, theins)


def print_obsmode(obsmode, instrument, ins=False):
    print('Observing Mode: {0.name!r} ({0.key})'.format(obsmode))
    print(' summary:', obsmode.summary)
    print(' instrument:', instrument.name)

