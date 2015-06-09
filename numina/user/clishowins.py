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


def show_instruments(args):
    for theins in args.drps.values():
        if not args.name or (theins.name in args.name):
            print_instrument(theins, modes=args.om)


def print_instrument(instrument, modes=True):
    print('Instrument:', instrument.name)
    for ic in instrument.configurations:
        print(' has configuration', repr(ic))
    for _, pl in instrument.pipelines.items():
        print(' has pipeline {0.name!r}, version {0.version}'.format(pl))
    if modes and instrument.modes:
        print(' has observing modes:')
        for mode in instrument.modes:
            print("  {0.name!r} ({0.key})".format(mode))

