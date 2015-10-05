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
from numina.core import init_drp_system

def add(subparsers):
    parser_show_ins = subparsers.add_parser(
        'show-instruments',
        help='show registered instruments'
        )

    parser_show_ins.set_defaults(command=show_instruments,
                                 verbose=0, what='om')
    parser_show_ins.add_argument(
        '-o', '--observing-modes',
        action='store_true', dest='om',
        help='list observing modes of each instrument')

#    parser_show_ins.add_argument('--verbose', '-v', action='count')

    parser_show_ins.add_argument(
        'name', nargs='*', default=None,
        help='filter instruments by name'
        )

    return parser_show_ins


def show_instruments(args):
    drps = init_drp_system()
    for theins in drps.values():
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

