#
# Copyright 2008-2016 Universidad Complutense de Madrid
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

"""User command line interface of Numina, show-instruments functionallity."""

from __future__ import print_function
from numina.core.pipeline import DrpSystem


def register(subparsers, config):
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


def show_instruments(args, extra_args):
    mm = DrpSystem()

    if args.name:
        for name in args.name:
            drp = mm.query_by_name(name)
            if drp:
                print_instrument(drp, modes=args.om)
            else:
                print_no_instrument(name)
    else:
        drps = mm.query_all()
        for drp in drps.values():
            print_instrument(drp, modes=args.om)

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


def print_no_instrument(name):
    print('No instrument named:', name)


