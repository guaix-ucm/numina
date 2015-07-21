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
    parser_show_mode = subparsers.add_parser(
        'show-modes',
        help='show information of observing modes'
        )

    parser_show_mode.set_defaults(
        command=show_observingmodes,
        verbose=0,
        what='om'
        )
#    parser_show_mode.add_argument('--verbose', '-v', action='count')
    parser_show_mode.add_argument(
        '-i', '--instrument',
        help='filter modes by instrument'
        )

    parser_show_mode.add_argument(
        'name', nargs='*', default=None,
        help='filter observing modes by name'
        )

    return parser_show_mode


def show_observingmodes(args):

    drps = init_drp_system()

    for theins in drps.values():
        if not args.instrument or (args.instrument == theins.name):
            for mode in theins.modes:
                if not args.name or (mode.key in args.name):
                    print_obsmode(mode, theins)


def print_obsmode(obsmode, instrument, ins=False):
    print('Observing Mode: {0.name!r} ({0.key})'.format(obsmode))
    print(' summary:', obsmode.summary)
    print(' instrument:', instrument.name)

