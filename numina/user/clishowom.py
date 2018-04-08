#
# Copyright 2008-2018 Universidad Complutense de Madrid
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

"""User command line interface of Numina."""

from __future__ import print_function

import numina.drps
from numina.user.clishowins import print_no_instrument


def register(subparsers, config):
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


def show_observingmodes(args, extra_args):

    drpsys = numina.drps.get_system_drps()

    if args.instrument:
        name = args.instrument
        res = [(name, drpsys.query_by_name(name))]
    else:
        res = drpsys.query_all().items()

    for name, theins in res:
        if theins:
            for mode in theins.modes.values():
                if not args.name or (mode.key in args.name):
                    print_obsmode(mode, theins)
        else:
            print_no_instrument(name)


def print_obsmode(obsmode, instrument, ins=False):
    print('Observing Mode: {0.name!r} ({0.key})'.format(obsmode))
    print(' summary:', obsmode.summary)
    print(' instrument:', instrument.name)
