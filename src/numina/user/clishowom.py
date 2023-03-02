#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

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
        try:
            val = drpsys.query_by_name(name)
        except KeyError:
            val = None
        res = [(name, val)]
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
    print(f'Observing Mode: {obsmode.name!r} ({obsmode.key})')
    print(' summary:', obsmode.summary)
    print(' instrument:', instrument.name)
