#
# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, show-instruments functionallity."""

from __future__ import print_function

import numina.drps
import numina.core.objimport as objimport

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

    mm = numina.drps.get_system_drps()

    if args.name:
        for name in args.name:
            try:
                drp = mm.query_by_name(name)
                print_instrument(drp, modes=args.om)
            except KeyError:
                print_no_instrument(name)
    else:
        drps = mm.query_all()
        for drp in drps.values():
            print_instrument(drp, modes=args.om)

def print_instrument(instrument, modes=True):
    print('Instrument:', instrument.name)
    for ic, conf in instrument.configurations.items():
        if ic == 'default':
            # skip default configuration
            continue
        msg = " has configuration '{}' uuid={}".format(conf.name, ic)
        print(msg)
    default_conf = instrument.configurations['default']
    msg = " default is '{}'".format(default_conf.name)
    print(msg)
    print(" has datamodel '{}'".format(objimport.fully_qualified_name(instrument.datamodel)))
    for _, pl in instrument.pipelines.items():
        print(' has pipeline {0.name!r}, version {0.version}'.format(pl))
    if modes and instrument.modes:
        print(' has observing modes:')
        for mode in instrument.modes:
            print("  {0.name!r} ({0.key})".format(mode))


def print_no_instrument(name):
    print('No instrument named:', name)


