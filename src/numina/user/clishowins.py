#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, show-instruments functionallity."""

import numina.drps
import numina.util.fqn as objimport
import numina.instrument.assembly as asbl


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
    # TODO: this could be the same option for run
    parser_show_ins.add_argument(
        '--profile-path', dest='profilepath',
        default=None,
        help='location of the instrument profiles'
        )
#    parser_show_ins.add_argument('--verbose', '-v', action='count')

    parser_show_ins.add_argument(
        'name', nargs='*', default=None,
        help='filter instruments by name'
        )

    return parser_show_ins


def show_instruments(args, extra_args):

    sys_drps = numina.drps.get_system_drps()
    prof_store = asbl.load_panoply_store(sys_drps, args.profilepath)

    if args.name:
        for name in args.name:
            try:
                drp = sys_drps.query_by_name(name)
                print_instrument(drp, prof_store, modes=args.om)
            except KeyError:
                print_no_instrument(name)
    else:
        drps = sys_drps.query_all()
        for drp in drps.values():
            print_instrument(drp, prof_store, modes=args.om)


def print_instrument(instrument, prof_store, modes=True):
    print('Instrument:', instrument.name)
    print(f" version is '{instrument.version}'")
    for key, val in prof_store.items():
        etype = val['type']
        name = val['name']
        if etype == 'instrument' and name == instrument.name:
            desc = val['description']
            uuid = val['uuid']
            msg = f" has configuration '{desc}' uuid={uuid}"
            print(msg)

    print(f" has datamodel '{objimport.fully_qualified_name(instrument.datamodel)}'")
    for _, pl in instrument.pipelines.items():
        print(f' has pipeline {pl.name!r}, version {pl.version}')
    if modes and instrument.modes:
        print(' has observing modes:')
        for mode in instrument.modes.values():
            print(f"  {mode.name!r} ({mode.key})")


def print_no_instrument(name):
    print('No instrument named:', name)


