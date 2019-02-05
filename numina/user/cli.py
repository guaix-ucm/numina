#
# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

from __future__ import print_function

import logging
import logging.config
import argparse
import os
import sys
from importlib import import_module

import six.moves.configparser as configparser
import pkg_resources
import yaml

from numina import __version__

from .xdgdirs import xdg_config_home
from .logconf import numina_cli_logconf

_logger = logging.getLogger("numina")


def main(args=None):
    """Entry point for the Numina CLI."""

    # Configuration args from a text file
    config = configparser.SafeConfigParser()

    # Building programatically
    config.add_section('numina')
    config.set('numina', 'format', 'yaml')

    # Custom values
    config.read([
        os.path.join(xdg_config_home, 'numina/numina.cfg'),
        '.numina.cfg'
    ])

    parser0 = argparse.ArgumentParser(
        description='Command line interface of Numina',
        prog='numina',
        epilog="For detailed help pass --help to a target",
        add_help=False
        )

    parser0.add_argument('--disable-plugins', action='store_true')

    args0, args = parser0.parse_known_args(args)

    # Load plugin commands if enabled
    subcmd_load = []

    if not args0.disable_plugins:
        for entry in pkg_resources.iter_entry_points(
                group='numina.plugins.1'
        ):
            try:
                register = entry.load()
                subcmd_load.append(register)
            except Exception as error:
                print(error, file=sys.stderr)

    parser = argparse.ArgumentParser(
        description='Command line interface of Numina',
        prog='numina',
        epilog="For detailed help pass --help to a target"
        )

    parser.add_argument('--disable-plugins', action='store_true',
                        help='disable plugin loading')
    parser.add_argument(
        '-l', action="store", dest="logging", metavar="FILE",
        help="FILE with logging configuration"
        )
    parser.add_argument(
        '-d', '--debug',
        action="store_true",
        dest="debug", default=False,
        help="make lots of noise"
        )

    parser.add_argument(
        '--standalone',
        action="store_true",
        dest="standalone", default=False,
        help="do not activate GTC compatibility code"
        )

    # Due to a problem with argparse
    # this command blocks the defaults of the subparsers
    # in argparse of Pyhton < 2.7.9
    # https://bugs.python.org/issue9351
    # parser.set_defaults(command=None)

    subparsers = parser.add_subparsers(
        title='Targets',
        description='These are valid commands you can ask numina to do.'
        )

    # Init subcommands
    cmds = ['clishowins', 'clishowom', 'clishowrecip',
            'clirun', 'clirunrec']
    for cmd in cmds:
        cmd_mod = import_module('.%s' % (cmd, ), 'numina.user')
        register = getattr(cmd_mod, 'register', None)
        if register is not None:
            register(subparsers, config)

    # Add commands by plugins
    for register in subcmd_load:
        try:
            register(subparsers, config)
        except Exception as error:
            print(error, file=sys.stderr)

    args, unknowns = parser.parse_known_args(args)

    extra_args = process_unknown_arguments(unknowns)
    # logger file
    if args.standalone:
        import numina.ext.gtc
        numina.ext.gtc.ignore_gtc_check()

    try:
        if args.logging is not None:
            loggingf = args.logging
        else:
            loggingf = config.get('numina', 'logging')

        with open(loggingf) as logfile:
            logconf = yaml.load(logfile)
            logging.config.dictConfig(logconf)
    except configparser.Error:
        logging.config.dictConfig(numina_cli_logconf)

    _logger.debug('Numina simple recipe runner version %s', __version__)
    if args.command:
        args.command(args, extra_args)


def process_unknown_arguments(unknowns):
    """Process arguments unknown to the parser"""

    result = argparse.Namespace()
    result.extra_control = {}
    # It would be interesting to use argparse internal
    # machinery for this
    for unknown in unknowns:
        # Check prefixes
        prefix = '--parameter-'
        if unknown.startswith(prefix):
            # process '='
            values = unknown.split('=')
            if len(values) == 2:
                key = values[0][len(prefix):]
                val = values[1]
                if key:
                    result.extra_control[key] = val
    return result
