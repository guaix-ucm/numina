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

"""User command line interface of Numina."""

import logging
import logging.config
import argparse
import os
from importlib import import_module

import six.moves.configparser as configparser
import pkg_resources

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

    parser = argparse.ArgumentParser(
        description='Command line interface of Numina',
        prog='numina',
        epilog="For detailed help pass --help to a target"
        )

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

    # Load plugin commands
    for entry in pkg_resources.iter_entry_points(group='numina_plugins.1'):
        register = entry.load()
        try:
            register(subparsers, config)
        except StandardError as error:
            print(error)

    args = parser.parse_args(args)

    # logger file
    if args.logging is not None:
        logging.config.fileConfig(args.logging)
    else:
        # This should be a default path in defaults.cfg
        try:
            args.logging = config.get('numina', 'logging')
            logging.config.fileConfig(args.logging)
        except configparser.Error:
            logging.config.dictConfig(numina_cli_logconf)

    _logger.info('Numina simple recipe runner version %s', __version__)

    args.command(args)

