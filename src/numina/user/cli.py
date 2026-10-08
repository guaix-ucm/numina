#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import argparse
import configparser
from importlib import import_module
from importlib.metadata import entry_points
import importlib.resources
import logging
import logging.config
import os
import sys


import yaml


from numina import __version__
from .xdgdirs import xdg_config_home
from .logconf import LOGCONF

_logger = logging.getLogger("numina")


def base_config():
    """Configuration with the default values in numina.cfg

    The configuration files of the user are not read.
    """
    config = configparser.ConfigParser()
    basecfg = importlib.resources.files("numina.user").joinpath("numina.cfg")
    with basecfg.open() as fd:
        config.read_file(fd)
    return config


def load_config():
    """Configuration of numina, as used by the command line

    The default values in numina.cfg, updated with the configuration
    files of the user, $XDG_CONFIG_HOME/numina/numina.cfg and
    .numina.cfg in the current directory, if they exist.
    """
    config = base_config()
    read_files = config.read([os.path.join(xdg_config_home, "numina/numina.cfg"), ".numina.cfg"])
    _logger.debug(f"Reading config files {read_files}")
    return config


def main(args=None):
    """Entry point for the Numina CLI."""

    # Configuration args from a text file
    config = load_config()

    parser0 = argparse.ArgumentParser(
        description="Command line interface of Numina",
        prog="numina",
        epilog="For detailed help pass --help to a target",
        add_help=False,
    )

    parser0.add_argument("--disable-plugins", action="store_true")

    args0, args = parser0.parse_known_args(args)

    # Load plugin commands if enabled
    subcmd_load = []

    if not args0.disable_plugins:
        for entry in entry_points(group="numina.plugins.1"):
            try:
                register = entry.load()
                subcmd_load.append(register)
            except Exception as error:
                print(f"exception loading plugin {entry}", file=sys.stderr)
                print(error, file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="Command line interface of Numina",
        prog="numina",
        epilog="For detailed help pass --help to a target",
    )

    parser.add_argument("--disable-plugins", action="store_true", help="disable plugin loading")
    parser.add_argument("-l", action="store", dest="logging", metavar="FILE", help="FILE with logging configuration")

    parser.add_argument("-c", action="store", dest="config", metavar="FILE", help="FILE with configuration")

    parser.add_argument("-d", "--debug", action="store_true", dest="debug", default=False, help="make lots of noise")

    parser.add_argument(
        "--standalone",
        action="store_true",
        dest="standalone",
        default=False,
        help="do not activate GTC compatibility code",
    )

    # Due to a problem with argparse
    # this command blocks the defaults of the subparsers
    # https://github.com/python/cpython/issues/53597
    # parser.set_defaults(command=None)

    subparsers = parser.add_subparsers(
        title="Targets", description="These are valid commands you can ask numina to do."
    )

    # Init subcommands
    cmds = ["clidentify", "clishowins", "clishowom", "clishowrecip", "clirun", "clirunrec"]
    for cmd in cmds:
        cmd_mod = import_module(f".{cmd}", "numina.user")
        register = getattr(cmd_mod, "register", None)
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
    if extra_args.rejected:
        parser.error(f"unrecognized arguments: {' '.join(extra_args.rejected)}")
    # logger file
    if args.standalone:
        import numina.ext.gtc

        numina.ext.gtc.ignore_gtc_check()

    # Config file from command line
    if args.config is not None:
        config.read_file(open(args.config))

    try:
        if args.logging is not None:
            loggingf = args.logging
        else:
            loggingf = config.get("numina", "logging")

        with open(loggingf) as logfile:
            logconf = yaml.safe_load(logfile)
            logging.config.dictConfig(logconf)
    except configparser.Error:
        logging.config.dictConfig(LOGCONF)

    if args.debug:
        _logger.setLevel(logging.DEBUG)
        for h in _logger.handlers:
            h.setLevel(logging.DEBUG)

    _logger.info(f"Numina simple recipe runner version {__version__}")
    command = getattr(args, "command", None)

    if command is not None:
        args.command(args, extra_args, config)


def process_unknown_arguments(unknowns):
    """Process arguments unknown to the parser

    Arguments like --parameter-NAME=VALUE are stored in extra_control,
    any other argument is stored in rejected.
    """

    result = argparse.Namespace()
    result.extra_control = {}
    result.rejected = []
    # It would be interesting to use argparse internal
    # machinery for this
    prefix = "--parameter-"
    for unknown in unknowns:
        if unknown.startswith(prefix) and "=" in unknown:
            # The value can contain '='
            key, val = unknown[len(prefix) :].split("=", 1)
            if key:
                result.extra_control[key] = val
                continue
        result.rejected.append(unknown)
    return result
