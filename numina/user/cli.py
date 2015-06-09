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

import logging.config
import argparse
import os

from six.moves import configparser

from numina import __version__
from numina.core import init_drp_system

from .xdgdirs import xdg_config_home
from .logconf import numina_cli_logconf

from .clishowins import show_instruments
from .clishowom import show_observingmodes
from .clishowrecip import show_recipes
from .clirun import mode_run_obsmode, mode_run_recipe

_logger = logging.getLogger("numina")


def main(args=None):
    '''Entry point for the Numina CLI.'''

    # Configuration args from a text file
    config = configparser.SafeConfigParser()

    # Building programatically
    config.add_section('numina')
    config.set('numina', 'format', 'yaml')

    # Custom values, site wide and local
    config.read(['.numina/numina.cfg',
                 os.path.join(xdg_config_home, 'numina/numina.cfg')])

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

    parser_show_rec = subparsers.add_parser(
        'show-recipes',
        help='show information of recipes'
        )

    parser_show_rec.set_defaults(command=show_recipes, template=False)

    parser_show_rec.add_argument(
        '-i', '--instrument',
        help='filter recipes by instrument'
        )
    parser_show_rec.add_argument(
        '-t', '--template', action='store_true',
        help='generate requirements YAML template'
        )

#    parser_show_rec.add_argument('--output', type=argparse.FileType('wb', 0))

    parser_show_rec.add_argument(
        'name', nargs='*', default=None,
        help='filter recipes by name'
        )

    parser_run = subparsers.add_parser(
        'run',
        help='process a observation result'
        )

    parser_run.set_defaults(command=mode_run_obsmode)

    parser_run.add_argument(
        '-c', '--task-control', dest='reqs',
        help='configuration file of the processing task', metavar='FILE'
        )
    parser_run.add_argument(
        '-r', '--requirements', dest='reqs',
        help='alias for --task-control', metavar='FILE'
        )
    parser_run.add_argument(
        '-i', '--instrument', dest='insconf',
        default="default",
        help='name of an instrument configuration'
        )
    parser_run.add_argument(
        '-p', '--pipeline', dest='pipe_name',
        help='name of a pipeline'
        )
    parser_run.add_argument(
        '--basedir', action="store", dest="basedir",
        default=os.getcwd(),
        help='path to create the following directories'
        )
    parser_run.add_argument(
        '--datadir', action="store", dest="datadir",
        help='path to directory containing pristine data'
        )
    parser_run.add_argument(
        '--resultsdir', action="store", dest="resultsdir",
        help='path to directory to store results'
        )
    parser_run.add_argument(
        '--workdir', action="store", dest="workdir",
        help='path to directory containing intermediate files'
        )
    parser_run.add_argument(
        '--cleanup', action="store_true", dest="cleanup",
        default=False, help='cleanup workdir on exit [disabled]'
        )
    parser_run.add_argument(
        'obsresult',
        help='file with the observation result'
        )

    parser_run_recipe = subparsers.add_parser(
        'run-recipe',
        help='run a recipe'
        )

    parser_run_recipe.set_defaults(command=mode_run_recipe)
    parser_run_recipe.add_argument(
        '-c', '--task-control', dest='reqs',
        help='configuration file of the processing task', metavar='FILE'
        )
    parser_run_recipe.add_argument(
        '-r', '--requirements', dest='reqs',
        help='alias for --task-control', metavar='FILE'
        )
    parser_run_recipe.add_argument(
        '--obs-res', dest='obsresult',
        help='observation result', metavar='FILE'
        )
    parser_run_recipe.add_argument(
        'recipe',
        help='fqn of the recipe'
        )
    parser_run_recipe.add_argument(
        '--basedir', action="store", dest="basedir",
        default=os.getcwd(),
        help='path to create the following directories'
        )
    parser_run_recipe.add_argument(
        '--datadir', action="store", dest="datadir",
        help='path to directory containing pristine data'
        )
    parser_run_recipe.add_argument(
        '--resultsdir', action="store", dest="resultsdir",
        help='path to directory to store results'
        )
    parser_run_recipe.add_argument(
        '--workdir', action="store", dest="workdir",
        help='path to directory containing intermediate files'
        )
    parser_run_recipe.add_argument(
        '--cleanup', action="store_true", dest="cleanup",
        default=False, help='cleanup workdir on exit [disabled]'
        )

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

    _logger = logging.getLogger("numina")
    _logger.info('Numina simple recipe runner version %s', __version__)

    args.drps = init_drp_system()

    args.command(args)

