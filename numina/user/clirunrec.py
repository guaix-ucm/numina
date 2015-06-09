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

import os

from .cliruncommon import mode_run_common

def add(subparsers):
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

    return parser_run_recipe

def mode_run_recipe(args):
    mode_run_common(args, mode='rec')

