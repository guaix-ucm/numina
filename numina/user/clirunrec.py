#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import os

from .clirundal import mode_run_common


def register(subparsers, config):
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


def mode_run_recipe(args, extra_args):
    mode_run_common(args, extra_args, mode='rec')

