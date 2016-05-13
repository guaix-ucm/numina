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

from .clirundal import mode_run_common


def register(subparsers, config):
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
        default=None,
        help='name of an instrument configuration'
        )
    parser_run.add_argument(
        '-p', '--pipeline', dest='pipe_name',
        default='default', help='name of a pipeline'
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

    return parser_run

def mode_run_obsmode(args):
    mode_run_common(args, mode='obs')
