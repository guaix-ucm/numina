#
# Copyright 2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, verify functionality."""

import logging
import os

from .helpers import create_datamanager, load_observations
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


def register(subparsers, config):
    parser_identify = subparsers.add_parser(
        'identify',
        help='identify'
    )
    parser_identify.set_defaults(command=identify)
    parser_identify.add_argument(
        'images',
        help='identify files in CL'
    )

    return parser_identify


def identify(args, extra_args):
    print("IDENTIFY")
    return 0


def run_identify(datastore, obsid, as_mode=None, requirements=None, copy_files=False,
               validate_inputs=False, validate_results=False):
    """Identify raw images"""
    print("run identify")
