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
import os
import uuid

from .clirundal import mode_run_common


def profile_uuid(value):
    """Check that the value of --profile is a UUID"""
    try:
        return str(uuid.UUID(value))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid UUID")


def complete_config(config):
    """Complete config with default values"""
    secname = "tool.run"
    if not config.has_section(secname):
        config.add_section(secname)

    values = {
        "basedir": os.getcwd(),
    }

    for k, v in values.items():
        if not config.has_option(secname, k):
            config.set(secname, k, v)

    return config


def register(subparsers, config):
    complete_config(config)
    parser_run = subparsers.add_parser("run", help="process a observation result")

    parser_run.add_argument(
        "-c",
        "--task-control",
        dest="reqs",
        help="configuration file of the processing task",
        metavar="FILE",
    )
    parser_run.add_argument(
        "-r",
        "--requirements",
        dest="reqs",
        help="alias for --task-control",
        metavar="FILE",
    )
    parser_run.add_argument(
        "--insconf",
        "--profile",
        dest="profile",
        type=profile_uuid,
        default=None,
        metavar="UUID",
        help="uuid of an instrument configuration (INSCONF), overrides the configuration selected from the images",
    )
    parser_run.add_argument(
        "--profile-path",
        dest="profilepath",
        default=None,
        help="directory with additional instrument configurations",
    )
    parser_run.add_argument(
        "-p",
        "--pipeline",
        dest="pipe_name",
        default=None,
        help="name of a pipeline, overrides the pipeline of the observation result ('default' if not defined)",
    )
    parser_run.add_argument(
        "--basedir",
        action="store",
        dest="basedir",
        help="path to create the following directories",
    )
    parser_run.add_argument(
        "--datadir",
        action="store",
        dest="datadir",
        help="path to directory containing pristine data",
    )
    parser_run.add_argument(
        "--calibsdir",
        "--rootdir",
        action="store",
        dest="calibsdir",
        help="path to directory containing calibration data",
        metavar="CALIBSDIR",
    )

    # If none is used, the value of copy_files in the configuration is used
    group_copy = parser_run.add_mutually_exclusive_group()
    group_copy.add_argument(
        "--copy-files",
        action="store_true",
        dest="copy_files",
        default=None,
        help="copy observation result and requirement files to the work directory",
    )
    group_copy.add_argument(
        "--link-files",
        "--not-copy-files",
        action="store_false",
        dest="copy_files",
        default=None,
        help="link observation result and requirement files in the work directory",
    )
    parser_run.add_argument(
        "-e",
        "--enable",
        action="append",
        default=[],
        metavar="BLOCKID",
        help="enable blocks by id",
    )
    parser_run.add_argument(
        "--dump-control",
        action="store_true",
        help="save the modified task control file",
    )
    parser_run.add_argument("--session", action="store_true", help="use the obresult file as a session file")
    parser_run.add_argument(
        "--validate",
        action="store_const",
        const=True,
        help="validate inputs and results of recipes",
    )

    group_strict = parser_run.add_mutually_exclusive_group()
    group_strict.add_argument(
        "--strict-reqs",
        action="store_true",
        help="end the program if a requirement name is not matched by the recipe",
    )
    group_strict.add_argument(
        "--loose-reqs",
        action="store_false",
        dest="strict_reqs",
        help="warn if a requirement name is not matched by the recipe",
    )
    parser_run.add_argument("obsresult", nargs="+", help="file with the observation result")

    parser_run.set_defaults(command=mode_run_obsmode, strict_reqs=True)

    return parser_run


def mode_run_obsmode(args, extra_args, config):
    mode_run_common(args, extra_args, config, mode="obs")
