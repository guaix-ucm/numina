#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, installation of calibrations.

``numina inittree INSTRUMENT`` downloads a dataset of calibrations
declared by the DRP of the instrument, and installs it in the directory of
calibrations (``calibsdir``) used by ``numina run``. Without instrument, it
lists the datasets of the DRPs.
"""

import logging
import os
import sys

_logger = logging.getLogger(__name__)


def register(subparsers, config):
    """Add the ``inittree`` command to the parser."""
    parser = subparsers.add_parser(
        "inittree",
        help="install the calibrations of an instrument",
        description="Download a dataset of calibrations declared by the DRP of an instrument, "
        "and install it in the directory of calibrations used by numina run. "
        "Without instrument, list the datasets of the DRPs.",
    )
    parser.set_defaults(command=inittree)
    parser.add_argument("instrument", nargs="?", help="name of the instrument, without it the datasets are listed")
    parser.add_argument("--dataset", help="name of the dataset, by default the last one declared by the DRP")
    parser.add_argument(
        "--base-path",
        default="calibrations",
        help="directory of calibrations, the files are installed under BASE_PATH/INSTRUMENT (default: calibrations)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace the files that exist with other contents")
    parser.add_argument("--dry-run", action="store_true", help="show what would be done, without changing anything")
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="do not set calibsdir in .numina.cfg of the current directory",
    )
    return parser


def error(msg):
    """Print an error message of ``numina inittree`` and return the exit status 2."""
    print(f"numina inittree: error: {msg}", file=sys.stderr)
    return 2


def list_datasets(drps):
    """Print the datasets of calibrations declared by each DRP and return 0."""
    found = False
    for name, drp in sorted(drps.query_all().items()):
        for dataset in getattr(drp, "calibrations", None) or []:
            found = True
            line = f"{name}  {dataset['name']}"
            if dataset.get("description"):
                line += f"  {dataset['description']}"
            print(line)
    if not found:
        print("no datasets of calibrations")
    return 0


def inittree(args, extra_args, config):
    """Install a dataset of calibrations of an instrument"""
    import numina.drps
    from numina.util.download import download_file

    from .inittree import extract, find_dataset, plan_install, update_config

    drps = numina.drps.get_system_drps()
    if args.instrument is None:
        return list_datasets(drps)
    try:
        drp = drps.query_by_name(args.instrument)
    except KeyError:
        return error(f"no DRP for instrument {args.instrument}")
    try:
        dataset = find_dataset(drp, args.dataset)
    except ValueError as err:
        return error(err)

    destination = os.path.join(args.base_path, dataset.get("directory", drp.name))
    print(f"dataset {dataset['name']} of {drp.name}, from {dataset['url']}")
    try:
        zip_path = download_file(dataset["url"], known_hash=dataset["known_hash"])
    except (OSError, ValueError) as err:
        return error(err)

    plan = plan_install(zip_path, destination)
    print(
        f"{len(plan.new)} new files, {len(plan.unchanged)} unchanged, "
        f"{len(plan.conflicts)} with other contents, in {destination}"
    )
    for _, target in plan.conflicts[:10]:
        print(f"  other contents: {target}")
    if len(plan.conflicts) > 10:
        print(f"  ... and {len(plan.conflicts) - 10} more")
    if plan.conflicts and not args.overwrite:
        return error("some files exist with other contents, nothing was installed, use --overwrite to replace them")

    cfg_path = ".numina.cfg"
    if args.dry_run:
        if not args.no_config:
            print(f"calibsdir would be set to {args.base_path} in {os.path.abspath(cfg_path)}")
        print("dry run, nothing was changed")
        return 0

    extract(zip_path, plan.new + plan.conflicts)
    print(f"installed {len(plan.new) + len(plan.conflicts)} files in {destination}")
    if not args.no_config:
        update_config(cfg_path, args.base_path, drp, dataset)
        print(f"calibsdir set to {args.base_path} in {os.path.abspath(cfg_path)}")
    return 0
