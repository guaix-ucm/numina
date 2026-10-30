#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""

import logging

from .baserun import run_reduce
from .helpers import ReductionRequests, create_datamanager, load_observations

_logger = logging.getLogger(__name__)


def mode_run_common(args, extra_args, config, mode):
    """Run the reduction in mode ``obs`` (observing blocks) or ``rec`` (recipe, not implemented).

    Raises
    ------
    ValueError
        If ``mode`` is not ``obs`` or ``rec``.
    """
    # FIXME: implement 'recipe' run mode
    if mode == "rec":
        print("Mode not implemented yet")
        return 1
    elif mode == "obs":
        return mode_run_common_obs(args, extra_args, config)
    else:
        raise ValueError(f"Not valid run mode {mode}")


def configure_run(args, config):
    """Update the configuration with the options of the command line"""
    if args.basedir:
        config["tool.run"]["basedir"] = args.basedir
    if args.datadir:
        config["tool.run"]["datadir"] = args.datadir
    if args.calibsdir:
        config["tool.run"]["calibsdir"] = args.calibsdir
    if args.db:
        if not config.has_section("tool.db"):
            config.add_section("tool.db")
        config["tool.db"]["file"] = args.db
    if args.copy_files is not None:
        config["tool.run"]["copy_files"] = str(args.copy_files)
    if args.validate:
        config["tool.run"]["validate"] = str(args.validate)
    if hasattr(args, "strict_reqs"):
        config["tool.run"]["strict_reqs"] = str(args.strict_reqs)


def run_oblocks(datamanager, oblock_ids, args, config, requests=None):
    """Reduce the observing blocks, with the options of the command line

    `requests` are the reduction requests (ReductionRequests) of the blocks.
    The pipeline and the profile of the command line override those of the
    requests.
    """
    copy_files = config["tool.run"].getboolean("copy_files")
    validate = config["tool.run"].getboolean("validate")
    strict_inputs = config["tool.run"].getboolean("strict_reqs")
    for oblock_id in oblock_ids:
        kwargs = {} if requests is None else requests.arguments(oblock_id)
        if args.pipe_name is not None:
            kwargs["pipeline"] = args.pipe_name
        if args.profile is not None:
            kwargs["profile"] = args.profile
        run_reduce(
            datamanager,
            oblock_id,
            copy_files=copy_files,
            validate_inputs=validate,
            validate_results=validate,
            strict_inputs=strict_inputs,
            **kwargs,
        )


def mode_run_common_obs(args, extra_args, config):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    sessions, loaded_obs = load_observations(args.obsresult)

    configure_run(args, config)
    datamanager = create_datamanager(config, args.reqs, extra_args.extra_control, profile_path_extra=args.profilepath)
    datamanager.backend.add_obs(loaded_obs)

    # Start processing
    jobs = []
    for session in sessions:
        for job in session:
            if job["enabled"] or job["id"] in args.enable:
                jobs.append(job["id"])

    requests = None if args.request is None else ReductionRequests.load(args.request)
    try:
        run_oblocks(datamanager, jobs, args, config, requests=requests)
    finally:
        if args.dump_control:
            _logger.debug("dump control status")
            with open("control_dump.yaml", "w") as fp:
                datamanager.backend.dump(fp)
