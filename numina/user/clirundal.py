#
# Copyright 2008-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina."""


import logging

from .baserun import run_reduce
from .helpers import create_datamanager, load_observations

_logger = logging.getLogger(__name__)


def mode_run_common(args, extra_args, mode):
    # FIXME: implement 'recipe' run mode
    if mode == 'rec':
        print('Mode not implemented yet')
        return 1
    elif mode == 'obs':
        return mode_run_common_obs(args, extra_args)
    else:
        raise ValueError(f'Not valid run mode {mode}')


def mode_run_common_obs(args, extra_args):
    """Observing mode processing mode of numina."""

    # Loading observation result if exists
    sessions, loaded_obs = load_observations(args.obsresult, args.session)

    datamanager = create_datamanager(args.reqs, args.basedir, args.datadir, extra_args.extra_control)
    datamanager.backend.add_obs(loaded_obs)

    # Start processing
    jobs = []
    for session in sessions:
        for job in session:
            if job['enabled'] or job['id'] in args.enable:
                jobs.append(job)

    for job in jobs:

        run_reduce(
            datamanager, job['id'], copy_files=args.copy_files,
            validate_inputs=args.validate, validate_results=args.validate
        )

    if args.dump_control:
        _logger.debug('dump control status')
        with open('control_dump.yaml', 'w') as fp:
            datamanager.backend.dump(fp)
