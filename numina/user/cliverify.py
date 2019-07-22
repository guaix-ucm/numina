#
# Copyright 2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, verify functionality."""

from __future__ import print_function

import os
import logging

from .helpers import create_datamanager, load_observations
from numina.util.context import working_directory


_logger = logging.getLogger(__name__)


def register(subparsers, config):

    task_control_base = config.get('run', 'task_control')

    parser_verify = subparsers.add_parser(
        'verify',
        help='verify a observation result'
    )

    parser_verify.set_defaults(command=verify)

    parser_verify.add_argument(
        '-c', '--task-control', dest='reqs', default=task_control_base,
        help='configuration file of the processing task', metavar='FILE'
    )
    parser_verify.add_argument(
        '-r', '--requirements', dest='reqs', default=task_control_base,
        help='alias for --task-control', metavar='FILE'
    )
    parser_verify.add_argument(
        '-i', '--instrument', dest='insconf',
        default=None,
        help='name of an instrument configuration'
    )
    parser_verify.add_argument(
        '--profile-path', dest='profilepath',
        default=None,
        help='location of the instrument profiles'
    )
    parser_verify.add_argument(
        '-p', '--pipeline', dest='pipe_name',
        default='default', help='name of a pipeline'
    )
    parser_verify.add_argument(
        '--basedir', action="store", dest="basedir",
        default=os.getcwd(),
        help='path to create the following directories'
    )
    parser_verify.add_argument(
        '--datadir', action="store", dest="datadir",
        default='data',
        help='path to directory containing pristine data'
    )
    parser_verify.add_argument(
        '--resultsdir', action="store", dest="resultsdir",
        help='path to directory to store results'
    )
    parser_verify.add_argument(
        '--workdir', action="store", dest="workdir",
        help='path to directory containing intermediate files'
    )
    parser_verify.add_argument(
        '--cleanup', action="store_true", dest="cleanup",
        default=False, help='cleanup workdir on exit [disabled]'
    )
    parser_verify.add_argument(
        '--not-copy-files', action="store_false", dest="copy_files",
        help='do not copy observation result and requirement files'
    )
    parser_verify.add_argument(
        '--link-files', action="store_false", dest="copy_files",
        help='do not copy observation result and requirement files'
    )
    parser_verify.add_argument(
        '--dump-control', action="store_true",
        help='save the modified task control file'
    )
    parser_verify.add_argument(
        '--session', action="store_true",
        help='use the obresult file as a session file'
    )
    parser_verify.add_argument(
        '--validate', action="store_true",
        help='validate inputs and results of recipes'
    )
    parser_verify.add_argument(
        'obsresult', nargs='+',
        help='file with the observation result'
    )

    return parser_verify


def verify(args, extra_args):

    # Loading observation result if exists
    sessions, loaded_obs = load_observations(args.obsresult, args.session)

    datamanager = create_datamanager(args.reqs, args.basedir, args.datadir, extra_args.extra_control)
    datamanager.backend.add_obs(loaded_obs)

    # Start processing
    jobs = []
    for session in sessions:
        for job in session:
            if job['enabled']:
                jobs.append(job)

    for job in jobs:
        run_verify(
            datamanager, job['id'], copy_files=args.copy_files,
            validate_inputs=args.validate, validate_results=args.validate
        )


def run_verify(datastore, obsid, as_mode=None, requirements=None, copy_files=False,
               validate_inputs=False, validate_results=False):
    """Verify raw images"""

    configuration = 'default'
    _logger.info("verify OB with id={}".format(obsid))

    # Roll back to cwd after leaving the context
    with working_directory(datastore.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(
            obsid, as_mode=as_mode, configuration=configuration
        )

        import json
        import numina.frame.schema as SC
        path1 = "/home/spr/devel/guaix/numina/schemas/image2.json"
        with open(path1) as fd:
            skd = json.load(fd)

        for f in obsres.frames:
            with f.open() as hdulist:
                print('verify', f)
                SC.validate(hdulist[0].header, schema=skd['hdus']['primary'])
                isbias = hdulist[0].header['OBSMODE'] == 'MegaraBiasImage'
                if isbias:
                    pass
                else:
                    SC.validate(hdulist['fibers'].header, schema=skd['hdus']['fibers'])
        return 0


        from jsonschema import validate

        path1 = "/home/spr/devel/guaix/numina/schemas/image1.json"
        with open(path1) as fd:
            schema = json.load(fd)

        for f in obsres.frames:
            with f.open() as hdulist:
                print('verify', f)
                mm = convert_headers(hdulist)
                validate(mm, schema=schema)
                print(mm)

        header_conv = []


def convert_headers(hdulist):
    headers = [convert_header(hdu.header) for hdu in hdulist]
    return headers


def convert_header(header):
    hdu_v = {}
    hdu_c = {}
    hdu_o = []
    hdu_repr = {'values': hdu_v, 'comments': hdu_c, 'ordering': hdu_o}

    for card in header.cards:
        key = card.keyword
        value = card.value
        comment = card.comment
        hdu_v[key] = value
        hdu_c[key] = comment
        hdu_o.append(key)

    return hdu_repr