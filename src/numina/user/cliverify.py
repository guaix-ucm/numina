#
# Copyright 2019-2023 Universidad Complutense de Madrid
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

    parser_verify = subparsers.add_parser(
        'verify',
        help='verify a observation result'
    )

    parser_verify.set_defaults(command=verify)

    parser_verify.add_argument(
        '-c', '--task-control', dest='reqs',
        help='configuration file of the processing task', metavar='FILE'
    )
    parser_verify.add_argument(
        '-r', '--requirements', dest='reqs',
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
        '--obs', action="store_true",
        help='validate files in OBs'
    )
    parser_verify.add_argument(
        'files', nargs='+',
        help='file with the observation result'
    )

    return parser_verify


def verify(args, extra_args):
    if args.obs:
        # verify as oblocks
        # Loading observation result if exists
        sessions, loaded_obs = load_observations(args.files, args.session)
        datamanager = create_datamanager(None, args.basedir, args.datadir)
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
                validate_inputs=True, validate_results=True
            )
    else:
        # This function loads the recipes
        datamanager = create_datamanager(None, args.basedir, args.datadir)
        for file in args.files:
            _logger.info(f'checking {file}')
            try:
                result = check_file(file)
            except Exception as error:
                result = False
                # _logger.warning('with error {}'.format(error))
            _logger.info(f'checked {file}, valid={result}')
            # print('done')

    return 0


def run_verify(datastore, obsid, as_mode=None, requirements=None, copy_files=False,
               validate_inputs=False, validate_results=False):
    """Verify raw images"""

    configuration = 'default'
    _logger.info(f"verify OB with id={obsid}")

    # Roll back to cwd after leaving the context
    with working_directory(datastore.datadir):

        obsres = datastore.backend.obsres_from_oblock_id(
            obsid, as_mode=as_mode, configuration=configuration
        )

        thisdrp = datastore.backend.drps.query_by_name(obsres.instrument)

        msg = f'the mode of this obsres is {obsres.instrument}.{obsres.mode}'
        _logger.info(msg)

        for v in thisdrp.modes.values():
            if v.key == obsres.mode:
                mode_obj = v
                break
        else:
            raise ValueError(f'unrecognized mode {obsres.mode}')

        image_is = mode_obj.rawimage

        for f in obsres.frames:
            with f.open() as hdulist:
                _logger.debug(f'checking {f.filename}')
                try:
                    check_image(hdulist, astype=image_is)
                except:
                    pass
        _logger.info('Checking that individual images are valid for this mode')
        mode_obj.validate(obsres)
        #


def check_file(filename, astype=None, level=None):
    import json
    import yaml
    import astropy.io.fits as fits

    json_ext = ['.json']
    fits_ext = ['.fits', '.fits.gz', '.fit']
    yaml_ext = ['.yaml', '.yml']

    fname, ext = os.path.splitext(filename)

    if ext in json_ext:
        # print('as json')
        with open(filename) as fd:
            obj = json.load(fd)
        return check_json(obj, astype=astype, level=level)
    elif ext in fits_ext:
        # print('as fits')
        with fits.open(filename) as img:
            return check_image(img, astype=None)
    elif ext in yaml_ext:
        with open(filename) as fd:
            obj = list(yaml.safe_load_all(fd))
        return check_yaml(obj, astype=astype, level=level)
    else:
        print(f'ignoring {filename}')
        return True


def check_json(obj, astype=None, level=None):
    import numina.core.config as cfg

    if 'instrument' in obj:
        instrument = obj['instrument']
        return cfg.check(instrument, obj)
    else:
        print("no 'instrument' field in object")
        # Try to check with other schema
        return False


def check_yaml(obj, astype=None, level=None):
    # import numina.core.config as cfg

    # import jsonschema
    # import json
    # try to verify as obsblock
    # path = "/home/spr/devel/guaix/numina/schemas/oblock-schema.json"
    # with open(path) as fd:
    #     schema = json.load(fd)
    #     jsonschema.validate(obj, schema=schema)
    pass


def check_image(hdulist, astype=None, level=None):

    import numina.core.config as cfg
    # Determine the instrument name
    hdr = hdulist[0].header
    instrument = hdr['INSTRUME']
    return cfg.check(instrument, hdulist, astype=astype, level=level)


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
