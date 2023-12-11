#
# Copyright 2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""User command line interface of Numina, verify functionality."""

import logging


_logger = logging.getLogger(__name__)


def register(subparsers, config):
    parser_identify = subparsers.add_parser(
        'identify',
        help='identify'
    )
    parser_identify.set_defaults(command=identify)
    parser_identify.add_argument(
        'files', nargs='+',
        help='identify files in CL'
    )

    return parser_identify


def identify(args, extra_args):

    # This function loads the recipes
    import numina.drps
    import numina.instrument.assembly as asbl
    import astropy.io.fits as fits

    sys_drps = numina.drps.get_system_drps()
    com_store = asbl.load_panoply_store(sys_drps, None)

    for f in args.files:
        with fits.open(f) as hdulist:
            _logger.debug(f'identify {f}')
            try:
                # Determine the instrument name
                hdr = hdulist[0].header
                instrument = hdr['INSTRUME']
                #
                this_drp = sys_drps.query_by_name(instrument)
                _logger.debug('assembly instrument model')
                key, date_obs, keyname = this_drp.select_profile_image(hdulist)
                config = asbl.assembly_instrument(com_store, key, date_obs, by_key=keyname)
                print('Instrument:', instrument)
                print('Instrument profile', config.uuid)

            except Exception as ex:
                print(ex)
    return 0
