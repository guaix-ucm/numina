#
# Copyright 2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Build a LoadableDRP from a yaml file"""

import numina.core.dataload


load = numina.core.dataload.DataLoaders()


@load.register('image/fits', priority=20)
def load_fits_0(pathname):
    import astropy.io.fits as fits
    return fits.open(pathname)


@load.register('application/json', priority=20)
def load_json(pathname):
    import json
    with open(pathname) as fd:
        return json.load(fd)


@load.register('application/json', numina.core.dataload.is_json_structured, priority=5)
def load_json(pathname):
    import json
    from numina.util.objimport import import_object

    with open(pathname) as fd:
        data = json.load(fd)
    type_fqn = data['type_fqn']
    cls = import_object(type_fqn)
    obj = cls.__new__(cls)
    obj.__setstate__(data)
    return obj


describe = numina.core.dataload.DataLoaders()


check = numina.core.dataload.DataChecker()


# Here we could have methods to extract
# this information from different files
# FITS and JSON
_describe_keys = [
    'instrument',
    'object',
    'observation_date',
    'uuid',
    'type',
    'mode',
    'exptime',
    'darktime',
    'insconf',
    'blckuuid',
    'quality_control'
]

@describe.register('image/fits', priority=20)
def describe_fits_0(pathname):
    import astropy.io.fits as fits
    with fits.open(pathname) as hdulist:
        prim = hdulist[0].header
        instrument = prim.get("INSTRUME", "unknown")
        obsmode = prim.get("OBSMODE", "unknown")

        return (instrument, obsmode)


@describe.register('application/json',
                   numina.core.dataload.is_json_structured,
                   priority=20)
def describe_json(pathname):
    import json
    from numina.util.objimport import import_object

    with open(pathname) as fd:
        data = json.load(fd)

    type_fqn = data['type_fqn']
    cls = import_object(type_fqn)
    obj = cls.__new__(cls)
    obj.__setstate__(data)
    return (obj.instrument, "TBD")