#
# Copyright 2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Registries of functions that load, describe and check files

``load`` and ``describe`` are :class:`~numina.core.dataload.DataLoaders`:
called with the path of a file, they return its contents and a
description (instrument and observing mode). The DRPs can register their
own functions, with a lower priority value to be used before these ones.
``check`` is a :class:`~numina.core.dataload.DataChecker` that checks an
object with the function of its instrument, the attribute ``checker`` of
its DRP.
"""

import numina.core.dataload

#: Functions that load a file, by media type
load = numina.core.dataload.DataLoaders()


@load.register("image/fits", priority=20)
def load_fits_0(pathname):
    """Open a FITS file"""
    import astropy.io.fits as fits

    return fits.open(pathname)


@load.register("application/json", priority=20)
def load_json(pathname):
    """Load a JSON file"""
    import json

    with open(pathname) as fd:
        return json.load(fd)


@load.register("application/json", numina.core.dataload.is_json_structured, priority=5)
def load_json(pathname):  # noqa: F811
    """Load a JSON file with a serialized object, of the class in 'type_fqn'"""
    import json
    from numina.util.objimport import import_object

    with open(pathname) as fd:
        data = json.load(fd)
    type_fqn = data["type_fqn"]
    cls = import_object(type_fqn)
    obj = cls.__new__(cls)
    obj.__setstate__(data)
    return obj


#: Functions that describe a file, by media type
describe = numina.core.dataload.DataLoaders()


#: Check an object with the checker of the DRP of its instrument
check = numina.core.dataload.DataChecker()


@describe.register("image/fits", priority=20)
def describe_fits_0(pathname):
    """Return the instrument and the observing mode of a FITS file, from INSTRUME and OBSMODE"""
    import astropy.io.fits as fits

    with fits.open(pathname) as hdulist:
        prim = hdulist[0].header
        instrument = prim.get("INSTRUME", "unknown")
        obsmode = prim.get("OBSMODE", "unknown")

        return instrument, obsmode


@describe.register("application/json", numina.core.dataload.is_json_structured, priority=20)
def describe_json(pathname):
    """Return the instrument of a serialized object; the observing mode is not known"""
    import json
    from numina.util.objimport import import_object

    with open(pathname) as fd:
        data = json.load(fd)

    type_fqn = data["type_fqn"]
    cls = import_object(type_fqn)
    obj = cls.__new__(cls)
    obj.__setstate__(data)
    return obj.instrument, "TBD"
