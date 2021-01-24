#
# Copyright 2019-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Load files stored in disk based on extension"""

import os
import logging

from numina.util.objimport import import_object


_logger = logging.getLogger(__name__)


# FIXME: this is already implemented, elsewhere
# There should be one-- and preferably only one --obvious way to do it.

def is_fits(filename, **kwargs):
    return filename.endswith('.fits')


def read_fits(filename):
    import numina.types.dataframe as df
    import astropy.io.fits as fits
    return df.DataFrame(frame=fits.open(filename))


def read_fits_later(filename):
    import numina.types.dataframe as df
    return df.DataFrame(filename=os.path.abspath(filename))


def is_json(filename, **kwargs):
    return filename.endswith('.json')


def read_json(filename):
    import json

    with open(filename) as fd:
        base = json.load(fd)
    return read_structured(base)


def is_yaml(filename, **kwargs):
    return filename.endswith('.yaml')


def read_yaml(filename):
    import yaml

    with open(filename) as fd:
        base = yaml.load(fd)
    return read_structured(base)


def read_structured(data):
    if 'type_fqn' in data:
        type_fqn = data['type_fqn']
        cls = import_object(type_fqn)
        obj = cls.__new__(cls)
        obj.__setstate__(data)
        return obj
    return data


def unserial(value):
    checkers = [(is_fits, read_fits_later), (is_json, read_json), (is_yaml, read_yaml)]
    if isinstance(value, str):
        for check_type, conv in checkers:
            if check_type(value):
                return conv(value)
        else:
            return value
    else:
        return value
