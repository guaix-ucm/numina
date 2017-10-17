#
# Copyright 2008-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import print_function

import numina.util.convert as conv


class KeyDefinition(object):
    def __init__(self, key, ext=None, default=None, convert=None):
        self.key = key
        self.ext = 0 if ext is None else ext
        self.default = default
        self.convert = convert

    def __call__(self, hdulist):
        value = hdulist[self.ext].header.get(self.key, self.default)
        if self.convert:
            return self.convert(value)
        return value


class FITSKeyExtractor(object):
    def __init__(self, values):
        self.map = {}
        for key, entry in values.items():
            if isinstance(entry, KeyDefinition):
                newval = entry
            elif isinstance(entry, tuple):
                if len(entry) == 3:
                    keyname = entry[0]
                    hduname = entry[1]
                    convert = entry[2]
                    default = None
                elif len(entry) == 2:
                    keyname = entry[0]
                    default = entry[1]
                    hduname = 0
                    convert = None
                else:
                    raise ValueError

                newval= KeyDefinition(
                    keyname,
                    ext=hduname,
                    convert=convert,
                    default=default
                )
            elif isinstance(entry, str):
                newval = KeyDefinition(
                    entry
                )
            else:
                newval = entry

            self.map[key] = newval

    def extract(self, value, hdulist):
        extractor = self.map[value]
        return extractor(hdulist)


class DataModel(object):
    """Model of the Data being processed"""

    db_info_keys = [
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
        'quality_control',
    ]

    def __init__(self, name='UNKNOWN', extractor=None):

        self.name = name

        default_values = {
            'instrument': 'INSTRUME',
            'object': 'OBJECT',
            'observation_date': ('DATE-OBS', 0, conv.convert_date),
            'uuid': 'uuid',
            'type': 'numtype',
            'mode': 'obsmode',
            'exptime': 'exptime',
            'darktime': 'darktime',
            'quality_control': ('NUMRQC', 0, conv.convert_qc),
            'insmode': ('INSMODE', 'undefined'),
        }

        values = {} if extractor is None else extractor
        values.update(default_values)
        self.extractor = FITSKeyExtractor(values)

    def get_data(self, img):
        return img['primary'].data

    def get_header(self, img):
        return img['primary'].header

    def get_variance(self, img):
        return img['variance'].data

    def get_imgid(self, img):
        imgid = img.filename()

        # More heuristics here...
        # get FILENAME keyword, for example...

        if not imgid:
            imgid = repr(img)

        return imgid

    def get_darktime(self, img):
        return self.get_exptime(img)

    def get_exptime(self, img):
        header = self.get_header(img)
        if 'EXPTIME' in header.keys():
            etime = header['EXPTIME']

        elif 'EXPOSED' in header.keys():
            etime = header['EXPOSED']
        else:
            etime = 1.0
        return etime

    def do_sky_correction(self, img):
        return True

    def gather_info(self, img):
        with img.open() as hdulist:
            info = self.gather_info_hdu(hdulist)
        return info

    def gather_info_hdu(self, hdulist):
        return {}

    def get_quality_control(self, img):
        return self.extractor.extract('quality_control', img)
