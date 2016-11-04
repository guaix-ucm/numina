#
# Copyright 2008-2014 Universidad Complutense de Madrid
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



class DataModel(object):
    """Model of the Data being processed"""

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