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

import warnings

from astropy.io import fits

from numina.array import resize_array


def get_hdu_shape(header):
    ndim = header['naxis']
    return tuple(header.get('NAXIS%d' % i) for i in range(1, ndim + 1))


def custom_slice_to_str(slc):
    if slc.step is None:
        return '%i:%i' % (slc.start, slc.stop)
    else:
        return '%i:%i:%i' % (slc.start, slc.stop, slc.step)


def custom_region_to_str(region):
    jints = [custom_slice_to_str(slc) for slc in region]
    return '[' + ','.join(jints) + ']'


def resize_hdu(hdu, newshape, region, window=None, fill=0.0,
               scale=1, conserve=True):
    basedata = hdu.data
    newdata = resize_array(basedata, newshape, region, window=window,
                           fill=fill, scale=scale, conserve=conserve)
    hdu.header.update('NVALREGI', custom_region_to_str(region),
                      'Valid region of resized FITS')
    if window:
        hdu.header.update('OVALREGI', custom_region_to_str(window),
                          'Valid region of original FITS')
    newhdu = fits.PrimaryHDU(newdata, hdu.header)
    return newhdu


def resize_fits(fitsfile, newfilename, newshape, region, window=None,
                scale=1, fill=0.0, clobber=True, conserve=True):

    close_on_exit = False
    if isinstance(fitsfile, basestring):
        hdulist = fits.open(fitsfile, mode='readonly')
        close_on_exit = True
    else:
        hdulist = fitsfile

    try:
        hdu = hdulist['primary']
        newhdu = resize_hdu(hdu, newshape, region, fill=fill,
                            window=window, scale=scale, conserve=conserve)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            newhdu.writeto(newfilename, clobber=clobber)
    finally:
        if close_on_exit:
            hdulist.close()
