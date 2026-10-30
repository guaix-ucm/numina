#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import warnings
from astropy.io import fits

import numina.array


def get_hdu_shape(header):
    """Return the shape given by the NAXISn keywords of a header, in FITS order ``(NAXIS1, NAXIS2, ...)``."""
    ndim = header["naxis"]
    return tuple(header.get(f"NAXIS{i:d}") for i in range(1, ndim + 1))


def custom_slice_to_str(slc):
    """Return a string ``start:stop`` or ``start:stop:step`` representing a slice."""
    if slc.step is None:
        return f"{slc.start:d}:{slc.stop:d}"
    else:
        return f"{slc.start:d}:{slc.stop:d}:{slc.step:d}"


def custom_region_to_str(region):
    """Return a string ``[start:stop,start:stop,...]`` representing a tuple of slices."""
    jints = [custom_slice_to_str(slc) for slc in region]
    return "[" + ",".join(jints) + "]"


def resize_hdu(hdu, newshape, region, window=None, fill=0.0, scale=1, conserve=True, dtype=None):
    """Return a new HDU, of the same class as ``hdu``, with its data placed in a region of a larger array.

    The data is resized with ``numina.array.resize_array``. The header is a copy of
    the original one, with the keyword NVALREGI set to ``region`` and, if ``window``
    is given, the keyword OVALREGI set to ``window``.
    """
    # from numina.array import resize_array

    basedata = hdu.data
    newdata = numina.array.resize_array(
        basedata, newshape, region, window=window, fill=fill, scale=scale, conserve=conserve, dtype=dtype
    )
    header = hdu.header.copy()
    header["NVALREGI"] = (custom_region_to_str(region), "Valid region of resized FITS")
    if window:
        header["OVALREGI"] = (custom_region_to_str(window), "Valid region of original FITS")
    # The same class as the input, a PrimaryHDU or an ImageHDU
    newhdu = type(hdu)(newdata, header)
    return newhdu


def resize_hdul(hdul, newshape, region, extensions=None, window=None, scale=1, fill=0.0, conserve=True):
    """Return a new HDUList with the HDUs in ``extensions`` resized with ``resize_hdu``.

    ``extensions`` is a list of HDU indices, only the primary HDU by default.
    The other HDUs are copied unchanged into the new HDUList.
    """
    from numina.frame import resize_hdu

    if extensions is None:
        extensions = [0]

    nhdul = [None] * len(hdul)
    for ext, hdu in enumerate(hdul):
        if ext in extensions:
            nhdul[ext] = resize_hdu(hdu, newshape, region, fill=fill, window=window, scale=scale, conserve=conserve)
        else:
            nhdul[ext] = hdu
    return fits.HDUList(nhdul)


def resize_fits(
    fitsfile, newfilename, newshape, region, window=None, scale=1, fill=0.0, overwrite=True, conserve=True, dtype=None
):
    """Resize the primary HDU of a FITS file with ``resize_hdu`` and write it to a new file.

    ``fitsfile`` is a file name or an HDUList. The new file contains only the resized
    HDU, written as the primary HDU.
    """

    close_on_exit = False
    if isinstance(fitsfile, str):
        hdulist = fits.open(fitsfile, mode="readonly")
        close_on_exit = True
    else:
        hdulist = fitsfile

    try:
        hdu = hdulist["primary"]
        newhdu = resize_hdu(
            hdu, newshape, region, fill=fill, window=window, scale=scale, conserve=conserve, dtype=dtype
        )
        # The file has only this HDU, the primary, also when the first
        # HDU of fitsfile is an extension, as in HDUList([hdul["BPM"]])
        if not isinstance(newhdu, fits.PrimaryHDU):
            newhdu = fits.PrimaryHDU(newhdu.data, newhdu.header)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            newhdu.writeto(newfilename, overwrite=overwrite)
    finally:
        if close_on_exit:
            hdulist.close()
