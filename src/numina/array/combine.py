#
# Copyright 2008-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Different methods for combining lists of arrays."""

import numpy

import numina.array._combine as intl  # noqa
from numina.array.crmasks.apply_crmasks import apply_crmasks


def mean(arrays, masks=None, dtype=None, out=None, out_res=None,
         out_var=None, out_pix=None, zeros=None, scales=None, weights=None):
    """Combine arrays using the mean, with masks and offsets.

    Arrays and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays
    :return: mean, variance of the mean and number of points stored

    Example:
        >>> import numpy
        >>> image = numpy.array([[1., 3.], [1., -1.4]])
        >>> inputs = [image, image + 1]
        >>> mean(inputs)
        array([[[ 1.5,  3.5],
                [ 1.5, -0.9]],
        <BLANKLINE>
               [[ 0.5,  0.5],
                [ 0.5,  0.5]],
        <BLANKLINE>
               [[ 2. ,  2. ],
                [ 2. ,  2. ]]])

    """
    return generic_combine(intl.mean_method(), arrays, masks=masks, dtype=dtype, out=out, out_res=out_res,
                           out_var=out_var, out_pix=out_pix, zeros=zeros, scales=scales, weights=weights)


def median(arrays, masks=None, dtype=None, out=None, out_res=None,
           out_var=None, out_pix=None, zeros=None, scales=None, weights=None):
    """Combine arrays using the median, with masks.

    Arrays and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays

    :return: median, variance of the median and number of points stored
    """

    return generic_combine(intl.median_method(), arrays, masks=masks, dtype=dtype, out=out,
                           out_res=out_res, out_var=out_var, out_pix=out_pix,
                           zeros=zeros, scales=scales, weights=weights)


def sigmaclip(arrays, masks=None, dtype=None, out=None, out_res=None, out_var=None,
              out_pix=None, zeros=None, scales=None, weights=None, low=3.0, high=3.0):
    """Combine arrays using the sigma-clipping, with masks.

    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays
    :param low:
    :param high:
    :return: mean, variance of the mean and number of points stored
    """

    return generic_combine(intl.sigmaclip_method(low, high), arrays, out=out, dtype=dtype, out_res=out_res,
                           out_var=out_var, out_pix=out_pix, masks=masks, zeros=zeros, scales=scales, weights=weights)


def minmax(arrays, masks=None, dtype=None, zeros=None, scales=None, weights=None, nmin=1, nmax=1):
    """Combine arrays using mix max rejection, with masks.

    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays
    :param nmin:
    :param nmax:
    :return: mean, variance of the mean and number of points stored
    """

    return generic_combine(intl.minmax_method(nmin, nmax), arrays, dtype=dtype,
                           zeros=zeros, scales=scales, weights=weights)


def quantileclip(arrays, masks=None, out_res=None, out_var=None, out_pix=None, out=None,
                 dtype=None, zeros=None, scales=None, weights=None, fclip=0.10):
    """Combine arrays using the sigma-clipping, with masks.

    Inputs and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the mean,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param out: optional output, with one more axis than the input arrays
    :param fclip: fraction of points removed on both ends. Maximum is 0.4 (80% of points rejected)
    :return: mean, variance of the mean and number of points stored
    """
    return generic_combine(intl.quantileclip_method(fclip), arrays, masks=masks, out=out, dtype=dtype,
                           out_res=out_res, out_var=out_var, out_pix=out_pix,
                           zeros=zeros, scales=scales, weights=weights)


def flatcombine(arrays, masks=None, scales=None, dtype=None, low=3.0, high=3.0, blank=1.0):
    """Combine flat arrays.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays
    :param blank: non-positive values are substituted by this on output
    :return: mean, variance of the mean and number of points stored
    """

    result = sigmaclip(arrays, masks=masks, dtype=dtype, scales=scales, low=low, high=high)

    # Substitute values <= 0 by blank
    mm = result[0] <= 0
    result[0, mm] = blank
    # Add values to mask
    result[1:2, mm] = 0

    return result


def zerocombine(arrays, masks, scales=None, dtype=None):
    """Combine zero arrays.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param scales:
    :return: median, variance of the median and number of points stored
    """

    result = median(arrays, masks=masks, scales=scales, dtype=dtype)

    return result


def sum(arrays, masks=None, out_res=None, out_var=None, out_pix=None, out=None, dtype=None, zeros=None, scales=None):
    """Combine arrays by addition, with masks and offsets.

    Arrays and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the sum,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param out: optional output, with one more axis than the input arrays
    :return: sum, variance of the sum and number of points stored

    Example:
        >>> import numpy
        >>> image = numpy.array([[1., 3.], [1., -1.4]])
        >>> inputs = [image, image + 1]
        >>> sum(inputs)
        array([[[ 1.5,  3.5],
                [ 1.5, -0.9]],
        <BLANKLINE>
               [[ 0.5,  0.5],
                [ 0.5,  0.5]],
        <BLANKLINE>
               [[ 2. ,  2. ],
                [ 2. ,  2. ]]])

    """
    return generic_combine(intl.sum_method(), arrays, out=out, dtype=dtype, out_res=out_res,
                           out_var=out_var, out_pix=out_pix, masks=masks, zeros=zeros, scales=scales)


def generic_combine(method, arrays, out_res=None, out_var=None, out_pix=None, out=None,
                    dtype=None, masks=None, zeros=None, scales=None, weights=None):

    # Trying to use all the arguments
    if out is not None:
        out_res = out[0]
        out_var = out[1]
        out_pix = out[2]
    else:
        if dtype is not None and len(arrays) > 0:
            out_res = numpy.empty_like(arrays[0], dtype=dtype)
            out_var = numpy.empty_like(arrays[0], dtype=dtype)
            out_pix = numpy.empty_like(arrays[0], dtype=dtype)

    return intl.generic_combine(method, arrays, out_res=out_res, out_var=out_var,
                                out_pix=out_pix, masks=masks, zeros=zeros, scales=scales)


def mediancr(arrays, crmasks, dtype, use_auxmedian=False, apply_flux_factor=None, bias=None):
    """Combine arrays using the median with cosmic ray mask.

    The function returns the median of the input arrays, applying
    a cosmic ray mask computed to detect cosmis rays hitting more that
    once in a single pixel. The masked pixels are replaced by the minimum
    value of the same pixel in the input arrays.
    """
    return apply_crmasks(
        list_arrays=arrays,
        hdul_masks=crmasks,
        use_auxmedian=use_auxmedian,
        combination='mediancr',
        dtype=dtype,
        apply_flux_factor=apply_flux_factor,
        bias=bias
    )


def meancr(arrays, crmasks, dtype, use_auxmedian=False, apply_flux_factor=None, bias=None):
    """Combine arrays using the mean with individual cosmic ray masks.

    The function returns the mean of the input arrays, applying
    a cosmic ray mask computed for each input array. This allows to employ
    numpy masked arrays to determine the mean of each pixel. For those
    pixels where the cosmic ray mask is True in all the input arrays,
    the minimum value of the same pixel in the input arrays is used
    as the value of the pixel in the output.
    """
    return apply_crmasks(
        list_arrays=arrays,
        hdul_masks=crmasks,
        use_auxmedian=use_auxmedian,
        combination='meancr',
        dtype=dtype,
        apply_flux_factor=apply_flux_factor,
        bias=bias
    )


def meancrt(arrays, crmasks, dtype, use_auxmedian=False, apply_flux_factor=None, bias=None):
    """Combine arrays using the mean with a single cosmic ray mask.

    The function returns the mean of the input arrays, applying
    a cosmic ray mask computed at once, by stacking the input arrays
    and trying to detect all the cosmic rays in all the exposures piled
    up in a single image. The masked pixels are replaced by the value
    of the same pixel when the combination is computed using 'mediancr'.
    """
    return apply_crmasks(
        list_arrays=arrays,
        hdul_masks=crmasks,
        use_auxmedian=use_auxmedian,
        combination='meancrt',
        dtype=dtype,
        apply_flux_factor=apply_flux_factor,
        bias=bias
    )
