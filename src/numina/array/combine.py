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
from numina.array.mediancr import _mediancr, _mediancrmask, _meancrmask


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


def mediancr(arrays,
             gain=None, rnoise=None, bias=0.0,
             flux_variation_min=1.0, flux_variation_max=1.0,
             ntest=100, knots_splfit=2, nsimulations=10000,
             times_boundary_extension=1.0, threshold=None,
             minimum_max2d_rnoise=5.0, interactive=False, dilation=1,
             compute_meancr=False,
             dtype=numpy.float32, plots=False, semiwindow=15, color_scale='minmax',
             maxplots=10):
    """Combine arrays using the median but correcting for double cosmic rays."""
    median2d_corrected, variance2d, map2d, mean2d = _mediancr(
        arrays,
        gain=gain, rnoise=rnoise, bias=bias,
        flux_variation_min=flux_variation_min, flux_variation_max=flux_variation_max,
        ntest=ntest, knots_splfit=knots_splfit, nsimulations=nsimulations,
        times_boundary_extension=times_boundary_extension,
        threshold=threshold, minimum_max2d_rnoise=minimum_max2d_rnoise,
        interactive=interactive, dilation=dilation,
        compute_meancr=compute_meancr,
        dtype=dtype, plots=plots, semiwindow=semiwindow, color_scale=color_scale,
        maxplots=maxplots
    )

    return median2d_corrected, variance2d, map2d


def meancr(arrays,
           gain=None, rnoise=None, bias=0.0,
           flux_variation_min=1.0, flux_variation_max=1.0,
           ntest=100, knots_splfit=2, nsimulations=10000,
           times_boundary_extension=1.0, threshold=None,
           minimum_max2d_rnoise=5.0, interactive=False, dilation=1,
           compute_meancr=True,
           dtype=numpy.float32, plots=False, semiwindow=15, color_scale='minmax',
           maxplots=10):
    """Combine arrays using the mean but correcting for double cosmic rays."""
    median2d_corrected, variance2d, map2d, mean2d_corrected = _mediancr(
        arrays,
        gain=gain, rnoise=rnoise, bias=bias,
        flux_variation_min=flux_variation_min, flux_variation_max=flux_variation_max,
        ntest=ntest, knots_splfit=knots_splfit, nsimulations=nsimulations,
        times_boundary_extension=times_boundary_extension,
        threshold=threshold, minimum_max2d_rnoise=minimum_max2d_rnoise,
        interactive=interactive, dilation=dilation,
        compute_meancr=compute_meancr,
        dtype=dtype, plots=plots, semiwindow=semiwindow, color_scale=color_scale,
        maxplots=maxplots
    )

    return mean2d_corrected, variance2d, map2d


def mediancrmask(arrays, mask_mediancr_file=None, dtype=numpy.float32):
    """Combine arrays using the median, replacing masked pixels by the minimum.

    This function makes use of a mask array previously computed using
    the mediancr function.
    """
    if mask_mediancr_file is None:
        raise ValueError("A mask array file must be provided for mediancrmask combination.")

    median2d_corrected, variance2d, map2d = _mediancrmask(
        arrays, mask_mediancr_file=mask_mediancr_file, dtype=dtype)

    return median2d_corrected, variance2d, map2d


def meancrmask(arrays, mask_mediancr_file=None, mask_meancr_file=None, dtype=numpy.float32):
    """Combine arrays using the mean, replacing masked pixels by the mediancr values.

    This function makes use of two mask arrays previously computed using
    the meancr function.
    """
    if mask_mediancr_file is None:
        raise ValueError("A mask_mediancr_file must be provided for meancrmask combination.")
    if mask_meancr_file is None:
        raise ValueError("A mask_meancr_file must be provided for meancrmask combination.")

    mean2d_corrected, variance2d, map2d = _meancrmask(
        arrays, mask_mediancr_file=mask_mediancr_file, mask_meancr_file=mask_meancr_file, dtype=dtype)

    return mean2d_corrected, variance2d, map2d
