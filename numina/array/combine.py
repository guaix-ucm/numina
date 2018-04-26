#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Different methods for combining lists of arrays."""

import numpy

import numina.array._combine as intl_combine


CombineError = intl_combine.CombineError
mean_method = intl_combine.mean_method


def mean(arrays, masks=None, dtype=None, out=None,
         zeros=None, scales=None,
         weights=None):
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
    return generic_combine(intl_combine.mean_method(), arrays, masks=masks,
                           dtype=dtype, out=out,
                           zeros=zeros, scales=scales,
                           weights=weights)


def median(arrays, masks=None, dtype=None, out=None,
           zeros=None, scales=None,
           weights=None):
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

    return generic_combine(intl_combine.median_method(), arrays, masks=masks,
                           dtype=dtype, out=out,
                           zeros=zeros, scales=scales, weights=weights)


def sigmaclip(arrays, masks=None, dtype=None, out=None,
              zeros=None, scales=None, weights=None,
              low=3., high=3.):
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

    return generic_combine(intl_combine.sigmaclip_method(low, high), arrays,
                           masks=masks, dtype=dtype, out=out,
                           zeros=zeros, scales=scales, weights=weights)


def minmax(arrays, masks=None, dtype=None, out=None, zeros=None,
           scales=None, weights=None, nmin=1, nmax=1):
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

    return generic_combine(intl_combine.minmax_method(nmin, nmax), arrays,
                           masks=masks, dtype=dtype, out=out,
                           zeros=zeros, scales=scales, weights=weights)


def quantileclip(arrays, masks=None, dtype=None, out=None,
                 zeros=None, scales=None, weights=None,
                 fclip=0.10):
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
    :param fclip: fraction of points removed on both ends. Maximum is 0.4 (80% of points rejected)
    :return: mean, variance of the mean and number of points stored
    """
    return generic_combine(intl_combine.quantileclip_method(fclip), arrays,
                           masks=masks, dtype=dtype, out=out,
                           zeros=zeros, scales=scales, weights=weights)


def flatcombine(arrays, masks=None, dtype=None, scales=None,
                low=3.0, high=3.0, blank=1.0):
    """Combine flat arrays.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param out: optional output, with one more axis than the input arrays
    :param blank: non-positive values are substituted by this on output
    :return: mean, variance of the mean and number of points stored
    """

    result = sigmaclip(arrays, masks=masks,
                       dtype=dtype, scales=scales,
                       low=low, high=high)

    # Substitute values <= 0 by blank
    mm = result[0] <= 0
    result[0, mm] = blank
    # Add values to mask
    result[1:2, mm] = 0

    return result


def zerocombine(arrays, masks, dtype=None, scales=None):
    """Combine zero arrays.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param scales:
    :return: median, variance of the median and number of points stored
    """

    result = median(arrays, masks=masks,
                    dtype=dtype, scales=scales)

    return result


def sum(arrays, masks=None, dtype=None, out=None,
         zeros=None, scales=None):
    """Combine arrays by addition, with masks and offsets.

    Arrays and masks are a list of array objects. All input arrays
    have the same shape. If present, the masks have the same shape
    also.

    The function returns an array with one more dimension than the
    inputs and with size (3, shape). out[0] contains the sum,
    out[1] the variance and out[2] the number of points used.

    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
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
    return generic_combine(intl_combine.sum_method(), arrays, masks=masks,
                           dtype=dtype, out=out,
                           zeros=zeros, scales=scales)


def generic_combine(method, arrays, masks=None, dtype=None,
                    out=None, zeros=None, scales=None, weights=None):
    """Stack arrays using different methods.

    :param method: the combination method
    :type method: PyCObject
    :param arrays: a list of arrays
    :param masks: a list of mask arrays, True values are masked
    :param dtype: data type of the output
    :param zeros:
    :param scales:
    :param weights:
    :return: median, variance of the median and number of points stored
    """

    arrays = [numpy.asarray(arr, dtype=dtype) for arr in arrays]
    if masks is not None:
        masks = [numpy.asarray(msk) for msk in masks]

    if out is None:
        # Creating out if needed
        # We need three numbers
        try:
            outshape = (3,) + tuple(arrays[0].shape)
            out = numpy.zeros(outshape, dtype)
        except AttributeError:
            raise TypeError('First element in arrays does '
                            'not have .shape attribute')
    else:
        out = numpy.asanyarray(out)

    intl_combine.generic_combine(
        method, arrays,
        out[0], out[1], out[2],
        masks, zeros, scales, weights
    )
    return out
