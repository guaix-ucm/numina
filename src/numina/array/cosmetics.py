#
# Copyright 2008-2022 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import logging
import itertools

import numpy
import scipy.stats
import scipy.ndimage
from numina.array.blocks import max_blk_coverage, blk_nd_short


# Values stored in integer masks
PIXEL_HOT = 1
PIXEL_DEAD = 1
PIXEL_VALID = 0

#
HIGH_SIGMA = 200
LOW_SIGMA = -200

_logger = logging.getLogger(__name__)


def update_mask(mask, gmask, newmask, value):
    f1_mask = mask[gmask]
    f1_mask[newmask] = value
    mask[gmask] = f1_mask
    gmask = mask == PIXEL_VALID
    smask = mask != PIXEL_VALID
    return mask, gmask, smask


# IRAF task
def ccdmask(flat1, flat2=None, mask=None, lowercut=6.0, uppercut=6.0,
            siglev=1.0, mode='region', nmed=(7, 7), nsig=(15, 15)):
    """Find cosmetic defects in a detector using two flat field images.

    Two arrays representing flat fields of different exposure times are
    required. Cosmetic defects are selected as points that deviate
    significantly of the expected normal distribution of pixels in
    the ratio between `flat2` and `flat1`. The median of the ratio
    is computed and subtracted. Then, the standard deviation is estimated
    computing the percentiles
    nearest to the pixel values corresponding to`siglev` in the normal CDF.
    The standard deviation is then the distance between the pixel values
    divided by two times `siglev`. The ratio image is then normalized with
    this standard deviation.

    The behavior of the function depends on the value of the parameter
    `mode`. If the value is 'region' (the default), both the median
    and the sigma are computed in boxes. If the value is 'full', these
    values are computed using the full array.

    The size of the boxes in 'region' mode is given by `nmed` for
    the median computation and `nsig` for the standard deviation.

    The values in the normalized ratio array above `uppercut`
    are flagged as hot pixels, and those below '-lowercut` are
    flagged as dead pixels in the output mask.

    :parameter flat1: an array representing a flat illuminated exposure.
    :parameter flat2: an array representing a flat illuminated exposure.
    :parameter mask: an integer array representing initial mask.
    :parameter lowercut: values below this sigma level are flagged as dead pixels.
    :parameter uppercut: values above this sigma level are flagged as hot pixels.
    :parameter siglev: level to estimate the standard deviation.
    :parameter mode: either 'full' or 'region'
    :parameter nmed: region used to compute the median
    :parameter nsig: region used to estimate the standard deviation
    :returns: the normalized ratio of the flats, the updated mask and standard deviation

    .. note::

        This function is based on the description of the task
        ccdmask of IRAF

    .. seealso::

        :py:func:`cosmetics`
            Operates much like this function but computes
            median and sigma in the whole image instead of in boxes

    """

    if flat2 is None:
        # we have to swap flat1 and flat2, and
        # make flat1 an array of 1s
        flat1, flat2 = flat2, flat1
        flat1 = numpy.ones_like(flat2)

    if mask is None:
        mask = numpy.zeros_like(flat1, dtype='int')

    ratio = numpy.zeros_like(flat1)
    invalid = numpy.zeros_like(flat1)
    invalid[mask == PIXEL_HOT] = HIGH_SIGMA
    invalid[mask == PIXEL_DEAD] = LOW_SIGMA

    gmask = mask == PIXEL_VALID
    _logger.info('valid points in input mask %d', numpy.count_nonzero(gmask))
    smask = mask != PIXEL_VALID
    _logger.info('invalid points in input mask %d', numpy.count_nonzero(smask))

    # check if there are zeros in flat1 and flat2
    zero_mask = numpy.logical_or(flat1[gmask] <= 0, flat2[gmask] <= 0)

    # if there is something in zero mask
    # we update the mask
    if numpy.any(zero_mask):
        mask, gmask, smask = update_mask(mask, gmask, zero_mask, PIXEL_DEAD)
        invalid[mask == PIXEL_DEAD] = LOW_SIGMA

    # ratio of flats
    ratio[gmask] = flat2[gmask] / flat1[gmask]
    ratio[smask] = invalid[smask]

    if mode == 'region':
        _logger.info('computing median in boxes of %r', nmed)
        ratio_med = scipy.ndimage.median_filter(ratio, size=nmed)
        # subtracting the median map
        ratio[gmask] -= ratio_med[gmask]
    else:
        _logger.info('computing median in full array')
        ratio_med = numpy.median(ratio[gmask])
        ratio[gmask] -= ratio_med

    # Quantiles that contain nsig sigma in normal distribution
    qns = 100 * scipy.stats.norm.cdf(siglev)
    pns = 100 - qns
    _logger.info('percentiles at siglev=%f', siglev)
    _logger.info('low %f%% high %f%%', pns, qns)

    # in several blocks of shape nsig
    # we estimate sigma
    sigma = numpy.zeros_like(ratio)

    if mode == 'region':
        mshape = max_blk_coverage(blk=nsig, shape=ratio.shape)
        _logger.info('estimating sigma in boxes of %r', nsig)
        _logger.info('shape covered by boxes is  %r', mshape)
        block_gen = blk_nd_short(blk=nsig, shape=ratio.shape)
    else:
        mshape = ratio.shape
        _logger.info('estimating sigma in full array')
        # slice(None) is equivalent to [:]
        block_gen = itertools.repeat(slice(None), 1)

    for blk in block_gen:
        # mask for this region
        m = mask[blk] == PIXEL_VALID
        valid_points = numpy.ravel(ratio[blk][m])
        ls = scipy.stats.scoreatpercentile(valid_points, pns)
        hs = scipy.stats.scoreatpercentile(valid_points, qns)

        _logger.debug('score at percentiles')
        _logger.debug('low %f high %f', ls, hs)

        # sigma estimation
        sig = (hs - ls) / (2 * siglev)
        _logger.debug('sigma estimation is %f ', sig)

        # normalized points
        sigma[blk] = sig

    # fill regions of sigma not computed
    fill0 = ratio.shape[0] - mshape[0]
    fill1 = ratio.shape[1] - mshape[1]
    if fill0 > 0:
        _logger.info('filling %d rows in sigma image', fill0)
        sigma[:, mshape[0]:] = sigma[:, mshape[0] - fill0:mshape[0]]

    if fill1 > 0:
        _logger.info('filling %d columns in sigma image', fill1)
        sigma[mshape[1]:, :] = sigma[mshape[1] - fill1:mshape[1], :]

    # invalid_sigma = sigma <= 0.0
    # if numpy.any(invalid_sigma):
    #     _logger.info('updating mask with points where sigma <=0')
    #     mask, gmask, smask = update_mask(mask, gmask, invalid_sigma, PIXEL_HOT)
    #     invalid[mask == PIXEL_HOT] = HIGH_SIGMA

    ratio[gmask] /= sigma[gmask]

    f1_ratio = ratio[gmask]
    f1_mask = mask[gmask]
    f1_mask[f1_ratio >= uppercut] = PIXEL_HOT
    f1_mask[f1_ratio <= -lowercut] = PIXEL_DEAD
    mask[gmask] = f1_mask

    return ratio, mask, sigma


def robust_std(valid, central, siglev):
    import scipy.stats
    qns = 100 * scipy.stats.norm.cdf(siglev)
    pns = 100 - qns

    ls = scipy.stats.scoreatpercentile(valid.flat - central, pns)
    hs = scipy.stats.scoreatpercentile(valid.flat - central, qns)

    # sigma estimation
    sig = (hs - ls) / (2 * siglev)

    return sig


def comp_ratio(img1, img2, mask):
    mask = mask == 1
    mask1 = img1 <= 0
    mask2 = img2 <= 0
    mask3 = mask1 | mask2 | mask
    with numpy.errstate(divide='ignore', invalid='ignore'):
        ratio = img1 / img2
    ratio[mask3] = 0.0
    return ratio, mask3


def cosmetics(flat1, flat2=None, mask=None, lowercut=6.0, uppercut=6.0, siglev=2.0):
    """Find cosmetic defects in a detector using two flat field images.

    Two arrays representing flat fields of different exposure times are
    required. Cosmetic defects are selected as points that deviate
    significantly of the expected normal distribution of pixels in
    the ratio between `flat2` and `flat1`.

    The median of the ratio array is computed and subtracted to it.

    The standard deviation of the distribution of pixels is computed
    obtaining the percentiles nearest the pixel values corresponding to
    `nsig` in the normal CDF. The standar deviation is then the distance
    between the pixel values divided by two times `nsig`.
    The ratio image is then normalized with this standard deviation.

    The values in the ratio above `uppercut` are flagged as hot pixels,
    and those below `-lowercut` are flagged as dead pixels in the output mask.

    :parameter flat1: an array representing a flat illuminated exposure.
    :parameter flat2: an array representing a flat illuminated exposure.
    :parameter mask: an integer array representing initial mask.
    :parameter lowercut: values bellow this sigma level are flagged as dead pixels.
    :parameter uppercut: values above this sigma level are flagged as hot pixels.
    :parameter siglev: level to estimate the standard deviation.
    :returns: the updated mask

    """

    if flat2 is None:
        flat1, flat2 = flat2, flat1
        flat1 = numpy.ones_like(flat2)

    if type(mask) is not numpy.ndarray:
        mask = numpy.zeros(flat1.shape, dtype='int')

    ratio, mask = comp_ratio(flat1, flat2, mask)
    fratio1 = ratio[~mask]
    central = numpy.median(fratio1)
    std = robust_std(fratio1, central, siglev)
    mask_u = ratio > central + uppercut * std
    mask_d = ratio < central - lowercut * std
    mask_final = mask_u | mask_d | mask
    return mask_final


if __name__ == "__main__":
    flat1 = numpy.zeros((2, 2))
    flat1[0] = 1
    flat2 = flat1
    mascara = cosmetics(flat1, flat2)
    print(mascara)
    print(mascara.sum())
