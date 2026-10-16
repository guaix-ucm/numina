#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Combination routines"""

import datetime
import logging
import uuid
import contextlib

from astropy.io import fits

from numina.array import combine
from numina.datamodel import get_imgid

# Methods of numina.array.combine that use masks of cosmic rays
CRMASK_METHODS = ("mediancr", "meancrt", "meancr", "meancr2")


def basic_processing_with_combination(
    rinput, reduction_flows, method=combine.mean, method_kwargs=None, errors=True, prolog=None
):
    """Combine the frames of the observation result of a recipe and process the result.

    It calls :func:`basic_processing_with_combination_frames` with the
    frames of ``rinput.obresult``. With the methods that use masks of
    cosmic rays (:data:`CRMASK_METHODS`), the masks are ``rinput.crmasks``.
    """
    if method.__name__ in CRMASK_METHODS:
        crmasks = getattr(rinput, "crmasks", None)
    else:
        crmasks = None
    return basic_processing_with_combination_frames(
        rinput.obresult.frames,
        reduction_flows,
        method=method,
        method_kwargs=method_kwargs,
        errors=errors,
        prolog=prolog,
        crmasks=crmasks,
    )


def basic_processing_with_combination_frames(
    frames, reduction_flows, method=combine.mean, method_kwargs=None, errors=True, prolog=None, crmasks=None
):
    """Process frames with reduction flows and combine them.

    The reduction flows are those returned by ``BaseRecipe.init_filters``,
    one for each sequence of getters of ``BaseRecipe.create_getters``.
    All the flows but the last are applied to each frame, then the frames
    are combined and the last flow is applied to the combined image. With
    a single flow (a callable or a list of one flow), the frames are
    combined and then processed, and with an empty list they are only
    combined.

    With the methods that use masks of cosmic rays (:data:`CRMASK_METHODS`),
    all the flows are applied to each frame before the combination, and
    the UUID of the masks is recorded in the HISTORY of the result.

    Parameters
    ----------
    frames : list of DataFrame
        Frames to combine.
    reduction_flows : callable or list of callable
        Flows applied to the frames and to the combined image.
    method, method_kwargs, errors, prolog
        As in :func:`combine_frames`.
    crmasks : DataFrame, optional
        Masks of cosmic rays, required by the methods in
        :data:`CRMASK_METHODS`.

    Returns
    -------
    astropy.io.fits.HDUList
        The combined and processed image.

    Raises
    ------
    ValueError
        If the method uses masks of cosmic rays and `crmasks` is None.
    """
    if isinstance(reduction_flows, (list, tuple)):
        image_flows = list(reduction_flows[:-1])
        combined_flows = list(reduction_flows[-1:])
    else:
        image_flows = []
        combined_flows = [reduction_flows]

    use_crmasks = method.__name__ in CRMASK_METHODS
    if use_crmasks:
        if crmasks is None:
            raise ValueError(f"Cosmic ray masks are required for {method.__name__}")
        # the combination with masks requires processed images
        image_flows.extend(combined_flows)
        combined_flows = []

    with contextlib.ExitStack() as stack:
        hduls = [stack.enter_context(dframe.open()) for dframe in frames]
        for flow in image_flows:
            hduls = [flow(hdul) for hdul in hduls]

        if use_crmasks:
            masks = stack.enter_context(crmasks.open())
            result = combine_imgs(
                hduls, method=method, method_kwargs=method_kwargs, errors=errors, prolog=prolog, crmasks=masks
            )
            result[0].header.add_history(f'Masks uuid:{masks[0].header["UUID"]}')
        else:
            result = combine_imgs(hduls, method=method, method_kwargs=method_kwargs, errors=errors, prolog=prolog)

    for flow in combined_flows:
        result = flow(result)

    return result


def combine_frames(frames, method=combine.mean, method_kwargs=None, errors=True, prolog=None):
    """Combine the images of a list of frames.

    The frames are opened and combined with :func:`combine_imgs`.

    Parameters
    ----------
    frames : list of DataFrame
        Frames to combine.
    method, method_kwargs, errors, prolog
        As in :func:`combine_imgs`.

    Returns
    -------
    astropy.io.fits.HDUList
        The combined image.
    """

    with contextlib.ExitStack() as stack:
        hduls = [stack.enter_context(dframe.open()) for dframe in frames]
        result = combine_imgs(hduls, method=method, method_kwargs=method_kwargs, errors=errors, prolog=prolog)

    return result


def combine_imgs(
    hduls, method=combine.mean, method_kwargs=None, errors=True, prolog=None, crmasks=None, use_lamedian=False
):
    """Combine the primary HDUs of a list of images.

    The header of the result is the header of the first image, with
    HISTORY entries that record the method and the combined images,
    NUM-NCOM, the number of raw images combined, and a new UUID. The
    extensions of the first image are copied to the result.

    Parameters
    ----------
    hduls : list of astropy.io.fits.HDUList
        Images to combine, at least one.
    method : callable, optional
        Combination function of :mod:`numina.array.combine`, the mean
        by default.
    method_kwargs : dict, optional
        Arguments passed to `method`. The default 'dtype' is 'float32'.
    errors : bool, optional
        If True, the variance and the number of pixels combined are
        appended as the extensions VARIANCE and MAP.
    prolog : str, optional
        Text added to the HISTORY of the result before the other entries.
    crmasks : astropy.io.fits.HDUList, optional
        Masks of cosmic rays, passed to the methods that use them
        (:data:`CRMASK_METHODS`).
    use_lamedian : bool, optional
        Not used.

    Returns
    -------
    astropy.io.fits.HDUList
        The combined image.

    Raises
    ------
    ValueError
        If `hduls` is empty.
    """

    _logger = logging.getLogger(__name__)

    cnum = len(hduls)
    if cnum == 0:
        raise ValueError("number of HDUList == 0")

    first_image = hduls[0]
    base_header = first_image[0].header.copy()
    last_header = hduls[-1][0].header.copy()

    method_kwargs = method_kwargs or {}
    if "dtype" not in method_kwargs:
        method_kwargs["dtype"] = "float32"

    _logger.info(f"stacking {cnum:d} images using '{method.__name__}'")
    if method.__name__ in CRMASK_METHODS:
        combined_data = method([d[0].data for d in hduls], crmasks=crmasks, **method_kwargs)
    else:
        combined_data = method([d[0].data for d in hduls], **method_kwargs)

    hdu = fits.PrimaryHDU(combined_data[0], header=base_header)
    _logger.debug("update result header")
    if prolog:
        _logger.debug("write prolog")
        hdu.header["history"] = prolog
    hdu.header["history"] = f"Combined {cnum:d} images using '{method.__name__}'"
    t_str = datetime.datetime.now(datetime.UTC).isoformat()
    hdu.header["history"] = f"Combination time {t_str}"

    for img in hduls:
        hdu.header["history"] = f"Image {get_imgid(img)}"

    prevnum = base_header.get("NUM-NCOM", 1)
    hdu.header["NUM-NCOM"] = prevnum * cnum
    hdu.header["UUID"] = str(uuid.uuid1())

    # Headers of last image, this is an EMIRISM
    if "TSUTC2" in hdu.header:
        hdu.header["TSUTC2"] = last_header["TSUTC2"]

    # Copy extensions and then append 'variance' and 'map'
    result = fits.HDUList([hdu])
    for ext in first_image[1:]:
        result.append(ext.copy())
    # Append error extensions
    if errors:
        varhdu = fits.ImageHDU(combined_data[1], name="VARIANCE")
        result.append(varhdu)
        num = fits.ImageHDU(combined_data[2].astype("int16"), name="MAP")
        result.append(num)

    return result


def main(args=None):
    """Command line program that combines FITS images with the mean or the median"""
    import argparse

    parser = argparse.ArgumentParser(prog="combine")
    parser.add_argument("-o", "--output", default="combined.fits")
    parser.add_argument("-e", "--errors", default=False, action="store_true")
    parser.add_argument("--method", default="mean", choices=["mean", "median"])
    parser.add_argument("image", nargs="+")
    args = parser.parse_args(args)

    if args.method == "mean":
        method = combine.mean
    elif args.method == "median":
        method = combine.median
    else:
        raise ValueError(f"wrong method {args.method}")

    errors = args.errors
    with contextlib.ExitStack() as stack:
        hduls = [stack.enter_context(fits.open(fname)) for fname in args.image]
        result = combine_imgs(hduls, method=method, errors=errors, prolog=None)

    result.writeto(args.output, overwrite=True)


if __name__ == "__main__":
    main()
