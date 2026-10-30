#
# Copyright 2019-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


import datetime
import collections.abc
import sys
import uuid
import warnings

import numpy
import astropy.io.fits as fits

import numina.array.combine as C
from numina.keydef import KeyDefinition


def _m_base(calc, img, mask, region):
    """Apply ``calc`` to the pixels of the primary HDU of ``img`` in ``region`` where ``mask`` is not > 0."""
    full_f = img[0].data
    arr = full_f[region]
    if mask is not None:
        mask_sub = mask[region]
        invalid = mask_sub > 0
        arr = arr[~invalid]
    # mode_half_sample works only with 1D arrays
    return calc(arr.ravel())


def _m_mean(img, mask, region):
    """Return the mean of the valid pixels of the image in the region."""
    return _m_base(numpy.mean, img, mask, region)


def _m_median(img, mask, region):
    """Return the median of the valid pixels of the image in the region."""
    return _m_base(numpy.median, img, mask, region)


def _m_mode(img, mask, region):
    """Return the mode (half sample mode) of the valid pixels of the image in the region."""
    from numina.array.mode import mode_half_sample

    return _m_base(mode_half_sample, img, mask, region)


_method_map = {"mean": _m_mean, "median": _m_median, "mode": _m_mode}


class Extension:
    """Reference to an extension of each image, used to pass the masks to ``combine``.

    Parameters
    ----------
    name : str
        Name of the extension.
    """

    def __init__(self, name):
        self.name = name


class _ExtractKeyword:
    """The value of a keyword in each image, defined by a KeyDefinition

    Parameters
    ----------
    keydef : KeyDefinition
        Definition of the keyword.
    """

    def __init__(self, keydef):
        self.keydef = keydef

    def __call__(self, img, mask, region):
        """Return the value of the keyword in ``img``, raise KeyError if it is not found."""
        value = self.keydef(img)
        if value is None:
            raise KeyError(f"keyword {self.keydef.key!r} not found in HDU {self.keydef.ext!r}")
        return value


def _is_schema_keyword(value):
    """True if `value` is a SchemaKeyword of the deprecated numina.frame.schema"""
    # The module is not imported here, it warns when imported
    schema = sys.modules.get("numina.frame.schema")
    return schema is not None and isinstance(value, schema.SchemaKeyword)


def _inspect_method(value):
    """Convert a value of ``zeros``, ``scales`` or ``weights`` of ``combine`` into a sequence or a callable.

    Returns
    -------
    value : object
        None, a sequence with a value per image or a callable ``func(img, mask, region)``.
    is_func : bool
        True if ``value`` is a callable to be applied to each image.

    Raises
    ------
    KeyError
        If ``value`` is a string that is not a valid method.
    TypeError
        If ``value`` is not of a valid type.
    """
    if value is None:
        return value, False
    elif isinstance(value, str):
        try:
            return _method_map[value], True
        except KeyError:
            raise KeyError(f"invalid method {value}, valid methods are {sorted(_method_map)}") from None
    elif isinstance(value, collections.abc.Sequence):
        return value, False
    elif isinstance(value, KeyDefinition):
        return _ExtractKeyword(value), True
    elif _is_schema_keyword(value):
        warnings.warn(
            "passing a SchemaKeyword to combine is deprecated, it will be removed in 0.42, "
            "use numina.keydef.KeyDefinition",
            DeprecationWarning,
            stacklevel=3,
        )
        return _ExtractKeyword(KeyDefinition(value.name)), True
    elif callable(value):
        return value, True
    else:
        raise TypeError("no callable")


def combine(
    method,
    images,
    masks=None,
    dtype=None,
    region=None,
    zeros=None,
    scales=None,
    weights=None,
    include_variance=True,
    datamodel=None,
    method_name="",
):
    """Combine HDUList objects using algorithm 'method'

    `zeros`, `scales` and `weights` can be a sequence with a value for
    each image, the name of a method ("mean", "median" or "mode") applied
    to the data of each image, a :class:`~numina.keydef.KeyDefinition`
    of a keyword of the images, or a callable ``func(img, mask, region)``.
    """

    import numina.datamodel

    nimages = len(images)

    # processing masks
    if masks is None:
        intl_masks = [None for _ in images]
    elif isinstance(masks, Extension):
        intl_masks = [img[masks.name].data for img in images]
    elif isinstance(masks, collections.abc.Sequence):
        intl_masks = [mask[0].data for mask in masks]
    else:
        raise TypeError("mask in invalid")

    if len(intl_masks) != nimages:
        raise TypeError("len(masks) != len(images)")

    # Processing region
    if region is None:
        region = Ellipsis  # meaning arr[...]

    # processing zeros, scales and weights
    num_values = dict(zeros=None, scales=None, weights=None)
    arg_values = dict(zeros=zeros, scales=scales, weights=weights)

    for key in num_values:
        arg = arg_values[key]  # locals()[key]
        value_arg, arg_is_func = _inspect_method(arg)
        if arg_is_func:
            num_values[key] = [value_arg(img, mask, region) for img, mask in zip(images, intl_masks)]
        else:
            num_values[key] = value_arg
        if num_values[key] is not None:
            # Intl combine works only with np arrays
            num_values[key] = numpy.array(num_values[key])

    arrays = [hdu[0].data for hdu in images]
    if masks is None:
        fmasks = None
    else:
        fmasks = intl_masks

    out = C.generic_combine(
        method,
        arrays,
        masks=fmasks,
        dtype=dtype,
        out=None,
        zeros=num_values["zeros"],
        scales=num_values["scales"],
        weights=num_values["weights"],
    )

    # Build HDUList
    headers = [img[0].header for img in images]

    base_header = headers[0].copy()
    hdu1 = fits.PrimaryHDU(data=out[0], header=base_header)
    list_of_hdu = [hdu1]

    # _logger.debug('update result header')
    prolog = ""
    if prolog:
        # _logger.debug('write prolog')
        hdu1.header["history"] = prolog

    if method_name != "":
        # _logger.info("Combined %d images using '%s'", nimages, method.__name__)
        hdu1.header["history"] = f"Combined {nimages:d} images using '{method_name}'"
    else:
        hdu1.header["history"] = f"Combined {nimages:d} images"

    t_str = datetime.datetime.now(datetime.UTC).isoformat()
    hdu1.header["history"] = f"Combination time {t_str}"

    if datamodel is None:
        datamodel = numina.datamodel.DataModel()

    for idx, img in enumerate(images, start=1):
        hdu1.header["history"] = f"Image{idx} {datamodel.get_imgid(img)}"

    prevnum = base_header.get("NUM-NCOM", 1)

    hdu1.header["NUM-NCOM"] = prevnum * nimages
    hdu1.header["UUID"] = str(uuid.uuid1())

    # Headers of last image
    # hdu1.header['TSUTC2'] = headers[-1]['TSUTC2']

    if include_variance:
        varhdu = fits.ImageHDU(out[1], name="VARIANCE")
        list_of_hdu.append(varhdu)

    num = fits.ImageHDU(out[2].astype("uint8"), name="MAP")
    list_of_hdu.append(num)

    result = fits.HDUList(list_of_hdu)
    return result


def __getattr__(name):
    """Return ``SchemaKeyword`` for the deprecated name ``Keyword``, with a warning."""
    if name == "Keyword":
        warnings.warn(
            "numina.frame.combine.Keyword is deprecated, it will be removed in 0.42, "
            "use numina.keydef.KeyDefinition",
            DeprecationWarning,
            stacklevel=2,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from numina.frame.schema import SchemaKeyword

        return SchemaKeyword
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
