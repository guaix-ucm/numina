#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Correct cosmic rays applying previously computed masks."""
import logging

from astropy.table import Table
import numpy as np
import numpy.ma as ma
from rich.logging import RichHandler

from .valid_parameters import VALID_COMBINATIONS


def apply_crmasks(list_arrays, hdul_masks=None, combination=None, use_lamedian=False,
                  dtype=np.float32, apply_flux_factor=True, bias=None):
    """
    Correct cosmic rays applying previously computed masks.

    The input arrays are bias subtracted, and optionally re-scaled by
    a flux factor read from the header of `hdul_masks`. Then, they are
    combined using the specified combination method.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined.
    hdul_masks : HDUList
        The HDUList containing the mask arrays for cosmic ray removal.
        The masks and auxiliary data should be in the following extensions:
        - 'MEDIANCR' mask for the median combination
        - 'MEANCRT' mask for the mean combination
        - 'CRMASK1', 'CRMASK2', etc. masks for the mean combination with
          individual masks
        - 'LAMEDIAN' (optional) image with the lacosmic-corrected median array
        - 'RPMEDIAN' (optional) pixels to be replaced by the local median value
          around them. Stored as a binary table with four columns:
          'X_pixel', 'Y_pixel', 'X_width', 'Y_width'. The (X, Y) coordinates
          are given following the FITS convention (starting at 1).
    combination : str
        The type of combination to apply. There are three options:
        - 'mediancr', the median combination is applied, and masked pixels
        (those equal to 1 in extension 'MEDIANCR' of `hdul_masks`) are
        replaced by the minimum value of the corresponding pixel in the
        input arrays.
        - 'meancrt', the mean combination is applied, and masked pixels
        (those equal to 1 in extension 'MEANCRT' of `hdul_masks`) are
        replaced by the mediancr value.
        - 'meancr', the mean combination is applied making use of
        the individual mask of each image (extensions 'CRMASK1', 'CRMASK2',
        etc. in `hdul_masks`). Those pixels that are masked in all the individual
        images are replaced by the minimum value of the corresponding pixel
        in the input arrays.
    use_lamedian : bool, optional
        If True, and if the extension 'LAMEDIAN' is present in `hdul_masks`,
        the lacosmic-corrected median array is used instead of the minimum
        value at each pixel. This affects differently depending on the
        combination method:
        - 'mediancr': all the masked pixels in the mask MEDIANCR are replaced.
        - 'meancrt': only the pixels coincident in masks MEANCRT and MEDIANCR;
          the rest of the pixels flagged in the mask MEANCRT are replaced by
          the value obtained when the combination method is 'mediancr'.
        - 'meancr': only the pixels flagged in all the individual exposures
          (i.e., those flagged simulatenously in all the CRMASKi masks);
          the rest of the pixels flagged in any of the `CRMASK1`, `CRMASK2`, etc.
          masks are replaced by the corresponding masked mean.
        Default is False.
    apply_flux_factor : bool, optional
        If True, the flux factor is applied to the input arrays before
        combining them. The flux factor is read from the header of the
        `hdul_masks` (keywords 'FLUXF1', 'FLUXF2', etc.). Default is True.
    bias : float or 2D array, optional
        The bias level to be subtracted from the input arrays. If a float is
        provided, it is assumed to be constant for all pixels. If a 2D array
        is provided, it must have the same shape as the input arrays. If None,
        the bias is assumed to be zero for all pixels. Default is None.
    dtype : data-type, optional
        The desired data type for the output arrays (default is np.float32).

    Returns
    -------
    combined2d: 2D array
        The combined bias-subtracted array with masked pixels replaced
        accordingly depending on the combination method.
    variance2d : 2D array
        The variance of the input arrays along the first axis.
    map2d : 2D array
        The number of input pixels used to compute the median at each pixel.
    """

    _logger = logging.getLogger(__name__)
    rich_configured = any(isinstance(handler, RichHandler) for handler in _logger.handlers)

    # Check that the input is a list
    if not isinstance(list_arrays, list):
        raise TypeError("Input must be a list of arrays.")

    # Check that the combination method is valid
    if combination not in VALID_COMBINATIONS:
        raise ValueError(f"Combination: {combination} must be one of {VALID_COMBINATIONS}.")

    # If use_lamedian is True, check that the extension 'LAMEDIAN' is present
    if use_lamedian:
        if 'LAMEDIAN' not in hdul_masks:
            raise ValueError("use_lamedian is True, but extension 'LAMEDIAN' is not present in hdul_masks.")

    # Check that the list contains numpy 2D arrays
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in list_arrays):
        raise ValueError("All elements in the list must be 2D numpy arrays.")

    # Check that the list contains at least 3 arrays
    num_images = len(list_arrays)
    if num_images < 3:
        raise ValueError("At least 3 images are required for a useful combination.")

    # Check that all arrays have the same shape
    for i, array in enumerate(list_arrays):
        if array.shape != list_arrays[0].shape:
            raise ValueError(f"Array {i} has a different shape than the first array.")
    naxis2, naxis1 = list_arrays[0].shape

    # Log the number of input arrays and their shapes
    _logger.info("number of input arrays: %d", len(list_arrays))
    for i, array in enumerate(list_arrays):
        _logger.info("array %d shape: %s, dtype: %s", i, array.shape, array.dtype)

    # Define the bias
    if bias is None:
        bias = np.zeros((naxis2, naxis1), dtype=float)
        _logger.info("bias not defined, assuming bias=0.0 for all pixels.")
    elif isinstance(bias, (float, int)):
        bias = np.full((naxis2, naxis1), bias, dtype=float)
        _logger.info("bias defined as a constant value: %f", bias[0, 0])
    elif isinstance(bias, np.ndarray):
        if bias.shape != (naxis2, naxis1):
            raise ValueError(f"bias must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("bias defined as a 2D array with shape: %s", bias.shape)
    else:
        raise TypeError(f"Invalid type for bias: {type(bias)}. Must be float, int, or numpy array.")

    # Read the flux factor from the masks
    _logger.info("apply_flux_factor: %s", apply_flux_factor)
    if apply_flux_factor:
        flux_factor = []
        for i in range(num_images):
            flux_factor.append(hdul_masks[0].header[f'FLUXF{i+1}'])
        flux_factor = np.array(flux_factor, dtype=float)
    else:
        flux_factor = np.ones(num_images, dtype=float)
    _logger.info("flux factor values: %s", str(flux_factor))

    # Convert the list of arrays to a 3D numpy array
    shape3d = (num_images, naxis2, naxis1)
    image3d = np.zeros(shape3d, dtype=dtype)
    _logger.info("applying bias and flux factors to input arrays...")
    for i, array in enumerate(list_arrays):
        image3d[i] = (array.astype(dtype) - bias) / flux_factor[i]

    # Compute minimum and median along the first axis of image3d
    min2d_rescaled = np.min(image3d, axis=0)
    median2d = np.median(image3d, axis=0)

    # Apply the requested combination method
    if rich_configured:
        rlabel_combination = f"[bold][magenta]{combination}[/magenta][/bold]"
    else:
        rlabel_combination = f"{combination}"
    _logger.info("applying combination method: %s", rlabel_combination)
    _logger.info("using crmasks in %s", hdul_masks[0].header['UUID'])
    if combination in ['mediancr', 'meancrt']:
        # Define the mask_mediancr
        mask_mediancr = hdul_masks['MEDIANCR'].data.astype(bool)
        _logger.info("applying mask MEDIANCR: %d masked pixels", np.sum(mask_mediancr))
        # Replace the masked pixels with the minimum value
        # of the corresponding pixel in the input arrays
        median2d_corrected = median2d.copy()
        if use_lamedian:
            median2d_corrected[mask_mediancr] = hdul_masks['LAMEDIAN'].data[mask_mediancr]
        else:
            median2d_corrected[mask_mediancr] = min2d_rescaled[mask_mediancr]

    if combination == 'mediancr':
        combined2d = median2d_corrected
        # Define the variance and map arrays
        variance2d = np.var(image3d, axis=0, ddof=1)
        variance2d[mask_mediancr] = 0.0  # Set variance to 0 for the masked pixels
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images
        map2d[mask_mediancr] = 1  # Set the map to 1 for the masked pixels
    elif combination == 'meancrt':
        # Define the mask_meancr
        mask_meancrt = hdul_masks['MEANCRT'].data.astype(bool)
        _logger.info("applying mask MEANCRT: %d masked pixels", np.sum(mask_meancrt))
        mean2d = np.mean(image3d, axis=0)
        # Replace the masked pixels in mean2d with the median2d_corrected value
        mean2d_corrected = mean2d.copy()
        mean2d_corrected[mask_meancrt] = median2d_corrected[mask_meancrt]
        combined2d = mean2d_corrected
        # Define the variance and map arrays
        variance2d = np.var(image3d, axis=0, ddof=1)
        variance2d[mask_meancrt] = 0.0  # Set variance to 0 for the masked pixels
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images
        map2d[mask_meancrt] = 1  # Set the map to 1 for the masked pixels
    elif combination == 'meancr':
        image3d_masked = ma.array(
            np.zeros(shape3d, dtype=dtype),
            mask=np.full(shape3d, fill_value=True, dtype=bool)
        )
        # Loop through each image and apply the corresponding mask
        total_mask = np.zeros((naxis2, naxis1), dtype=int)
        for i in range(num_images):
            image3d_masked[i, :, :] = image3d[i, :, :]
            mask = hdul_masks[f'CRMASK{i+1}'].data
            _logger.info("applying mask %s: %d masked pixels", f'CRMASK{i+1}', np.sum(mask))
            total_mask += mask.astype(int)
            image3d_masked[i, :, :].mask = mask.astype(bool)
        # Compute the mean of the masked 3D array
        combined2d = ma.mean(image3d_masked, axis=0).data
        # Replace pixels without data with the minimum value
        mask_nodata = total_mask == num_images
        if np.any(mask_nodata):
            if use_lamedian:
                _logger.info("replacing %d pixels without data by the LAMEDIAN value", np.sum(mask_nodata))
                combined2d[mask_nodata] = hdul_masks['LAMEDIAN'].data[mask_nodata]
            else:
                _logger.info("replacing %d pixels without data by the minimum value", np.sum(mask_nodata))
                combined2d[mask_nodata] = min2d_rescaled[mask_nodata]
        else:
            _logger.info("no pixels without data found, no replacement needed")
        # Define the variance and map arrays
        variance2d = ma.var(image3d_masked, axis=0, ddof=1).data
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images - total_mask
    else:
        raise ValueError(f"Invalid combination method: {combination}. "
                         f"Valid options are {VALID_COMBINATIONS}.")

    # Replaced pixels defined in the RPMEDIAN extension of hdul_masks by
    # the local median around them (excluding the considered pixel)
    if 'RPMEDIAN' in hdul_masks:
        _logger.info("replacing pixels defined in RPMEDIAN by the local median around them")
        table_rpmedian = Table(hdul_masks['RPMEDIAN'].data)
        for row in table_rpmedian:
            xpixel = row['X_pixel'] - 1  # Convert from FITS to zero-based index
            ypixel = row['Y_pixel'] - 1  # Convert from FITS to zero-based index
            xwidth = row['X_width']
            ywidth = row['Y_width']
            # Define the box around the pixel, ensuring it is within image bounds
            x1 = max(xpixel - xwidth // 2, 0)
            x2 = min(xpixel + xwidth // 2 + 1, naxis1)
            y1 = max(ypixel - ywidth // 2, 0)
            y2 = min(ypixel + ywidth // 2 + 1, naxis2)
            # Extract the local box from the combined2d array
            local_box = combined2d[y1:y2, x1:x2]
            # Create a mask for the local box that excludes the central pixel
            local_mask = np.ones(local_box.shape, dtype=bool)
            if y1 <= ypixel < y2 and x1 <= xpixel < x2:
                local_mask[ypixel - y1, xpixel - x1] = False
            # Compute the median of the unmasked pixels in the local box
            if np.any(local_mask):
                local_median = np.median(local_box[local_mask])
                _logger.info(f"- pixel (x={xpixel+1}, y={ypixel+1}): "
                             f"old value={combined2d[ypixel, xpixel]:.2f}, new value={local_median:.2f}")
                combined2d[ypixel, xpixel] = local_median
            else:
                _logger.warning(f"cannot compute local median for pixel (x={xpixel+1}, y={ypixel+1}) "
                                "as no surrounding pixels are available")

    return combined2d.astype(dtype), variance2d.astype(dtype), map2d
