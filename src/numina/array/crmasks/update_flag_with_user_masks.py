#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Update the flag array with user-defined cosmic ray masks."""
import numpy as np


def update_flag_with_user_masks(flag, pixels_to_be_flagged_as_cr, pixels_to_be_ignored_as_cr, _logger):
    """Update the flag array with user-defined masks.

    Parameters
    ----------
    flag : 2D numpy array
        The input flag array to be updated.
    pixels_to_be_flagged_as_cr : list of (x, y) tuples, or None
        List of pixel coordinates to be included in the masks
        (FITS criterium; first pixel is (1, 1)).
    pixels_to_be_ignored_as_cr : list of (x, y) tuples, or None
        List of pixel coordinates to be excluded from the masks
        (FITS criterium; first pixel is (1, 1)).
    _logger : logging.Logger
        The logger to use for logging.

    Returns
    -------
    None
    """
    # Include pixels to be forced to be masked
    if pixels_to_be_flagged_as_cr is not None:
        ix_pixels_to_be_flagged_as_cr = np.array([p[0] for p in pixels_to_be_flagged_as_cr], dtype=int) - 1
        iy_pixels_to_be_flagged_as_cr = np.array([p[1] for p in pixels_to_be_flagged_as_cr], dtype=int) - 1
        if np.any(ix_pixels_to_be_flagged_as_cr < 0) or np.any(ix_pixels_to_be_flagged_as_cr >= flag.shape[1]):
            raise ValueError("Some x coordinates in pixels_to_be_flagged_as_cr are out of bounds.")
        if np.any(iy_pixels_to_be_flagged_as_cr < 0) or np.any(iy_pixels_to_be_flagged_as_cr >= flag.shape[0]):
            raise ValueError("Some y coordinates in pixels_to_be_flagged_as_cr are out of bounds.")
        neff = 0
        for iy, ix in zip(iy_pixels_to_be_flagged_as_cr, ix_pixels_to_be_flagged_as_cr):
            if not flag[iy, ix]:
                flag[iy, ix] = True
                neff += 1
            else:
                _logger.warning("Pixel (%d, %d) to be masked was already masked.", ix + 1, iy + 1)
        _logger.info("Added %d/%d user-defined pixels to be masked.", neff, len(pixels_to_be_flagged_as_cr))

    # Exclude pixels to be excluded from the mask
    if pixels_to_be_ignored_as_cr is not None:
        ix_pixels_to_be_ignored_as_cr = np.array([p[0] for p in pixels_to_be_ignored_as_cr], dtype=int) - 1
        iy_pixels_to_be_ignored_as_cr = np.array([p[1] for p in pixels_to_be_ignored_as_cr], dtype=int) - 1
        if np.any(ix_pixels_to_be_ignored_as_cr < 0) or np.any(ix_pixels_to_be_ignored_as_cr >= flag.shape[1]):
            raise ValueError("Some x coordinates in pixels_to_be_ignored_as_cr are out of bounds.")
        if np.any(iy_pixels_to_be_ignored_as_cr < 0) or np.any(iy_pixels_to_be_ignored_as_cr >= flag.shape[0]):
            raise ValueError("Some y coordinates in pixels_to_be_ignored_as_cr are out of bounds.")
        neff = 0
        for iy, ix in zip(iy_pixels_to_be_ignored_as_cr, ix_pixels_to_be_ignored_as_cr):
            if flag[iy, ix]:
                flag[iy, ix] = False
                neff += 1
            else:
                _logger.warning("Pixel (%d, %d) to be unmasked was not masked.", ix + 1, iy + 1)
        _logger.info("Removed %d/%d user-defined pixels from the mask.", neff, len(pixels_to_be_ignored_as_cr))
