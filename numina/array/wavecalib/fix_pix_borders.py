from __future__ import division
from __future__ import print_function

import numpy as np


def find_pix_borders(sp, sought_value):
    """Find useful region of a given spectrum

    Detemine the useful region of a given spectrum by skipping
    the initial (final) pixels with values equal to 'sought_value'.

    Parameters
    ----------
    sp : 1D numpy array
        Input spectrum.
    sought_value : int, float, bool
        Pixel value that indicate missing data in the spectrum.

    Returns
    -------
    jmin, jmax : tuple (integers)
        Valid spectrum region (in array coordinates, from 0 to
        NAXIS1 - 1). If the values of all the pixels in the spectrum 
        are equal to 'sought_value', the returned values are jmin=-1 
        and jmax=naxis1.

    """

    if sp.ndim != 1:
        raise ValueError('Unexpected number of dimensions:', sp.ndim)
    naxis1 = len(sp)

    jborder_min = -1
    jborder_max = naxis1

    # only spectra with values different from 'sought_value'
    if not np.alltrue(sp == sought_value):
        # left border
        while True:
            jborder_min += 1
            if sp[jborder_min] != sought_value:
                break
        # right border
        while True:
            jborder_max -= 1
            if sp[jborder_max] != sought_value:
                break

    return jborder_min, jborder_max


def fix_pix_borders(image2d, nreplace, sought_value, replacement_value):
    """Replace a few pixels at the borders of each spectrum.

    Set to 'replacement_value' 'nreplace' pixels at the beginning (at
    the end) of each spectrum just after (before) the spectrum value
    changes from (to) 'sought_value', as seen from the image borders.

    Parameters
    ----------
    image2d : numpy array
        Initial 2D image.
    nreplace : int
        Number of pixels to be replaced in each border.
    sought_value : int, float, bool
        Pixel value that indicate missing data in the spectrum.
    replacement_value : int, float, bool
        Pixel value to be employed in the 'nreplace' pixels.

    Returns
    -------
    image2d : numpy array
        Final 2D image.

    """

    naxis2, naxis1 = image2d.shape

    for i in range(naxis2):
        # only spectra with values different from 'sought_value'
        jborder_min, jborder_max = find_pix_borders(image2d[i, :],
                                                    sought_value=sought_value)
        # left border
        if jborder_min != -1:
            j1 = jborder_min
            j2 = min(j1 + nreplace, naxis1)
            image2d[i, j1:j2] = replacement_value
        # right border
        if jborder_max != naxis1:
            j2 = jborder_max + 1
            j1 = max(j2 - nreplace, 0)
            image2d[i, j1:j2] = replacement_value

    return image2d
