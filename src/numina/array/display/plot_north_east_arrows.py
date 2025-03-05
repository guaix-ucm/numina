#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Unit, Quantity
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np


def plot_north_east_arrows(
        ax,
        wcs2d,
        arrow_length,
        scale_length=None,
        scale_location=1,
        fits_criterion=False,
        color='grey',
        color_scale=None,
        fontsize_scale_relative_factor=0.8,
        verbose=False
):
    """Display North & East arrow using WCS information.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Instance of `matplotlib.axes.Axes`.
    wcs2d : astropy.wcs.WCS
        Instance of a 2D `astropy.wcs.WCS`.
    arrow_length : Quantity
        Arrow length with units.
    scale_length : Quantity, optional
        If None, 'arrow_length' is assumed.
    scale_location : int
        Location of the scale (with the same length as the NE arrows).
        0: do not plot ruler
        1: upper right
        2: upper left
        3: lower left
        4: lower right
    fits_criterion : bool
        If True, assume FITS criteria to compute the central pixel.
    color : str
        Color to use for plotting the NE arrows.
    color_scale : str or None
        Color scale to use for plotting the scale.
        If None, 'color' is employed.
    fontsize_scale_relative_factor : float
        Scaling factor to determine the font size employed in the
        scale. A value of 1.0 indicates that the font size is the
        same as the one employed in the plot labels.
    verbose : bool
        If True, display additional information.
    """

    # protections
    if scale_length is None:
        scale_length = arrow_length
    if not isinstance(arrow_length, Quantity):
        raise TypeError("arrow_length must be a Quantity")
    if not isinstance(scale_length, Quantity):
        raise TypeError("scale_length must be a Quantity")
    if not isinstance(wcs2d, WCS):
        raise TypeError("wcs2d must be a WCS object.")
    if scale_location not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid scale_location value {scale_location}.")

    if verbose:
        print(wcs2d)
    x_center, y_center = wcs2d.wcs.crpix
    center_coord = wcs2d.pixel_to_world(x_center - 1, y_center - 1)
    if verbose:
        print(f'{x_center=}, {y_center=}')
        print(f'{center_coord=}')

    # define North and East direction
    north = SkyCoord(ra=center_coord.ra, dec=center_coord.dec + Angle(arrow_length))
    east = SkyCoord(ra=center_coord.ra + Angle(arrow_length), dec=center_coord.dec)

    # convert celestial coordinates to pixel coordinates
    north_pix = wcs2d.world_to_pixel(north)
    east_pix = wcs2d.world_to_pixel(east)
    if fits_criterion:
        north_pix = (north_pix[0] + 1, north_pix[1] + 1)
        east_pix = (east_pix[0] + 1, east_pix[1] + 1)

    if verbose:
        print(f'north_pix: {north_pix}')
        print(f'east_pix: {east_pix}')

    # North and East vectors
    arrowprops = dict(color=color, shrinkA=0, shrinkB=0, arrowstyle='<-')
    ax.annotate('N', xy=(x_center, y_center), xytext=north_pix, arrowprops=arrowprops, color=color)
    ax.annotate('E', xy=(x_center, y_center), xytext=east_pix, arrowprops=arrowprops, color=color)

    # scale
    if scale_location != 0:
        xmin, xmax = ax.get_xlim()
        dx = xmax - xmin
        ymin, ymax = ax.get_ylim()
        dy = ymax - ymin
        pixel_scales = proj_plane_pixel_scales(wcs2d)
        # check the scale is the same in both axis
        if np.isclose(pixel_scales[0], pixel_scales[1]):
            pixel_scale = pixel_scales[0] * Unit('deg')
            scale_pixels = scale_length / pixel_scale
            if scale_location in [1, 4]:
                x_scale_max = xmax - dx / 20
                x_scale_min = x_scale_max - scale_pixels
            else:
                x_scale_min = xmin + dx / 20
                x_scale_max = x_scale_min + scale_pixels
            if scale_location in [1, 2]:
                y_scale = ymax - dy / 10
            else:
                y_scale = ymin + dy / 20

            if color_scale is None:
                color_scale = color
            ax.plot(np.array([x_scale_min, x_scale_max]),
                    np.array([y_scale, y_scale]),
                    color=color_scale)
            ax.plot(np.array([x_scale_min, x_scale_min]),
                    np.array([y_scale-dy/80, y_scale+dy/80]),
                    color=color_scale)
            ax.plot(np.array([x_scale_max, x_scale_max]),
                    np.array([y_scale-dy/80, y_scale+dy/80]),
                    color=color_scale)
            scale_font_size = ax.xaxis.get_label().get_fontsize() * fontsize_scale_relative_factor
            ax.text((x_scale_min + x_scale_max) / 2, y_scale + 0.02*dy,
                    f'{scale_length}', fontsize=scale_font_size,
                    ha='center', va='bottom', color=color_scale)
