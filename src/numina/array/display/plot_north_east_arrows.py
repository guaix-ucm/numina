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
        arrow_length=None,
        scale_length=None,
        scale_location=1,
        parangle=None,
        fits_criterion=False,
        color='grey',
        color_scale=None,
        color_parangle=None,
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
    arrow_length : astropy.coordinates.Angle, optional
        Arrow length with Angle units.
    scale_length : Astropy.coordinates.Angle
        Size of the scale indication, with Angle units.
        If None, 'arrow_length' is assumed.
    scale_location : int
        Location of the scale (with the same length as the NE arrows).
        0: do not plot ruler
        1: upper right
        2: upper left
        3: lower left
        4: lower right
    parangle : Astropy.coordinates.Angle
        Parallactic angle with Angle units. Displayed only if this
        parameter is given.
    fits_criterion : bool
        If True, assume FITS criteria to compute the central pixel.
    color : str
        Color for plotting the NE arrows.
    color_scale : str or None
        Color for plotting the scale segment.
        If None, 'color' is employed.
    color_parangle : str or None
        Color for parangle line. If none, 'color' is employed.
    fontsize_scale_relative_factor : float
        Scaling factor to determine the font size employed in the
        scale. A value of 1.0 indicates that the font size is the
        same as the one employed in the plot labels.
    verbose : bool
        If True, display additional information.
    """

    # protections
    if arrow_length is not None:
        if not isinstance(arrow_length, Angle):
            raise TypeError("arrow_length must be an Angle or None")
    if scale_length is not None:
        if not isinstance(scale_length, Angle):
            raise TypeError("scale_length must be an Angle")
    if not isinstance(wcs2d, WCS):
        raise TypeError("wcs2d must be a WCS object.")
    if scale_location not in [0, 1, 2, 3, 4]:
        raise ValueError(f"Invalid scale_location value {scale_location}.")
    if parangle is not None:
        if not isinstance(parangle, Angle):
            raise TypeError("parangle must be an Angle")

    if verbose:
        print(wcs2d)

    pixel_scales = proj_plane_pixel_scales(wcs2d)
    if not np.isclose(pixel_scales[0], pixel_scales[1]):
        raise ValueError(f'{pixel_scales[0]=} must be equal to {pixel_scales[1]=} to draw NE arrows')
    pixel_scale = pixel_scales[0] * Unit('deg')

    # origin of the NE arrows
    x_center, y_center = wcs2d.wcs.crpix
    center_coord = wcs2d.pixel_to_world(x_center - 1, y_center - 1)
    if verbose:
        print(f'{x_center=}, {y_center=} (FITS criterion)')
        print(f'{center_coord=}')

    # current plot limits
    xmin, xmax = ax.get_xlim()
    dx = xmax - xmin
    ymin, ymax = ax.get_ylim()
    dy = ymax - ymin
    diagonal_pixels = np.sqrt(dx * dx + dy * dy)
    diagonal_deg = diagonal_pixels * pixel_scale
    # define relevant lengths if necessary
    if arrow_length is None:
        arrow_length = Angle(diagonal_pixels / 8 * pixel_scale)
    if scale_length is None:
        scale_length = arrow_length
    parangle_length = Angle(diagonal_pixels / 1.5 * pixel_scale)   # 1.5 < sqrt(2)=1.41


    # define North and East direction
    tgap = 1.10
    north = SkyCoord(ra=center_coord.ra, dec=center_coord.dec + arrow_length)
    north_text = SkyCoord(ra=center_coord.ra, dec=center_coord.dec + tgap * arrow_length)
    east = SkyCoord(ra=center_coord.ra + arrow_length/np.cos(center_coord.dec), dec=center_coord.dec)
    east_text = SkyCoord(ra=center_coord.ra + tgap * arrow_length/np.cos(center_coord.dec), dec=center_coord.dec)

    # convert celestial coordinates to pixel coordinates
    north_pix = wcs2d.world_to_pixel(north)
    north_text_pix = wcs2d.world_to_pixel(north_text)
    east_pix = wcs2d.world_to_pixel(east)
    east_text_pix = wcs2d.world_to_pixel(east_text)

    # FITS criterion
    north_pix = (north_pix[0] + 1, north_pix[1] + 1)
    north_text_pix = (north_text_pix[0] + 1, north_text_pix[1] + 1)
    east_pix = (east_pix[0] + 1, east_pix[1] + 1)
    east_text_pix = (east_text_pix[0] + 1, east_text_pix[1] + 1)
    if verbose:
        print(f'North pixel.....: {north_pix} (FITS criterion)')
        print(f'North text pixel: {north_text_pix} (FITS criterion)')
        print(f'East  pixel: {east_pix} (FITS criterion)')
        print(f'East  text pixel: {east_text_pix} (FITS criterion)')

    # North and East vectors
    if fits_criterion:
        xyoffset = 0
    else:
        xyoffset = 1
    ax.plot(np.array([x_center - xyoffset, north_pix[0] - xyoffset]),
            np.array([y_center - xyoffset, north_pix[1] - xyoffset]), '-', color=color)
    ax.text(north_text_pix[0] - xyoffset, north_text_pix[1] - xyoffset,
            'N', color=color, ha='center', va='center')
    ax.plot(np.array([x_center - xyoffset, east_pix[0] - xyoffset]),
            np.array([y_center - xyoffset, east_pix[1] - xyoffset]), '-', color=color)
    ax.text(east_text_pix[0] - xyoffset, east_text_pix[1] - xyoffset,
            'E', color=color, ha='center', va='center')

    # parallactic angle
    if parangle is not None:
        for theta in [0, 180]:
            new_north = center_coord.directional_offset_by(
                position_angle=parangle + theta * Unit('degree'),
                separation=parangle_length / 2
            )
            new_north_pix = wcs2d.world_to_pixel(new_north)
            new_north_pix = (new_north_pix[0] + 1, new_north_pix[1] + 1)
            ax.plot(np.array([x_center - xyoffset, new_north_pix[0] - xyoffset]),
                    np.array([y_center - xyoffset, new_north_pix[1] - xyoffset]),
                    linestyle='--', color=color_parangle)

    # scale
    if scale_location != 0:
        xmin, xmax = ax.get_xlim()
        dx = xmax - xmin
        ymin, ymax = ax.get_ylim()
        dy = ymax - ymin
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
        if diagonal_deg.value * 3600 < 60:
            label = f"{scale_length.to(Unit('arcsec')):.2f}"
        elif diagonal_deg.value * 60 < 60:
            label = f"{scale_length.to(Unit('arcmin')):.2f}"
        else:
            label = f"{scale_length.to(Unit('degree')):.2f}"
        ax.text((x_scale_min + x_scale_max) / 2, y_scale + 0.02*dy, label,
                fontsize=scale_font_size, ha='center', va='bottom', color=color_scale)
