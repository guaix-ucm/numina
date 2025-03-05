#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
from astropy.coordinates import SkyCoord, Angle
from astropy.units import Unit
from astropy.wcs import WCS


def plot_north_east_arrows(ax, wcs2d, arrow_length_arcsec, fits_criterion=False, color='grey', verbose=False):
    """Display North & East arrow using WCS information.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Instance of `matplotlib.axes.Axes`.
    wcs2d : astropy.wcs.WCS
        Instance of a 2D `astropy.wcs.WCS`.
    arrow_length_arcsec : float
        Arrow length in arcsecond.
    fits_criterion : bool
        If True, assume FITS criteria to compute the central pixel.
    color : str
        Color to use for plotting.
    verbose : bool
        If True, display additional information.
    """

    if not isinstance(wcs2d, WCS):
        raise TypeError("wcs2d must be a WCS object.")

    if verbose:
        print(wcs2d)
    x_center, y_center = wcs2d.wcs.crpix
    center_coord = wcs2d.pixel_to_world(x_center - 1, y_center - 1)

    # define North and East direction
    north = SkyCoord(ra=center_coord.ra, dec=center_coord.dec + Angle(arrow_length_arcsec, Unit('arcsec')))
    east = SkyCoord(ra=center_coord.ra + Angle(arrow_length_arcsec, Unit('arcsec')), dec=center_coord.dec)

    # convert celestial coordinates to pixel coordinates
    north_pix = wcs2d.world_to_pixel(north)
    east_pix = wcs2d.world_to_pixel(east)
    if fits_criterion:
        north_pix = (north_pix[0] + 1, north_pix[1] + 1)
        east_pix = (east_pix[0] + 1, east_pix[1] + 1)

    #ax.arrow(x_center, y_center, north_pix[0] - x_center, north_pix[1] - y_center, shape='right')
    ax.annotate('N', xy=(x_center, y_center),
                xytext=north_pix, arrowprops=dict(color=color, arrowstyle='<-'), color=color)
    ax.annotate('E', xy=(x_center, y_center),
                xytext=east_pix, arrowprops=dict(color=color, arrowstyle='<-'), color=color)
