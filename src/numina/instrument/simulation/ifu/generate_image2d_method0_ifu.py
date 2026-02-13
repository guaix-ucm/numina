#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import logging

from astropy.io import fits
from astropy.units import Quantity
import astropy.units as u
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathlib import Path

from .define_3d_wcs import wcs_to_header_using_cd_keywords


def generate_image2d_method0_ifu(
    wcs3d,
    header_keys,
    noversampling_whitelight,
    simulated_x_ifu_all,
    simulated_y_ifu_all,
    prefix_intermediate_fits,
    instname,
    subtitle,
    scene,
    plots,
    logger=None,
    output_dir=".",
):
    """Compute image2d IFU (white image), method0

    The image is calculated by performing a two-dimensional histogram
    over the X, Y coordinates of the simulated photons in the IFU,
    regardless of the wavelength assigned to each photon.

    The result is saved as a FITS file.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the WCS information.
    noversampling_whitelight : int
        Oversampling factor (integer number) to generate the method0
        white image.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    instname : str or None
        Instrument name.
    subtitle : str or None
        Plot subtitle.
    scene : str
        YAML scene file name.
    plots : bool
        If True, plot intermediate results.
    logger : `~logging.Logger` or None
        Logger for logging messages. If None, the root logger is used.
    output_dir : str or `~pathlib.Path`, optional
        Output directory for intermediate FITS files.
    """
    if logger is None:
        logger = logging.getLogger()

    # select the 2D spatial info of the 3D WCS
    wcs2d = wcs3d.sub(axes=[1, 2])

    naxis1_ifu_oversampled = Quantity(value=wcs2d.array_shape[1] * noversampling_whitelight, unit=u.pix, dtype=int)
    naxis2_ifu_oversampled = Quantity(value=wcs2d.array_shape[0] * noversampling_whitelight, unit=u.pix, dtype=int)

    bins_x_ifu_oversampled = (0.5 + np.arange(naxis1_ifu_oversampled.value + 1)) * u.pix
    bins_y_ifu_oversampled = (0.5 + np.arange(naxis2_ifu_oversampled.value + 1)) * u.pix

    crpix1_orig, crpix2_orig = wcs2d.wcs.crpix
    crpix1_oversampled = (naxis1_ifu_oversampled.value + 1) / 2
    crpix2_oversampled = (naxis2_ifu_oversampled.value + 1) / 2

    wcs2d.wcs.crpix = crpix1_oversampled, crpix2_oversampled

    if simulated_x_ifu_all is not None and simulated_y_ifu_all is not None:
        nphotons_all = len(simulated_x_ifu_all)
        if nphotons_all != len(simulated_y_ifu_all):
            raise ValueError("Length of simulated_x_ifu_all and simulated_y_ifu_all must be the same.")
    else:
        nphotons_all = 0

    if nphotons_all > 0:
        # (important: reverse X <-> Y)
        image2d_method0_ifu, xedges, yedges = np.histogram2d(
            x=(simulated_y_ifu_all.value - crpix2_orig) * noversampling_whitelight + crpix2_oversampled,
            y=(simulated_x_ifu_all.value - crpix1_orig) * noversampling_whitelight + crpix1_oversampled,
            bins=(bins_y_ifu_oversampled.value, bins_x_ifu_oversampled.value),
        )
    else:
        image2d_method0_ifu = np.zeros((naxis2_ifu_oversampled.value, naxis1_ifu_oversampled.value), dtype=int)

    wcs2d.wcs.cd /= noversampling_whitelight

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        outfile = f"{prefix_intermediate_fits}_ifu_white2D_method0_os{noversampling_whitelight:d}.fits"
        if image2d_method0_ifu.max() <= 65535:
            bitpix = 16
            hdu = fits.PrimaryHDU(image2d_method0_ifu.astype(np.uint16))
        else:
            # use float to avoid saturation problem
            logger.warning(f"The maximum value in {outfile} is greater than 65535.")
            logger.warning("Saving the image using float32 to avoid saturation.")
            bitpix = -32
            hdu = fits.PrimaryHDU(image2d_method0_ifu.astype(np.float32))
        pos0 = len(hdu.header) - 1
        hdu.header.extend(wcs_to_header_using_cd_keywords(wcs2d), update=True)
        hdu.header.update(header_keys)
        hdu.header.insert(pos0, ("COMMENT", "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ("COMMENT", "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H")
        )
        hdul = fits.HDUList([hdu])
        logger.info(f"Saving file: {outfile} (BITPIX={bitpix})")
        hdul.writeto(f"{Path(output_dir) / outfile}", overwrite=True)

    # display result
    if plots:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        img = ax.imshow(image2d_method0_ifu, origin="lower", interpolation="None")
        ax.set_xlabel("X axis (array index)  [parallel to the slices]")
        ax.set_ylabel("Y axis (array index)  [perpendicular to the slices]")
        if instname is not None:
            title = f"{instname} "
        else:
            title = ""
        title += f"IFU image, method0 (oversampling={noversampling_whitelight})"
        if subtitle is not None:
            title += f"\n{subtitle}"
        title += f"\nscene: {scene}"
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, label="Number of photons")
        plt.tight_layout()
        plt.show()
