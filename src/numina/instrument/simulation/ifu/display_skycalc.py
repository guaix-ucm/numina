#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import matplotlib.pyplot as plt

from .raise_valueerror import raise_ValueError


def display_skycalc(faux_skycalc):
    """
    Display sky radiance and transmission.

    Data generated with the SKYCALC Sky Model Calculator tool
    provided by ESO.
    See https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html

    Parameters
    ----------
    faux_skycalc : str
        FITS file name with SKYCALC predictions.
    """

    with fits.open(faux_skycalc) as hdul:
        skycalc_header = hdul[1].header
        skycalc_table = hdul[1].data

    if skycalc_header["TTYPE1"] != "lam":
        raise_ValueError(f"Unexpected TTYPE1: {skycalc_header['TTYPE1']}")
    if skycalc_header["TTYPE2"] != "flux":
        raise_ValueError(f"Unexpected TTYPE2: {skycalc_header['TTYPE2']}")

    wave = skycalc_table["lam"]
    flux = skycalc_table["flux"]
    trans = skycalc_table["trans"]

    # plot radiance
    fig, ax = plt.subplots()
    ax.plot(wave, flux, "-", linewidth=1)
    cwave_unit = skycalc_header["TUNIT1"]
    cflux_unit = skycalc_header["TUNIT2"]
    ax.set_xlabel(f"Vacuum Wavelength ({cwave_unit})")
    ax.set_ylabel(f"{cflux_unit}")
    ax.set_title("SKYCALC prediction")
    plt.tight_layout()
    plt.show()

    # plot transmission
    fig, ax = plt.subplots()
    ax.plot(wave, trans, "-", linewidth=1)
    ax.set_xlabel(f"Vacuum Wavelength ({cwave_unit})")
    ax.set_ylabel("Transmission fraction")
    ax.set_title("SKYCALC prediction")
    plt.tight_layout()
    plt.show()
