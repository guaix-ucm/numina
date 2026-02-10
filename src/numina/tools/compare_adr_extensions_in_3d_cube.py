#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compare ADR corrections in a 3D FITS file"""
import argparse
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from rich_argparse import RichHelpFormatter
import sys


def plot_reference_wavelengths(ax, refewave1, refewave2, extname1, extname2):
    """Auxiliary function to plot reference wavelengths

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Instance of matplotlib.axes.Axes to plot on.
    refewave1 : float
        Reference wavelength in m, corresponding to 'extname1'
    refewave2 : float
        Reference wavelength in m, corresponding to 'extname2'
    extname1 : str
        Name of the first extension.
    extname2 : str
        Name of the second extension.
    """
    if refewave1 is not None:
        ax.axvline(refewave1, color='C0', linestyle=':',
                   label=r"$\lambda_{\rm ref1}$" + f" {extname1}")
    if refewave2 is not None:
        ax.axvline(refewave2, color='C1', linestyle=':',
                   label=r"$\lambda_{\rm ref2}$" + f" {extname2}")
    if refewave1 is not None:
        ax.text(0.0, 1.05, r"$\lambda_{\rm ref1}=$" + f"{refewave1}",
                ha='left', va='bottom', color='C0', transform=ax.transAxes)
    if refewave2 is not None:
        ax.text(1.0, 1.05, r"$\lambda_{\rm ref2}=$" + f"{refewave2}",
                ha='right', va='bottom', color='C1', transform=ax.transAxes)


def compare_adr_extensions_in_3d_cube(filename, extname1, extname2=None, suptitle=None):
    """Compare 2 ADR corrections stored in 2 extensions

    Parameters
    ----------
    filename : str, file-like or `pathlib.Path`
        FITS filename to be updated.
    extname1 : str
        First extension with ADR correction data.
    extname2 : str or None
        Second extension with ADR correction data. If not `None`,
        the differences are also displayed.
    suptitle : str or None
        Centered super title to the figure. If `None`, the 'filename'
        is employed.
    """
    refewave1 = None
    refewave2 = None
    with fits.open(filename) as hdul:
        header = hdul[0].header
        if extname1 in hdul:
            table1_adrcross = hdul[extname1].data
            refewave1 = hdul[extname1].header['REFEWAVE']
        else:
            raise ValueError(f"Extension '{extname1}' not found in FITS file '{filename}'")
        if extname2 is not None:
            if extname2 in hdul:
                table2_adrcross = hdul[extname2].data
                refewave2 = hdul[extname1].header['REFEWAVE']
            else:
                raise ValueError(f"Extension '{extname2}' not found in FITS file '{filename}'")
        else:
            table2_adrcross = None

    wcs3d = WCS(header)
    naxis1, naxis2, naxis3 = wcs3d.pixel_shape
    wave = wcs3d.spectral.pixel_to_world(np.arange(naxis3))

    delta_x1 = table1_adrcross['Delta_x']
    delta_y1 = table1_adrcross['Delta_y']
    if table2_adrcross is not None:
        delta_x2 = table2_adrcross['Delta_x']
        delta_y2 = table2_adrcross['Delta_y']
        num_figures = 4
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    else:
        delta_x2 = None
        delta_y2 = None
        num_figures = 2
        fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5))

    axarr = axarr.flatten()
    for iplot in range(num_figures):
        ax = axarr[iplot]
        if iplot == 0:
            ax.plot(wave, delta_x1, 'C0.', label=extname1)
            if table2_adrcross is not None:
                ax.plot(wave, delta_x2, 'C1-', linewidth=1, label=extname2)
            plot_reference_wavelengths(ax, refewave1, refewave2, extname1, extname2)
            ax.set_ylabel(r'$\Delta$x_ifu (pixel)')
        elif iplot == 1:
            ax.plot(wave, delta_y1, '.', label=extname1)
            if table2_adrcross is not None:
                ax.plot(wave, delta_y2, '-', linewidth=1, label=extname2)
            plot_reference_wavelengths(ax, refewave1, refewave2, extname1, extname2)
            ax.set_ylabel(r'$\Delta$y_ifu (pixel)')
        elif iplot == 2:
            ax.plot(wave, delta_x1 - delta_x2, '.')
            plot_reference_wavelengths(ax, refewave1, refewave2, extname1, extname2)
            ax.set_ylabel(r'difference in $\Delta$x_ifu (pixel)')
        else:
            ax.plot(wave, delta_y1 - delta_y2, '.')
            plot_reference_wavelengths(ax, refewave1, refewave2, extname1, extname2)
            ax.set_ylabel(r'difference in $\Delta$y_ifu (pixel)')
        ax.set_xlabel(f'Wavelength ({wave.unit})')
        ax.axhline(0, linestyle='--', color='gray')
        if iplot in [0, 1]:
            ax.legend()
    if suptitle is None:
        if extname2 is not None:
            plt.suptitle(f'file: {filename} (extensions: {extname1}, {extname2})')
        else:
            plt.suptitle(f'file: {filename} (extension: {extname1})')
    else:
        plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(description="Compare ADR extensions in 3D cube", formatter_class=RichHelpFormatter)
    parser.add_argument("filename", help="Input 3D FITS file")
    parser.add_argument("extname1", help="First extension name", type=str)
    parser.add_argument("extname2", help="Second extension name (optional)", type=str)
    parser.add_argument("--verbose", help="Display intermediate information", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(f'{arg}: {value}')

    if args.echo:
        print('[bold red]Executing:\n' + ' '.join(sys.argv) + '[/bold red]')

    for extname in [args.extname1, args.extname2]:
        if len(extname) > 8:
            raise ValueError(f"Extension '{extname}' must be less than 9 characters")

    compare_adr_extensions_in_3d_cube(
        filename=args.filename,
        extname1=args.extname1.upper(),
        extname2=args.extname2.upper()
    )


if __name__ == '__main__':
    main()
