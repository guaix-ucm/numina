#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Interactive examination of 3D data cubes with ds9.
"""

from pathlib import Path
import shutil
import subprocess
import sys

import argparse
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np

from .ctext import ctext
from .extract_2d_slice_from_3d_cube import extract_slice


def ds9cmd(cmd, pipe=False):
    """Run a command in ds9 using xpa.

    Parameters
    ----------
    cmd : str
        The command to run in ds9.
    pipe : bool, optional
        If True, the command is run in a shell. This is useful
        when 'cm' contains a pipe, which gives an error when
        trying to run the command as a list.
        If False, the command is run as a list.
        In both cases, the output is captured.

    Returns
    -------
    str
        The output of the command, if successful.

    Raises
    ------
    ValueError
        If the command fails.
    """

    if pipe:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    else:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)

    if result.stderr == '':
        return result.stdout.strip()
    else:
        raise ValueError(result.stderr)


def init_ds9_plot(fpath, wave):
    """Initialize ds9 line plot."""
    ds9cmd('xpaset -p ds9 plot line {' +
           fpath.name +
           '} {' + 
           'Value along NAXIS3 direction' + 
           '} {' +
           'Signal' +
           '} xy')
    ds9cmd('xpaset -p ds9 plot font title size 12')
    ds9cmd('xpaset -p ds9 plot font labels size 12')
    ds9cmd('xpaset -p ds9 plot legend yes')
    ds9cmd(f'xpaset -p ds9 plot axis x min {np.min(wave.value)}')
    ds9cmd(f'xpaset -p ds9 plot axis x max {np.max(wave.value)}')
    ds9cmd('xpaset -p ds9 plot axis y min 0')
    ds9cmd('xpaset -p ds9 plot axis y max 1')
    ds9cmd('xpaset -p ds9 plot legend position top')
    current_plot = ds9cmd('xpaget ds9 plot current')
    return current_plot


def update_splot(data, source_mask, continuum_mask, wave,
                 fig, ax, line_objects, firstplot, plot_render):
    """Update plot with source and continuum spectra
    
    Parameters
    ----------
    data : np.ndarray
        Array containing the 3D data cube.
    source_mask : np.ndarray
        The source mask.
    continuum_mask : np.ndarray
        The continuum mask.
    wave : np.ndarray
        Array with values along NAXIS3.
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Axes object of the plot.
    line_objects : list
        List of ax.plot() instances corresponding to the source,
        continuum and subtrated spectra.
    firstplot : bool
        If True, the next plot is the first one
    plot_render : str
        Display to display spectra: matplotlib or ds9
    """

    naxis3, naxis2, naxis1 = data.shape
    nmax = naxis2 * naxis1
    array2d_sp_source = np.zeros((nmax, naxis3))
    array2d_sp_continuum = np.zeros((nmax, naxis3))
    k_source = 0
    k_continuum = 0
    for i in range(naxis2):
        for j in range(naxis1):
            if source_mask[i, j] == 1:
                array2d_sp_source[k_source, :] = data[:, i, j]
                k_source += 1
            if continuum_mask[i, j] == 1:
                array2d_sp_continuum[k_continuum, :] = data[:, i, j]
                k_continuum += 1

    if k_source > 0:
        sp_source = np.nanmean(array2d_sp_source[:k_source, :], axis=0)
    else:
        sp_source = np.zeros(naxis3)
    sp_source_nonan = np.nan_to_num(sp_source, nan=0.0)

    if k_continuum > 0:
        sp_continuum = np.nanmean(array2d_sp_continuum[:k_continuum, :], axis=0)
    else:
        sp_continuum = np.zeros(naxis3)
    sp_continuum_nonan = np.nan_to_num(sp_continuum, nan=0.0)

    sp_subtracted = sp_source - sp_continuum
    sp_subtracted_nonan = sp_source_nonan - sp_continuum_nonan

    if plot_render in ['ds9', 'both']:
        if not firstplot:
            while ds9cmd('xpaget ds9 plot current dataset'):
                ds9cmd('xpaset -p ds9 plot delete dataset')

        for sp, sptype, npix, color in zip(
            [sp_subtracted_nonan, sp_source_nonan, sp_continuum_nonan],
            ['subtracted', 'source', 'continuum'],
            [None, k_source, k_continuum],
            ['blue', 'orange', 'green']
        ):
            np.savetxt(
                f'tmp_spectrum_{sptype}.dat',
                np.column_stack((wave.value, sp)),
                fmt='%e'
            )
            cnpix = f' ({npix})' if npix is not None else ''
            ds9cmd(f'cat tmp_spectrum_{sptype}.dat | xpaset ds9 plot data xy', pipe=True)
            ds9cmd(f'xpaset -p ds9 plot line color {color}')
            ds9cmd('xpaset -p ds9 plot name {' +
                f'{sptype}' + cnpix +
                '}')

    if plot_render in ['matplotlib', 'both']:
        # recompute global Y-axis limits
        sp_concatenate = np.concatenate((sp_source, sp_continuum, sp_subtracted))
        ymin = np.nanmin(sp_concatenate)
        ymax = np.nanmax(sp_concatenate)
        dy = ymax - ymin
        if dy == 0:
            dy = 0.1
        ymin -= dy / 20
        ymax += dy / 20
        ax.set_ylim(ymin, ymax)

        line_source, line_continuum, line_subtracted = line_objects
        line_source.set_ydata(sp_source)
        line_source.set_label(f'source ({k_source})')
        line_continuum.set_ydata(sp_continuum)
        line_continuum.set_label(f'continuum ({k_continuum})')
        line_subtracted.set_ydata(sp_subtracted)
        ax.draw_artist(line_source)
        ax.draw_artist(line_continuum)
        ax.draw_artist(line_subtracted)
        # The legend in Matplotlib is a separate artist that doesn't automatically
        # udpate when you change plot elements unless you explicitly update or redraw it.
        # For that reason it is necessary to remove the old legend and redraw the new one.
        old_legend = ax.get_legend()
        if old_legend:
            old_legend.remove()
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        fig.canvas.draw()
        fig.canvas.flush_events()


def update_ds9regions(data, source_mask, continuum_mask, tmp_mask, wave,
                      fig, ax, line_objects, firstplot, plot_render):
    """Update ds9 regions interactively using the source and continuum masks.

    Parameters
    ----------
    data : np.array
        Array containing the 3D data cube.
    source_mask : np.ndarray
        The source mask to update.
    continuum_mask : np.ndarray
        The continuum mask to update.
    tmp_mask : np.ndarray or None
        A temporary mask to display starting point of a new
        rectangular region.
    wave : np.ndarray
        Array with values along NAXIS3.
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The Axes object of the plot.
    line_objects: list
        List of ax.plot() instances corresponding to the source,
        continuum and subtrated spectra.
    firstplot = bool
        If True, the next plot is the first one.
    plot_render : str
        Display to display spectra: matplotlib or ds9
    """
    # update file with regions
    lines = [
        '# Region file format: DS9 version 4.1',
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman"' +
        'select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
        'physical'
    ]
    naxis2, naxis1 = source_mask.shape
    for i in range(naxis2):
        for j in range(naxis1):
            if source_mask[i, j]:
                lines.append(f'box({j+1},{i+1},1,1,0) # fill=1 color=red source=1')
            if continuum_mask[i, j]:
                lines.append(f'box({j+1},{i+1},1,1,0) # fill=1 color=green source=0')
            if tmp_mask is not None:
                if tmp_mask[i, j] != 0:
                    lines.append(f'box({j+1},{i+1},1,1,0) # fill=1 color=cyan')
    with open('tmp_regions_ds9.reg', 'wt', encoding='ascii') as f:
        for line in lines:
            f.write(line + '\n')
    try:
        ds9cmd('xpaset -p ds9 region delete')
        ds9cmd('xpaset -p ds9 region load tmp_regions_ds9.reg')
    except ValueError as exc:
        print(f'WARNING: {exc}')
    # Update splot
    update_splot(
        data=data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        wave=wave,
        fig=fig,
        ax=ax,
        line_objects=line_objects,
        firstplot=firstplot,
        plot_render=plot_render
    )


def display_help_menu(plot_render):
    """Display help text when selecting pixels.
    
    Parameters
    ----------
    plot_render: str
        Display to display spectra: matplotlib or ds9
    """
    print("Click on the ds9 window to select pixels:")
    print("  - Press 's' to select a single source pixel")
    print("  - Press 'c' to select a single continuum pixel")
    print("  - Press 'x' to remove a single pixel from any mask")
    print("  - Press 'r' to reset both masks")
    print("  - Press 'a' to start selecting a rectangular region")
    print("    (then press 's' or 'c' in the opposite corner to define the mask type)")
    if plot_render in ['matplotlib', 'both']:
        print("  - Press 'p' to pause pixel selection and allow matplotlib interaction")
    print("  - Press 'q' to quit (stop pixel selection)")
    print("  - Press 'h' to display this help")


def update_masks(filename, data, source_mask, continuum_mask, wave, verbose, plot_render):
    """Update source and continuum masks interactively using ds9.

    Parameters
    ----------
    filename : str
        File name retrieved from ds9.
    data : np.ndarray
        Array containing the 3D data cube.
    source_mask : np.ndarray
        The source mask to update.
    continuum_mask : np.ndarray
        The continuum mask to update.
    wave : astropy.units.Quantity
        The wavelength array corresponding to the spectral axis.
    verbose : bool
        If True, print verbose output.
    plot_render : str
        Display to display spectra: matplotlib or ds9

    """

    if source_mask.shape != continuum_mask.shape:
        print(f'{source_mask.shape=}')
        print(f'{continuum_mask.shape=}')
        raise ValueError('Incompatible mask shapes')
    naxis2, naxis1 = source_mask.shape
    tmp_mask = None

    if plot_render in ['matplotlib', 'both']:
        plt.ion()  # enable interactive mode
        fig, ax = plt.subplots()
        sp_source = np.zeros(len(wave))
        sp_continuum = np.zeros(len(wave))
        sp_subtracted = np.zeros(len(wave))
        (line_subtracted,) = ax.plot(wave, sp_subtracted, 'C0-', label='subtracted', zorder=3)
        (line_source,) = ax.plot(wave, sp_source, 'C1-', label='source', zorder=2)
        (line_continuum,) = ax.plot(wave, sp_continuum, 'C2-', label='continuum', zorder=1)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        line_objects = [line_source, line_continuum, line_subtracted]
        ax.set_title(f'{filename}')
        ax.set_xlabel('Value along NAXIS3 direction')
        ax.set_ylabel('Signal')
    else:
        fig = None
        ax = None
        line_objects = [None, None, None]

    update_ds9regions(
        data=data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        tmp_mask=tmp_mask,
        wave=wave,
        fig=fig,
        ax=ax,
        line_objects=line_objects,
        firstplot=True,
        plot_render=plot_render
    )

    display_help_menu(plot_render)

    loop = True
    last_key_pos = [None, None, None]
    ix1, ix2, iy1, iy2 = None, None, None, None  # avoid PyCharm warning
    while loop:
        try:
            key, x, y = ds9cmd('xpaget ds9 iexam key coordinate image').split()
        except ValueError as exc:
            if verbose:
                print(f'WARNING: {exc}')
        if key in ['s', 'c', 'r', 'a', 'x']:
            x = str(round(float(x)))
            y = str(round(float(y)))
            if verbose:
                print(f'key: {key}: selecting pixel {x=}, {y=}')
            iy = int(y) - 1
            ix = int(x) - 1
            if key == 'a':
                last_key_pos = [key, ix, iy]
                tmp_mask = np.zeros(shape=(naxis2, naxis1), dtype=np.uint8)
                tmp_mask[iy, ix] = 1
            elif key in ['s', 'c'] and last_key_pos[0] is not None:
                ix1 = min(ix, last_key_pos[1])
                ix2 = max(ix, last_key_pos[1])
                iy1 = min(iy, last_key_pos[2])
                iy2 = max(iy, last_key_pos[2])
                last_key_pos = [None, None, None]
            elif key in ['s', 'c', 'x']:
                ix1 = ix
                ix2 = ix
                iy1 = iy
                iy2 = iy
                #last_key_pos = [key, ix, iy]
            if key in ['s', 'c', 'x']:
                for iy in range(iy1, iy2+1):
                    for ix in range(ix1, ix2+1):
                        if key == 's':
                            if continuum_mask[iy, ix] == 0:
                                source_mask[iy, ix] = 1
                            else:
                                continuum_mask[iy, ix] = 0
                                source_mask[iy, ix] = 1
                        elif key == 'c':
                            if source_mask[iy, ix] == 0:
                                continuum_mask[iy, ix] = 1
                            else:
                                source_mask[iy, ix] = 0
                                continuum_mask[iy, ix] = 1
                        elif key == 'x':
                            source_mask[iy, ix] = 0
                            continuum_mask[iy, ix] = 0
                        else:
                            raise ValueError('Unexpected error')
            elif key == 'r':
                for i in range(naxis2):
                    for j in range(naxis1):
                        source_mask[i, j] = 0
                        continuum_mask[i, j] = 0
            elif key == 'a':
                pass
            else:
                raise ValueError('Unexpected error')
            # update file with regions
            update_ds9regions(
                data=data,
                source_mask=source_mask,
                continuum_mask=continuum_mask,
                tmp_mask=tmp_mask,
                wave=wave,
                fig=fig,
                ax=ax,
                line_objects=line_objects,
                firstplot=False,
                plot_render=plot_render
            )
            if key == 'a':
                tmp_mask = None
        elif key == 'p':
            if plot_render in ['matplotlib', 'both']:
                input('Press RETURN to continue with pixel selection...')
        elif key == 'h':
            display_help_menu(plot_render)
        elif key == 'q':
            cquit = input('Do you want to quit? (y/[n]) ')
            if cquit.lower() in ['y', 'yes']:
                loop = False
                if verbose:
                    print('Selection of pixels finished!')

    if plot_render in ['matplotlib', 'both']:
        # keep the splot open after updates
        plt.ioff()
        print("Press 'q' to close matplotlib window and stop the program")
        plt.show(block=True)


def main(args=None):
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Interactive examination of 3D data cubes with ds9."
    )

    parser.add_argument("datacube",
                        help="Input 3D FITS data cube",
                        type=str)
    parser.add_argument("--i1",
                        help='First pixel along NAXIS3 (default 1)',
                        type=int, default=1)
    parser.add_argument("--i2",
                        help='Last pixel along NAXIS3 (default NAXIS3)',
                        type=str)
    parser.add_argument("--ds9exec",
                        help="Command line to launch ds9 (default 'ds9')",
                        type=str, default='ds9')
    parser.add_argument("--plot_render",
                        help="Display to display spectra (default=matplotlib)",
                        choices=['matplotlib', 'ds9', 'both'],
                        default='matplotlib')
    parser.add_argument("--input_masks",
                        help="Path to a FITS file with source and continuum masks",
                        type=str)
    parser.add_argument("--output_masks",
                        help="Path to the output FITS file with source and continuum masks",
                        type=str)
    parser.add_argument("--verbose",
                        help="Display intermediate information",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    file_datacube = args.datacube
    ds9exec = args.ds9exec
    verbose = args.verbose
    plot_render = args.plot_render

    # Check XPA is installed
    list_required_executables = ['xpaget', 'xpaset']
    for required_executable in list_required_executables:
        execfile = shutil.which(required_executable)
        if execfile is None:
            raise SystemExit(f'The program {required_executable} is not available')
        else:
            if verbose:
                print(f'Required program {execfile} found!')

    # Check if ds9 is already running
    print('Checking if a previous ds9 instance is running...')
    try:
        filename = ds9cmd('xpaget ds9 file')
    except Exception as exc:
        filename = ''
        if verbose:
            print(f'{exc=}')
            print('No previous ds9 instance found. OK!')
    if len(filename) > 0:
        print('A previous instance of ds9 is already running.\n' +
              f'Filename: {filename}\n' +
              'Please close it before using this program')
        raise SystemExit()

    # Get header and data of the FITS file
    fpath = Path(file_datacube)
    header = fits.getheader(fpath)
    wcs = WCS(header)
    if verbose:
        print(f'WCS: {wcs}')

    data = fits.getdata(fpath)
    if len(data.shape) != 3:
        raise ValueError(f'Expected a 3D cube, but got {data.shape}')

    naxis3, naxis2, naxis1 = data.shape
    if verbose:
        print(f'{naxis1=}')
        print(f'{naxis2=}')
        print(f'{naxis3=}')

    i1 = args.i1
    if i1 < 1 or i1 > naxis3:
        raise ValueError(f'Invalid first pixel={i1} along NAXIS3={naxis3}')
    if args.i2 is None:
        i2 = naxis3
    else:
        i2 = int(args.i2)
        if i2 < i1 or i2 > naxis3:
            raise ValueError(f'Invalid last pixel={i2} along NAXIS3={naxis3}')

    # Collapse the data cube along NAXIS3
    if verbose:
        print('Collapsing 3D cubes along NAXIS3... ', end='')
    extract_slice(
        input=file_datacube,
        axis=3,
        i1=i1,
        i2=i2,
        method='sum',
        wavecal='none',
        transpose=False,
        vmin=None,
        vmax=None,
        noplot=True,
        output='tmp_collapsed_3D.fits'
    )
    print('OK!')

    # Launch ds9
    cmd = f'{ds9exec.split()[0]} tmp_collapsed_3D.fits {' '.join(ds9exec.split()[1:])} &'
    if verbose:
        print('Executing:')
        print(cmd)
    # Note: use shell=True below to make the ds9 alias in the system available
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=True)
    if verbose:
        print(f'{result.stderr=}')
        print(f'{result.stdout=}')
    input('Press RETURN after ds9 has properly started...')
    try:
        filename = ds9cmd('xpaget ds9 file')
    except Exception as exc:
        raise SystemExit('Fatal error: ds9 is not running') from exc
    if verbose:
        print(f'ds9 working with file: {filename}')

    # Generate array in the spectral direction
    wcs1d_spectral = wcs.spectral
    wave = wcs1d_spectral.pixel_to_world(np.arange(naxis3))
    if verbose:
        print(f'Minimum value along NAXIS3: {wave.min()}')
        print(f'Maximum value along NAXIS3: {wave.max()}')

    # Read source and continuum masks or create them
    if args.input_masks:
        source_mask = fits.getdata(args.input_masks, extname='SOURMASK')
        if source_mask.shape != (naxis2, naxis1):
            raise ValueError(f'{source_mask.shape=} does not match {(naxis2, naxis1)=}')
        if source_mask.dtype != np.uint8:
            raise ValueError(f'Source mask dtype {source_mask.dtype} is not uint8')
        continuum_mask = fits.getdata(args.input_masks, extname='CONTMASK')
        if continuum_mask.shape != (naxis2, naxis1):
            raise ValueError(f'{continuum_mask.shape=} does not match {(naxis2, naxis1)=}')
        if continuum_mask.dtype != np.uint8:
            raise ValueError(f'Continuum mask dtype {continuum_mask.dtype} is not uint8')
    else:
        source_mask = np.zeros((naxis2, naxis1), dtype=np.uint8)
        continuum_mask = np.zeros((naxis2, naxis1), dtype=np.uint8)

    if plot_render in ['ds9', 'both']:
        current_plot = init_ds9_plot(fpath=fpath, wave=wave)
        if verbose:
            print(f'Opening ds9 plot: {current_plot}')

    update_masks(
        filename=file_datacube,
        data=data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        wave=wave,
        verbose=verbose,
        plot_render=plot_render
    )

    # Save the masks to a FITS file
    if args.output_masks:
        output_masks = Path(args.output_masks)
    else:
        output_masks = Path('tmp_masks.fits')
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(source_mask, name='SOURMASK')
    hdu2 = fits.ImageHDU(continuum_mask, name='CONTMASK')
    hdul = fits.HDUList([hdu0, hdu1, hdu2])
    hdul.writeto(output_masks, overwrite=True)
    if verbose:
        print(f'Masks saved to {output_masks}')

    print('Remember to close the running session of ds9 before re-executing this program!')


if __name__ == "__main__":
    main()
