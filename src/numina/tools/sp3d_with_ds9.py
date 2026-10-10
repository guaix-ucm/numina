#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Interactive examination of 3D data cubes with ds9."""

import os
from pathlib import Path
import shutil
import subprocess
import warnings

import argparse
from astropy.io import fits
import astropy.units as u
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich_argparse import RichHelpFormatter
import sys

from .extract_2d_slice_from_3d_cube import extract_slice
from .hdul_utils import get_hdu_from_hdul, get_wcs_from_hdu
from .initialize_script_with_args import NuminaScriptDefinition

# Store the original quit keys to restore them later
QUIT_KEYS_MATPLOTLIB_ORIG = list(mpl.rcParams["keymap.quit"])
mpl.rcParams["keymap.quit"] = []  # Disable the default quit keys in Matplotlib


def ds9cmd(cmd: str, pipe: bool = False) -> str:
    """Run a shell command, typically an XPA command to control ds9.

    Parameters
    ----------
    cmd : str
        The full command to run, including the XPA executable
        (e.g. 'xpaset -p ds9 ...' or 'xpaget ds9 ...').
    pipe : bool, optional
        If True, the command is run in a shell. This is useful
        when 'cmd' contains a pipe, which gives an error when
        trying to run the command as a list.
        If False, the command is split on whitespace with
        `str.split()` (quotes and braces are not taken into
        account) and run as a list, without a shell.
        In both cases, the output is captured.

    Returns
    -------
    str
        The standard output of the command, with leading and
        trailing whitespace removed.

    Raises
    ------
    ValueError
        If the command writes anything to standard error. The
        return code of the command is not checked.
    FileNotFoundError
        If `pipe` is False and the executable is not found.
    """

    if pipe:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    else:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, check=False)

    if result.stderr == "":
        return result.stdout.strip()
    else:
        raise ValueError(result.stderr)


def init_ds9_plot(fpath: Path, wave: u.Quantity) -> str:
    """Initialize a line plot in ds9 to display spectra.

    A new ds9 plot is created through XPA, with the file name as
    title and generic axis labels. The font sizes and the legend
    are configured, and the X-axis limits are set to the range of
    `wave`. The Y-axis limits are initially set to [0, 1].

    Parameters
    ----------
    fpath : pathlib.Path
        Path to the 3D data cube. Only its name (without the
        directory) is used, as the plot title.
    wave : astropy.units.Quantity
        Values along the NAXIS3 (spectral) direction. Only the
        numerical values (`wave.value`) are used, to set the
        X-axis limits, so the unit is not displayed in the plot.

    Returns
    -------
    str
        Name of the current ds9 plot, as returned by
        'xpaget ds9 plot current'.

    Raises
    ------
    ValueError
        If any of the XPA commands writes to standard error
        (see `ds9cmd`).
    """
    ds9cmd(
        "xpaset -p ds9 plot line {" + fpath.name + "} {" + "Value along NAXIS3 direction" + "} {" + "Signal" + "} xy"
    )
    ds9cmd("xpaset -p ds9 plot font title size 12")
    ds9cmd("xpaset -p ds9 plot font labels size 12")
    ds9cmd("xpaset -p ds9 plot legend yes")
    ds9cmd(f"xpaset -p ds9 plot axis x min {np.min(wave.value)}")
    ds9cmd(f"xpaset -p ds9 plot axis x max {np.max(wave.value)}")
    ds9cmd("xpaset -p ds9 plot axis y min 0")
    ds9cmd("xpaset -p ds9 plot axis y max 1")
    ds9cmd("xpaset -p ds9 plot legend position top")
    current_plot = ds9cmd("xpaget ds9 plot current")
    return current_plot


def update_splot(
    data,
    footprint_data,
    source_mask,
    continuum_mask,
    wave,
    fig,
    ax,
    ax_footprint,
    line_objects,
    filename,
    firstplot,
    plot_render,
):
    """Update plot with source and continuum spectra.

    Parameters
    ----------
    data : np.ndarray
        Array containing the 3D data cube.
    footprint_data : nd.ndarray
        The footprint data to display, if available.
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
    ax_footprint : matplotlib.axes.Axes or None
        The Axes object of the footprint plot, if available.
    line_objects : list
        List of ax.plot() instances corresponding to the source,
        continuum and subtrated spectra.
    filename : str
        File name.
    firstplot : bool
        If True, the next plot is the first one
    plot_render : str
        Display to display spectra: matplotlib or ds9
    """

    naxis3, naxis2, naxis1 = data.shape
    nmax = naxis2 * naxis1
    array2d_sp_source = np.zeros((nmax, naxis3))
    array2d_sp_continuum = np.zeros((nmax, naxis3))
    array2d_sp_footprint = np.zeros((nmax, naxis3)) if footprint_data is not None else None
    k_source = 0
    k_continuum = 0
    for i in range(naxis2):
        for j in range(naxis1):
            if source_mask[i, j] == 1:
                if not np.all(np.isnan(data[:, i, j])):
                    array2d_sp_source[k_source, :] = data[:, i, j]
                    if footprint_data is not None:
                        array2d_sp_footprint[k_source, :] = footprint_data[:, i, j]
                    k_source += 1
            if continuum_mask[i, j] == 1:
                if not np.all(np.isnan(data[:, i, j])):
                    array2d_sp_continuum[k_continuum, :] = data[:, i, j]
                    k_continuum += 1

    # Compute the mean spectra for source and continuum, handling NaN values
    if k_source > 0:
        # Use warnings.catch_warnings() to suppress the "Mean of empty slice" warning
        # when computing the mean of an empty array
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            sp_source = np.nanmean(array2d_sp_source[:k_source, :], axis=0)
            if footprint_data is not None:
                sp_footprint = np.nanmean(array2d_sp_footprint[:k_source, :], axis=0)
            else:
                sp_footprint = None
    else:
        sp_source = np.zeros(naxis3)
        if footprint_data is not None:
            sp_footprint = np.zeros(naxis3)
        else:
            sp_footprint = None
    sp_source_nonan = np.nan_to_num(sp_source, nan=0.0)

    if k_continuum > 0:
        # Use warnings.catch_warnings() to suppress the "Mean of empty slice" warning
        # when computing the mean of an empty array
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            sp_continuum = np.nanmean(array2d_sp_continuum[:k_continuum, :], axis=0)
    else:
        sp_continuum = np.zeros(naxis3)
    sp_continuum_nonan = np.nan_to_num(sp_continuum, nan=0.0)

    sp_subtracted = sp_source - sp_continuum
    sp_subtracted_nonan = np.nan_to_num(sp_subtracted, nan=0.0)

    if plot_render in ["ds9", "both"]:
        if not firstplot:
            while ds9cmd("xpaget ds9 plot current dataset"):
                ds9cmd("xpaset -p ds9 plot delete dataset")

        for sp, sptype, npix, color in zip(
            [sp_subtracted_nonan, sp_source_nonan, sp_continuum_nonan],
            ["subtracted", "source", "continuum"],
            [None, k_source, k_continuum],
            ["blue", "orange", "green"],
        ):
            np.savetxt(f"tmp_spectrum_{sptype}.dat", np.column_stack((wave.value, sp)), fmt="%e")
            cnpix = f" ({npix})" if npix is not None else ""
            ds9cmd(f"cat tmp_spectrum_{sptype}.dat | xpaset ds9 plot data xy", pipe=True)
            ds9cmd(f"xpaset -p ds9 plot line color {color}")
            ds9cmd("xpaset -p ds9 plot name {" + f"{sptype}" + cnpix + "}")

    if plot_render in ["matplotlib", "both"]:
        # recompute global Y-axis limits
        sp_concatenate = np.concatenate((sp_source_nonan, sp_continuum_nonan, sp_subtracted_nonan))
        ymin = np.min(sp_concatenate)
        ymax = np.max(sp_concatenate)
        dy = ymax - ymin
        if dy == 0:
            dy = 0.1
        ymin -= dy / 20
        ymax += dy / 20
        ax.set_ylim(ymin, ymax)

        line_source, line_continuum, line_subtracted, line_footprint = line_objects
        line_source.set_ydata(sp_source_nonan)
        line_source.set_label(f"source ({k_source})")
        line_continuum.set_ydata(sp_continuum_nonan)
        line_continuum.set_label(f"continuum ({k_continuum})")
        line_subtracted.set_ydata(sp_subtracted_nonan)
        ax.draw_artist(line_source)
        ax.draw_artist(line_continuum)
        ax.draw_artist(line_subtracted)
        if sp_footprint is not None:
            sp_footprint_nonan = np.nan_to_num(sp_footprint, nan=0.0)
            line_footprint.set_ydata(sp_footprint_nonan)
            line_footprint.set_label(f"footprint ({k_source})")
            ax_footprint.draw_artist(line_footprint)

        # The legend in Matplotlib is a separate artist that doesn't automatically
        # udpate when you change plot elements unless you explicitly update or redraw it.
        # For that reason it is necessary to remove the old legend and redraw the new one.
        old_legend = ax.get_legend()
        if old_legend:
            old_legend.remove()
        handles1, labels1 = ax.get_legend_handles_labels()
        if sp_footprint is not None:
            handles2, labels2 = ax_footprint.get_legend_handles_labels()
            for handle, label in zip(handles2, labels2):
                handles1.append(handle)
                labels1.append(label)
            ncol_legend = 4
        else:
            ncol_legend = 3
        ax.legend(
            handles=handles1,
            labels=labels1,
            title=filename,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.35),
            ncol=ncol_legend,
        )
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()


def update_ds9regions(
    data,
    footprint_data,
    source_mask,
    continuum_mask,
    tmp_mask,
    wave,
    fig,
    ax,
    ax_footprint,
    line_objects,
    filename,
    firstplot,
    plot_render,
):
    """Update ds9 regions interactively using the source and continuum masks.

    Parameters
    ----------
    data : np.array
        Array containing the 3D data cube.
    footprint_data : np.ndarray or None
        The footprint data to display, if available.
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
    ax_footprint : matplotlib.axes.Axes or None
        The Axes object of the footprint plot, if available.
    line_objects: list
        List of ax.plot() instances corresponding to the source,
        continuum and subtrated spectra.
    filename : str
        File name.
    firstplot = bool
        If True, the next plot is the first one.
    plot_render : str
        Display to display spectra: matplotlib or ds9
    """
    logger = logging.getLogger(__name__)

    # update file with regions
    lines = [
        "# Region file format: DS9 version 4.1",
        'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman"'
        + "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1",
        "physical",
    ]
    naxis2, naxis1 = source_mask.shape
    for i in range(naxis2):
        for j in range(naxis1):
            if source_mask[i, j]:
                lines.append(f"box({j+1},{i+1},1,1,0) # fill=1 color=red source=1")
            if continuum_mask[i, j]:
                lines.append(f"box({j+1},{i+1},1,1,0) # fill=1 color=green source=0")
            if tmp_mask is not None:
                if tmp_mask[i, j] != 0:
                    lines.append(f"box({j+1},{i+1},1,1,0) # fill=1 color=cyan")
    with open("tmp_regions_ds9.reg", "wt", encoding="ascii") as f:
        for line in lines:
            f.write(line + "\n")
    try:
        ds9cmd("xpaset -p ds9 region delete")
        ds9cmd("xpaset -p ds9 region load tmp_regions_ds9.reg")
    except ValueError as exc:
        logger.warning(f"WARNING: {exc}")
    # Update splot
    update_splot(
        data=data,
        footprint_data=footprint_data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        wave=wave,
        fig=fig,
        ax=ax,
        ax_footprint=ax_footprint,
        line_objects=line_objects,
        filename=filename,
        firstplot=firstplot,
        plot_render=plot_render,
    )


def display_help_menu(plot_render):
    """Display help text when selecting pixels.

    Parameters
    ----------
    plot_render: str
        Display to display spectra: matplotlib or ds9
    """
    logger = logging.getLogger(__name__)
    logger.info("Click on the ds9 window to select pixels:")
    logger.info("  - Press 's' to select a single source pixel")
    logger.info("  - Press 'c' to select a single continuum pixel")
    logger.info("  - Press 'e' to erase a single pixel from any mask")
    logger.info("  - Press 'r' to reset both masks")
    logger.info("  - Press 'a' to start selecting a rectangular region")
    logger.info("    (then press 's' or 'c' in the opposite corner to define the mask type)")
    if plot_render in ["matplotlib", "both"]:
        logger.info("  - Press 'x' to exit from pixel selection and allow matplotlib interaction")
    logger.info("  - Press 'q' to quit (stop pixel selection)")
    logger.info("  - Press 'h' to display this help")


def update_masks(filename, data, footprint_data, source_mask, continuum_mask, wave, plot_render):
    """Update source and continuum masks interactively using ds9.

    Parameters
    ----------
    filename : str
        File name retrieved from ds9.
    data : np.ndarray
        Array containing the 3D data cube.
    footprint_data : np.ndarray or None
        The footprint data to display, if available.
    source_mask : np.ndarray
        The source mask to update.
    continuum_mask : np.ndarray
        The continuum mask to update.
    wave : astropy.units.Quantity
        The wavelength array corresponding to the spectral axis.
    plot_render : str
        Display to display spectra: matplotlib or ds9

    """
    logger = logging.getLogger(__name__)

    if source_mask.shape != continuum_mask.shape:
        logger.error(f"[red]{source_mask.shape=}[/red]")
        logger.error(f"[red]{continuum_mask.shape=}[/red]")
        raise ValueError("Incompatible mask shapes")
    naxis2, naxis1 = source_mask.shape
    tmp_mask = None

    if plot_render in ["matplotlib", "both"]:
        plt.ion()  # enable interactive mode
        fig, ax = plt.subplots()

        def on_key(event):
            if event.key == "q":
                logger.info("\nKey 'q' has been disabled in Matplotlib to avoid closing the window")
                logger.info("Press RETURN in console to continue with pixel selection in ds9 window...")

        fig.canvas.mpl_connect("key_press_event", on_key)
        sp_source = np.zeros(len(wave))
        sp_continuum = np.zeros(len(wave))
        sp_subtracted = np.zeros(len(wave))
        sp_footprint = np.zeros(len(wave)) if footprint_data is not None else None
        (line_subtracted,) = ax.plot(wave, sp_subtracted, "C0-", label="subtracted", zorder=3)
        (line_source,) = ax.plot(wave, sp_source, "C1-", label="source", zorder=2)
        (line_continuum,) = ax.plot(wave, sp_continuum, "C2-", label="continuum", zorder=1)
        npix_extra = 2
        deltawave_min = wave[1] - wave[0]
        deltawave_max = wave[-1] - wave[-2]
        ax.set_xlim(wave[0].value - npix_extra * deltawave_min.value, wave[-1].value + npix_extra * deltawave_max.value)
        line_objects = [line_source, line_continuum, line_subtracted]
        ax.set_xlabel("Value along NAXIS3 direction")
        ax.set_ylabel("Mean signal")
        # display upper horizontal scale in pixel units along NAXIS3
        ax2 = ax.twiny()
        ax2.set_xlabel("Pixel along NAXIS3 direction")
        ax2.set_xlim(1 - npix_extra, len(wave) + npix_extra)
        plt.tight_layout()
        # display the footprint spectrum if available
        if sp_footprint is not None:
            ax_footprint = ax.twinx()
            ax_footprint.spines["right"].set_color("gray")
            ax_footprint.tick_params(axis="y", colors="gray")
            ax_footprint.yaxis.label.set_color("gray")
            ax.spines["right"].set_visible(False)
            ymin_footprint = np.nanmin(footprint_data)
            ymax_footprint = np.nanmax(footprint_data)
            dy_footprint = ymax_footprint - ymin_footprint
            if dy_footprint == 0:
                dy_footprint = 0.1
            ymin_footprint -= dy_footprint / 20
            ymax_footprint += dy_footprint / 20

            # fix the y-limits of the footprint spectrum to avoid automatic rescaling
            # when the user zooms in the main plot
            def fix_ylim_footprint(a):
                a.set_ylim(ymin_footprint, ymax_footprint, emit=False)

            ax_footprint.callbacks.connect("ylim_changed", fix_ylim_footprint)
            (line_footprint,) = ax_footprint.plot(
                wave, sp_footprint, "-", color="gray", alpha=0.5, label="footprint", zorder=0
            )
            # add label to the right y-axis for the footprint spectrum
            ax_footprint.set_ylabel("Mean footprint", color="gray")
            # add entry to the ax.legend for the footprint spectrum
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax_footprint.get_legend_handles_labels()
            for handle, label in zip(handles2, labels2):
                handles1.append(handle)
                labels1.append(label)
            ax.legend(
                handles=handles1, labels=labels1, title=filename, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=4
            )
        else:
            ax_footprint = None
            line_footprint = None
            ax.legend(title=filename, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3)
        line_objects.append(line_footprint)
    else:
        fig = None
        ax = None
        ax_footprint = None
        line_objects = [None, None, None]

    update_ds9regions(
        data=data,
        footprint_data=footprint_data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        tmp_mask=tmp_mask,
        wave=wave,
        fig=fig,
        ax=ax,
        ax_footprint=ax_footprint,
        line_objects=line_objects,
        filename=filename,
        firstplot=True,
        plot_render=plot_render,
    )

    display_help_menu(plot_render)

    loop = True
    last_key_pos = [None, None, None]
    ix1, ix2, iy1, iy2 = None, None, None, None  # avoid PyCharm warning
    while loop:
        try:
            key, x, y = ds9cmd("xpaget ds9 iexam key coordinate image").split()
        except ValueError as exc:
            logger.warning(f"WARNING: {exc}")
        if key in ["s", "c", "r", "a", "e"]:
            x = str(round(float(x)))
            y = str(round(float(y)))
            logger.info(f"key: {key}: selecting pixel {x=}, {y=}")
            iy = int(y) - 1
            ix = int(x) - 1
            if key == "a":
                last_key_pos = [key, ix, iy]
                tmp_mask = np.zeros(shape=(naxis2, naxis1), dtype=np.uint8)
                tmp_mask[iy, ix] = 1
            elif key in ["s", "c"] and last_key_pos[0] is not None:
                ix1 = min(ix, last_key_pos[1])
                ix2 = max(ix, last_key_pos[1])
                iy1 = min(iy, last_key_pos[2])
                iy2 = max(iy, last_key_pos[2])
                last_key_pos = [None, None, None]
            elif key in ["s", "c", "e"]:
                ix1 = ix
                ix2 = ix
                iy1 = iy
                iy2 = iy
                # last_key_pos = [key, ix, iy]
            if key in ["s", "c", "e"]:
                for iy in range(iy1, iy2 + 1):
                    for ix in range(ix1, ix2 + 1):
                        if key == "s":
                            if continuum_mask[iy, ix] == 1:
                                continuum_mask[iy, ix] = 0
                            source_mask[iy, ix] = 1
                        elif key == "c":
                            if source_mask[iy, ix] == 1:
                                source_mask[iy, ix] = 0
                            continuum_mask[iy, ix] = 1
                        elif key == "e":
                            if source_mask[iy, ix] == 1:
                                source_mask[iy, ix] = 0
                            if continuum_mask[iy, ix] == 1:
                                continuum_mask[iy, ix] = 0
                        else:
                            raise ValueError("Unexpected error")
            elif key == "r":
                for i in range(naxis2):
                    for j in range(naxis1):
                        source_mask[i, j] = 0
                        continuum_mask[i, j] = 0
            elif key == "a":
                pass
            else:
                raise ValueError("Unexpected error")
            # update file with regions
            update_ds9regions(
                data=data,
                footprint_data=footprint_data,
                source_mask=source_mask,
                continuum_mask=continuum_mask,
                tmp_mask=tmp_mask,
                wave=wave,
                fig=fig,
                ax=ax,
                ax_footprint=ax_footprint,
                line_objects=line_objects,
                filename=filename,
                firstplot=False,
                plot_render=plot_render,
            )
            if key == "a":
                tmp_mask = None
        elif key == "x":
            if plot_render in ["matplotlib", "both"]:
                input("Press RETURN to continue with pixel selection...")
        elif key == "h":
            display_help_menu(plot_render)
        elif key == "q":
            cquit = input("Do you want to quit? (y/[n]) ")
            if cquit.lower() in ["y", "yes"]:
                loop = False
                logger.info("Selection of pixels finished!")

    if plot_render in ["matplotlib", "both"]:
        # keep the splot open after updates
        plt.ioff()
        logger.info("Press 'q' to close matplotlib window and stop the program")
        mpl.rcParams["keymap.quit"] = QUIT_KEYS_MATPLOTLIB_ORIG  # Restore the original quit keys in Matplotlib
        plt.show(block=True)


def main(args=None):
    """Main function"""

    parser = argparse.ArgumentParser(
        description="Interactive examination of 3D data cubes with ds9.", formatter_class=RichHelpFormatter
    )

    parser.add_argument("datacube", help="Input 3D FITS data cube", type=str)
    parser.add_argument(
        "-e",
        "--extnum",
        help="Extension number for image in input files.",
        type=int,
    )
    parser.add_argument(
        "--extname",
        help="Extension name for image in input files.",
        type=str,
    )
    parser.add_argument("--i1", help="First pixel along NAXIS3 (default 1)", type=int, default=1)
    parser.add_argument("--i2", help="Last pixel along NAXIS3 (default NAXIS3)", type=str)
    parser.add_argument("--footprint", help="Plot footprint along NAXIS3 (default False)", action="store_true")
    parser.add_argument(
        "--wave-unit", help="Unit of the wavelength axis (default Angstrom)", type=str, default="Angstrom"
    )
    parser.add_argument("--ds9exec", help="Command line to launch ds9 (default 'ds9')", type=str)
    parser.add_argument(
        "--plot_render",
        help="Display to display spectra (default=matplotlib)",
        choices=["matplotlib", "ds9", "both"],
        default="matplotlib",
    )
    parser.add_argument("--input-masks", help="Path to a FITS file with source and continuum masks", type=str)
    parser.add_argument("--output-masks", help="Path to the output FITS file with source and continuum masks", type=str)

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser)
    args = myscript.args
    logger = myscript.logger
    console = myscript.console

    file_datacube = args.datacube
    extnum = args.extnum
    extname = args.extname

    # Check wavelength unit is valid
    wave_unit = u.Unit(args.wave_unit)
    if not wave_unit.is_equivalent(u.m):
        logger.error(f"[red]Invalid wavelength unit: {wave_unit}[/red]")
        sys.exit(1)

    ds9exec = args.ds9exec
    if ds9exec is None:
        # find environment variable DS9EXEC, if not found, use 'ds9'
        logger.info("Searching for environment variable DS9EXEC...")
        ds9exec = os.environ.get("DS9EXEC")
        console.print(
            f"DS9EXEC={ds9exec}", highlight=False
        )  # Do not apply highlighting to the command line, as it may contain special characters
        if ds9exec is None:
            logger.info("Environment variable DS9EXEC not found. Using 'ds9' as default.")
            ds9exec = "ds9"
    else:
        logger.info(f"Using provided ds9exec: {ds9exec}")
    plot_render = args.plot_render

    # Check XPA is installed
    list_required_executables = ["xpaget", "xpaset"]
    for required_executable in list_required_executables:
        execfile = shutil.which(required_executable)
        if execfile is None:
            logger.error(f"[red]Required program {required_executable} is not available[/red]")
            raise SystemExit(1)
        else:
            logger.info(f"Required program {execfile} found!")

    # Check if ds9 is already running
    logger.info("Checking if a previous ds9 instance is running...")
    try:
        filename = ds9cmd("xpaget ds9 file")
    except Exception as exc:
        filename = ""
        logger.info(f"{exc=}")
        logger.info("No previous ds9 instance found. OK!")
    if len(filename) > 0:
        logger.warning(
            "[red]A previous instance of ds9 is already running.\n"
            + f"Filename: {filename}\n"
            + "Please close it before using this program[/red]"
        )
        raise SystemExit(1)

    # Get HDU and data of the FITS file
    fpath = Path(file_datacube)
    with fits.open(fpath) as hdul:
        hdu = get_hdu_from_hdul(hdul, extnum=extnum, extname=extname)
        data = hdu.data
        if args.footprint:
            if "FOOTPRINT" not in hdul:
                logger.error("[red]FOOTPRINT extension not found in the FITS file[/red]")
                raise SystemExit(1)
            footprint_data = hdul["FOOTPRINT"].data
        else:
            footprint_data = None
    if len(data.shape) != 3:
        raise ValueError(f"Expected a 3D cube, but got {data.shape}")
    if footprint_data is not None and footprint_data.shape != data.shape:
        logger.error(
            f"[red]FOOTPRINT extension shape {footprint_data.shape} does not match data shape {data.shape}[/red]"
        )
        raise SystemExit(1)

    naxis3, naxis2, naxis1 = data.shape
    logger.info(f"{naxis1=}")
    logger.info(f"{naxis2=}")
    logger.info(f"{naxis3=}")

    i1 = args.i1
    if i1 < 1 or i1 > naxis3:
        raise ValueError(f"Invalid first pixel={i1} along NAXIS3={naxis3}")
    if args.i2 is None:
        i2 = naxis3
    else:
        i2 = int(args.i2)
        if i2 < i1 or i2 > naxis3:
            raise ValueError(f"Invalid last pixel={i2} along NAXIS3={naxis3}")

    # Get WCS of the HDU
    wcs = get_wcs_from_hdu(hdu)
    logger.info(f"WCS: {wcs}")

    # Collapse the data cube along NAXIS3
    logger.info("Collapsing 3D cubes along NAXIS3... ")
    extract_slice(
        input=file_datacube,
        extnum=extnum,
        extname=extname,
        axis=3,
        i1=i1,
        i2=i2,
        method="sum",
        wavecal="none",
        transpose=False,
        vmin=None,
        vmax=None,
        noplot=True,
        output="tmp_collapsed_3D.fits",
    )

    # Launch ds9
    cmd = f"{ds9exec.split()[0]} tmp_collapsed_3D.fits {' '.join(ds9exec.split()[1:])} &"
    logger.info("Executing:")
    console.print(
        cmd, highlight=False
    )  # Do not apply highlighting to the command line, as it may contain special characters
    # Note: use shell=True below to make the ds9 alias in the system available
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=True)
    if result.stderr != "":
        logger.error(f"[red]{result.stderr}[/red]")
        logger.error("[red]Check environment variable DS9EXEC or make use of the --ds9exec option[/red]")
        raise SystemExit(1)
    input("Press RETURN after ds9 has properly started...")
    try:
        filename = ds9cmd("xpaget ds9 file")
    except Exception as exc:
        logger.error("[red]Fatal error: ds9 is not running[/red]")
        raise SystemExit(1) from exc
    logger.info(f"ds9 working with file: {filename}")

    # Generate array in the spectral direction
    wcs1d_spectral = wcs.spectral
    wave = wcs1d_spectral.pixel_to_world(np.arange(naxis3)).to(wave_unit)
    logger.info(f"Minimum value along NAXIS3: {wave.min()}")
    logger.info(f"Maximum value along NAXIS3: {wave.max()}")

    # Read source and continuum masks or create them
    if args.input_masks:
        source_mask = fits.getdata(args.input_masks, extname="SOURMASK")
        if source_mask.shape != (naxis2, naxis1):
            raise ValueError(f"{source_mask.shape=} does not match {(naxis2, naxis1)=}")
        if source_mask.dtype != np.uint8:
            raise ValueError(f"Source mask dtype {source_mask.dtype} is not uint8")
        continuum_mask = fits.getdata(args.input_masks, extname="CONTMASK")
        if continuum_mask.shape != (naxis2, naxis1):
            raise ValueError(f"{continuum_mask.shape=} does not match {(naxis2, naxis1)=}")
        if continuum_mask.dtype != np.uint8:
            raise ValueError(f"Continuum mask dtype {continuum_mask.dtype} is not uint8")
    else:
        source_mask = np.zeros((naxis2, naxis1), dtype=np.uint8)
        continuum_mask = np.zeros((naxis2, naxis1), dtype=np.uint8)

    if plot_render in ["ds9", "both"]:
        current_plot = init_ds9_plot(fpath=fpath, wave=wave)
        logger.info(f"Opening ds9 plot: {current_plot}")

    update_masks(
        filename=file_datacube,
        data=data,
        footprint_data=footprint_data,
        source_mask=source_mask,
        continuum_mask=continuum_mask,
        wave=wave,
        plot_render=plot_render,
    )

    # Save the masks to a FITS file
    if args.output_masks:
        output_masks = Path(args.output_masks)
    else:
        output_masks = Path("tmp_masks.fits")
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(source_mask, name="SOURMASK")
    hdu2 = fits.ImageHDU(continuum_mask, name="CONTMASK")
    hdul = fits.HDUList([hdu0, hdu1, hdu2])
    hdul.writeto(output_masks, overwrite=True)
    logger.info(f"Masks saved to {output_masks}")

    logger.info("[red]Remember to close the running session of ds9 before re-executing this program![/red]")

    # Display goodbye message and save console log if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
