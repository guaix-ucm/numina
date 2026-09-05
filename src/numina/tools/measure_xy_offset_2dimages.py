#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Determine (X,Y) offsets between 2 2D images using cross-correlation.

The inputs are two 2D images (numpy arrays) with the same dimensions.

The computed (X,Y) offsets indicate how much the second image is shifted
with respect to the first image. The offsets are returned in pixels.

The input images can be pre-processed by subtracting the median background
and/or rescaling to the range [0, 1].

NaN values in the input images are replaced with zeros before computing
the cross-correlation.

It is possible to use --test mode to create synthetic images with a known offset,
which can be used to validate the offset measurement. The synthetic images can also
be saved to FITS files for further analysis.

Usage examples:
    numina-measure_xy_offset_2dimages --test --subtract-background --rescale-to-01 --plots
    numina-measure_xy_offset_2dimages --image1 image1.fits --image2 image2.fits --subtract-background --rescale-to-01 --plots
"""

import argparse
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter
from skimage.registration import phase_cross_correlation
import sys
import teareduce as tea

from numina.array.rescale_array_z1z2 import rescale_array_to_z1z2
from numina.array.yx_offsets_correlate2d import yx_offsets_correlate2d
from numina.tools.initialize_script_with_args import initialize_script_with_args

from numina._version import __version__


def simulate_images(fwhm, amplitude, background, noise, xoffset=0, yoffset=0, num_nans=0, seed=None):
    """Simulate two 2D images with a known offset.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the Gaussian star.
    amplitude : float
        Amplitude of the Gaussian star.
    background : float
        Background level of the images.
    noise : float
        Noise level of the images.
    xoffset : float
        X offset between the two images.
    yoffset : float
        Y offset between the two images.
    num_nans : int
        Number of NaN pixels to insert in each image.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        Two 2D numpy arrays representing the images.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Simulating two 2D images with known offset: xoffset={xoffset}, yoffset={yoffset}")
    logger.info(f"FWHM={fwhm}, amplitude={amplitude}, background={background}, noise={noise}, seed={seed}")

    nx, ny = 101, 101  # dimensions (columns, rows)
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)  # random number generator

    x0, y0 = (nx - 1) / 2, (ny - 1) / 2  # center of the first Gaussian

    # FWHM -> sigma
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Create a grid of (x,y) coordinates
    y, x = np.mgrid[0:ny, 0:nx]

    # Create the first image with a Gaussian star at the center
    star = Gaussian2D(amplitude=amplitude, x_mean=x0, y_mean=y0, x_stddev=sigma, y_stddev=sigma)
    image1_data = star(x, y).astype(np.float32)
    image1_data += background  # add background
    image1_data += rng.normal(loc=0.0, scale=noise, size=image1_data.shape).astype(np.float32)  # add noise

    # Create the second image with a Gaussian star at a different location
    x0_shifted = x0 + xoffset  # shift the center by xoffset pixels in the x direction
    y0_shifted = y0 + yoffset  # shift the center by yoffset pixels in the y direction
    star_shifted = Gaussian2D(amplitude=amplitude, x_mean=x0_shifted, y_mean=y0_shifted, x_stddev=sigma, y_stddev=sigma)
    image2_data = star_shifted(x, y).astype(np.float32)
    image2_data += background  # add background
    image2_data += rng.normal(loc=0.0, scale=noise, size=image2_data.shape).astype(np.float32)  # add noise

    # Insert NaN values in random locations near the borders of both images
    if num_nans > 0:
        for _ in range(num_nans):
            # Randomly choose a pixel near the border of image1
            loop = True
            while loop:
                x_nan1 = rng.integers(0, nx)
                y_nan1 = rng.integers(0, ny)
                if x_nan1 < 5 or x_nan1 >= nx - 5 or y_nan1 < 5 or y_nan1 >= ny - 5:
                    image1_data[y_nan1, x_nan1] = np.nan
                    loop = False

            # Randomly choose a pixel near the border of image2
            loop = True
            while loop:
                x_nan2 = rng.integers(0, nx)
                y_nan2 = rng.integers(0, ny)
                if x_nan2 < 5 or x_nan2 >= nx - 5 or y_nan2 < 5 or y_nan2 >= ny - 5:
                    image2_data[y_nan2, x_nan2] = np.nan
                    loop = False

    return image1_data, image2_data


def cross_correlation_map2d(img1, img2, normalization="phase"):
    """Compute the cross-correlation map between two 2D images.

    Auxiliary function to comput the cross-correlation map between
    two 2D images using FFT. Note that phase_cross_correlation
    from skimage.registration is more robust and should be preferred
    for measuring offsets. In any case, phase_cross_correlation
    does not return the cross-correlation map, so this function
    is provided only for visualization purposes.

    Parameters
    ----------
    img1 : np.ndarray
        First 2D image.
    img2 : np.ndarray
        Second 2D image.
    normalization : str, optional
        Normalization method for the cross-correlation. Default is 'phase'.

    Returns
    -------
    np.ndarray
        The cross-correlation map.
    """
    F1 = np.fft.fft2(img1)
    F2 = np.fft.fft2(img2)
    cross_power = F1 * np.conj(F2)
    if normalization == "phase":
        cross_power /= np.abs(cross_power)
    corr = np.fft.ifft2(cross_power)
    return np.fft.fftshift(np.abs(corr))  # peak centered in the middle of the array, not at the corners


def measure_xy_offset_2dimages(
    image1, image2, subtract_background=True, rescale_to_01=True, method=1, plots=False, log_messages=True
):
    """Determine (X,Y) offsets between 2 2D images using cross-correlation.

    The inputs are two 2D images (numpy arrays) with the same dimensions,
    and the output is a tuple with the (x_offset, y_offset) between the two images.

    The computed offsets indicate how much the second image is shifted
    with respect to the first image. The offsets are returned in pixels.

    The input images can be pre-processed by subtracting the median background
    and/or rescaling to the range [0, 1].

    NaN values in the input images are replaced with zeros before computing
    the cross-correlation.

    Parameters
    ----------
    image1 : np.ndarray
        First 2D image.
    image2 : np.ndarray
        Second 2D image.
    subtract_background : bool
        Whether to subtract the median background from the images
        before computing the cross-correlation. This background is estimated
        as the median of the pixel values in each image.
    rescale_to_01 : bool
        Whether to rescale the images to the range [0, 1] before
        computing the cross-correlation.
    method : int
        Two methods are implemented:
        - method 1: using scipy.signal.correlate2d
        - method 2: using skimage.registration.phase_cross_correlations
    plots : bool
        Whether to generate plots of the images and cross-correlation.
    log_messages : bool
        Whether to log messages about the processing steps.

    Returns
    -------
    tuple
        (x_offset, y_offset) between the two images.
    """
    logger = logging.getLogger(__name__)

    mask1_nan = np.isnan(image1)
    mask2_nan = np.isnan(image2)
    num_mask1_nan = np.sum(mask1_nan)
    num_mask2_nan = np.sum(mask2_nan)
    if log_messages:
        logger.info(f"Image 1: {num_mask1_nan} NaN pixels")
        logger.info(f"Image 2: {num_mask2_nan} NaN pixels")

    if plots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        tea.imshow(fig, ax1, image1, title="Image 1", ds9mode=True)
        tea.imshow(fig, ax2, image2, title="Image 2", ds9mode=True)
        plt.tight_layout()
        plt.show()

    # Subtract the median background from the images if requested
    if subtract_background:
        bkg1 = np.nanmedian(image1)
        bkg2 = np.nanmedian(image2)
        if log_messages:
            logger.info(f"Image 1 background median: {bkg1:.3f}")
            logger.info(f"Image 2 background median: {bkg2:.3f}")
        image1_bkg_sub = image1 - bkg1
        image2_bkg_sub = image2 - bkg2
        if plots:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            tea.imshow(fig, ax1, image1_bkg_sub, title="Image 1 - Background", ds9mode=True)
            tea.imshow(fig, ax2, image2_bkg_sub, title="Image 2 - Background", ds9mode=True)
            plt.tight_layout()
            plt.show()
    else:
        if log_messages:
            logger.warning(
                "Background subtraction is disabled.\n"
                " -> This may affect the accuracy of the offset measurement.\n"
                " -> Consider using --subtract-background for better results."
            )
        image1_bkg_sub = image1
        image2_bkg_sub = image2

    # Replace NaN values with zeros
    if num_mask1_nan > 0:
        image1_bkg_sub = np.nan_to_num(image1_bkg_sub, nan=0.0)
        if log_messages:
            logger.warning(f"Image 1: {num_mask1_nan} NaN pixels replaced with zeros.")
    if num_mask2_nan > 0:
        image2_bkg_sub = np.nan_to_num(image2_bkg_sub, nan=0.0)
        if log_messages:
            logger.warning(f"Image 2: {num_mask2_nan} NaN pixels replaced with zeros.")
    if (num_mask1_nan > 0 or num_mask2_nan > 0) and plots:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        tea.imshow(fig, ax1, image1_bkg_sub, title="Image 1 - Background (NaN replaced)", ds9mode=True)
        tea.imshow(fig, ax2, image2_bkg_sub, title="Image 2 - Background (NaN replaced)", ds9mode=True)
        plt.tight_layout()
        plt.show()

    # Rescale the images if requested
    if rescale_to_01:
        image1_rescaled, _ = rescale_array_to_z1z2(image1_bkg_sub, (0, 1))
        image2_rescaled, _ = rescale_array_to_z1z2(image2_bkg_sub, (0, 1))
        if plots:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            tea.imshow(fig, ax1, image1_rescaled, title="Rescaled Image 1 - Background", ds9mode=True)
            tea.imshow(fig, ax2, image2_rescaled, title="Rescaled Image 2 - Background", ds9mode=True)
            plt.tight_layout()
            plt.show()
    else:
        if log_messages:
            logger.warning(
                "Rescaling to [0, 1] is disabled.\n"
                " -> This may affect the accuracy of the offset measurement.\n"
                " -> Consider using --rescale-to-01 for better results."
            )
        image1_rescaled = image1_bkg_sub
        image2_rescaled = image2_bkg_sub

    # Compute the cross-correlation
    if method == 1:
        # Use the correlate2d function from scipy.signal to compute the cross-correlation
        # This funcion makes use of correlate2d(..., mode='full', boundary='fill', fillvalue=0),
        # which is equivalent to zero-padding the images before computing the cross-correlation.
        yx_offsets = yx_offsets_correlate2d(reference_image=image1_rescaled, moving_image=image2_rescaled)
    elif method == 2:
        if plots:
            # Display the cross-correlation map (this map is not returned by phase_cross_correlation,
            # but it can be useful for visualization)
            cross_corr_map = cross_correlation_map2d(image1_rescaled, image2_rescaled)
            fig, ax = plt.subplots(figsize=(5, 5))
            tea.imshow(fig, ax, cross_corr_map, title="Cross-Correlation Map", ds9mode=True)
            plt.show()
        # There is no need to apply zero-padding here because phase_cross_correlation handles it internally.
        yx_offsets, _, _ = phase_cross_correlation(
            reference_image=image1_rescaled,
            moving_image=image2_rescaled,
            upsample_factor=100,
            overlap_ratio=0.90,
            disambiguate=True,
        )
    else:
        raise ValueError("Invalid method. Use 1 or 2.")

    return -yx_offsets[1], -yx_offsets[0]  # return (x_offset, y_offset)


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Determine (X,Y) offsets between 2 2D images using cross-correlation.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--image1", type=str, help="Path to the first 2D image")
    parser.add_argument("--image2", type=str, help="Path to the second 2D image")
    parser.add_argument("--extnum1", type=int, default=0, help="Extension number for the first image (default: 0)")
    parser.add_argument("--extnum2", type=int, default=0, help="Extension number for the second image (default: 0)")
    parser.add_argument("--subtract-background", action="store_true", help="Subtract median background from images")
    parser.add_argument(
        "--rescale-to-01",
        action="store_true",
        help="Rescale images to the range [0, 1] before computing cross-correlation",
    )
    parser.add_argument("--method", help="Method (1: scipy (default), 2: skimage)", type=int, choices=[1, 2], default=1)
    parser.add_argument("--plots", action="store_true", help="Generate plots of the images and cross-correlation")

    parser.add_argument("--test", action="store_true", help="Run test mode with synthetic images")
    parser.add_argument(
        "--test-fwhm", type=float, default=10.0, help="FWHM of the synthetic Gaussian star (default: 10.0)"
    )
    parser.add_argument(
        "--test-amplitude",
        type=float,
        default=1000.0,
        help="Amplitude of the synthetic Gaussian star (default: 1000.0)",
    )
    parser.add_argument(
        "--test-background", type=float, default=100.0, help="Background level of the synthetic images (default: 100.0)"
    )
    parser.add_argument(
        "--test-noise", type=float, default=5.0, help="Noise level of the synthetic images (default: 5.0)"
    )
    parser.add_argument(
        "--test-xoffset", type=float, default=5.0, help="X offset of the synthetic images (default: 5.0)"
    )
    parser.add_argument(
        "--test-yoffset", type=float, default=3.0, help="Y offset of the synthetic images (default: 3.0)"
    )
    parser.add_argument(
        "--test-num-nans",
        type=int,
        default=20,
        help="Number of NaN pixels to insert in each synthetic image (default: 20)",
    )
    parser.add_argument("--test-seed", type=int, default=1234, help="Random seed for synthetic images (default: 1234)")
    parser.add_argument("--save-test-images", action="store_true", help="Save synthetic images to FITS files")

    parser.add_argument("--output-dir", help="Output directory (default: .)", type=str, default=".")
    parser.add_argument("--record", help="Record terminal output", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--version", help="Display version", action="store_true")
    parser.add_argument(
        "--log-level",
        help="Set the logging level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger = initialize_script_with_args(sys.argv, parser, args, __name__, __version__)

    # If test mode is enabled, create synthetic images.
    # Otherwise, read the images from the provided paths.
    if args.test:
        if args.image1 is not None or args.image2 is not None:
            logger.warning("Test mode enabled: --image1 and --image2 will be ignored.")
        # Creating synthetic images
        logger.info("Test mode enabled: synthetic images will be created.")
        image1_data, image2_data = simulate_images(
            fwhm=args.test_fwhm,
            amplitude=args.test_amplitude,
            background=args.test_background,
            xoffset=args.test_xoffset,
            yoffset=args.test_yoffset,
            noise=args.test_noise,
            num_nans=args.test_num_nans,
            seed=args.test_seed,
        )
        if args.save_test_images:
            # Save the synthetic images to FITS files
            image1_fits_path = Path(args.output_dir) / "test1.fits"
            image2_fits_path = Path(args.output_dir) / "test2.fits"
            fits.writeto(image1_fits_path, image1_data, overwrite=True)
            fits.writeto(image2_fits_path, image2_data, overwrite=True)
            logger.info(f"Synthetic images saved to {image1_fits_path} and {image2_fits_path}")
    else:
        # Check input images
        if args.image1 is None or args.image2 is None:
            logger.error("Both --image1 and --image2 must be provided.")
            parser.print_usage()
            raise SystemExit()
        # Read the images
        image1_path = Path(args.image1)
        image2_path = Path(args.image2)
        if not image1_path.exists():
            logger.error(f"Image file {image1_path} does not exist.")
            raise SystemExit()
        if not image2_path.exists():
            logger.error(f"Image file {image2_path} does not exist.")
            raise SystemExit()
        with fits.open(image1_path) as hdul1:
            image1_data = hdul1[args.extnum1].data
        with fits.open(image2_path) as hdul2:
            image2_data = hdul2[args.extnum2].data

    # Check the images have the same shape
    if image1_data.shape != image2_data.shape:
        logger.error("The two images must have the same shape.")
        raise SystemExit()

    # Compute the offsets
    x_offset, y_offset = measure_xy_offset_2dimages(
        image1=image1_data,
        image2=image2_data,
        subtract_background=args.subtract_background,
        rescale_to_01=args.rescale_to_01,
        method=args.method,
        plots=args.plots,
    )
    logger.info(f"Computed offsets (pixels): x_offset = {x_offset}, y_offset = {y_offset}")

    # Execution time
    datetime_end = datetime.now()
    time_elapsed = datetime_end - datetime_ini
    logger.info("Total time elapsed: %s", str(time_elapsed))

    # Goodbye message
    console.rule("[bold magenta] Goodbye! [/bold magenta]")

    # Save console log if recording is enabled
    if args.record:
        output_dir_path = Path(args.output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        log_filename = Path(args.output_dir) / "terminal_output.txt"
        with open(log_filename, "wt") as f:
            f.write(console.export_text(styles=True))
        logger.info(f"terminal output recorded in [green]{log_filename}[/green]")


if __name__ == "__main__":
    main()
