#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Computation of cosmic ray masks using several equivalent exposures."""
import ast
import inspect
import io
import logging
import sys
import uuid

from astropy.io import fits
from ccdproc import cosmicray_lacosmic
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from numina.array.distortion import shift_image2d
from numina.array.numsplines import spline_positive_derivative
import teareduce as tea

from .valid_parameters import VALID_CRMETHODS
from .valid_parameters import VALID_LACOSMIC_CLEANTYPE
from .valid_parameters import VALID_BOUNDARY_FITS
from .all_valid_numbers import all_valid_numbers
from .compute_flux_factor import compute_flux_factor
from .define_piecewise_linear_function import define_piecewise_linear_function
from .diagnostic_plot import diagnostic_plot
from .display_detected_cr import display_detected_cr
from .estimate_diagnostic_limits import estimate_diagnostic_limits
from .gausskernel2d_elliptical import gausskernel2d_elliptical


def decorate_output(func):
    """Decorator to capture stdout and stderr of a function and log it."""

    def wrapper(*args, **kwargs):
        _logger = logging.getLogger(__name__)
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            result = func(*args, **kwargs)
        # Split into lines
        output_lines = buf.getvalue().splitlines()
        # Remove trailing empty line
        if output_lines and output_lines[-1] == '':
            output_lines = output_lines[:-1]
        if output_lines:
            _logger.info("\n".join(output_lines))
        return result

    return wrapper


@decorate_output
def decorated_cosmicray_lacosmic(*args, **kwargs):
    """Wrapper for cosmicray_lacosmic with decorated output."""
    return cosmicray_lacosmic(*args, **kwargs)


def compute_crmasks(
        list_arrays,
        gain=None,
        rnoise=None,
        bias=None,
        crmethod='mm_lacosmic',
        use_lamedian=False,
        flux_factor=None,
        flux_factor_regions=None,
        apply_flux_factor_to=None,
        interactive=True,
        dilation=1,
        regions_to_be_skipped=None,
        pixels_to_be_flagged_as_cr=None,
        pixels_to_be_ignored_as_cr=None,
        pixels_to_be_replaced_by_local_median=None,
        dtype=np.float32,
        verify_cr=False,
        semiwindow=15,
        color_scale='minmax',
        maxplots=-1,
        debug=False,
        _logger=None,
        la_gain_apply=True,
        la_sigclip=None,
        la_sigfrac=None,
        la_objlim=None,
        la_satlevel=None,
        la_niter=None,
        la_sepmed=None,
        la_cleantype=None,
        la_fsmode=None,
        la_psfmodel=None,
        la_psffwhm_x=None,
        la_psffwhm_y=None,
        la_psfsize=None,
        la_psfbeta=None,
        la_verbose=False,
        mm_xy_offsets=None,
        mm_crosscorr_region=None,
        mm_boundary_fit=None,
        mm_knots_splfit=3,
        mm_fixed_points_in_boundary=None,
        mm_nsimulations=10,
        mm_niter_boundary_extension=3,
        mm_weight_boundary_extension=10.0,
        mm_threshold=0.0,
        mm_minimum_max2d_rnoise=5.0,
        mm_seed=None
        ):
    """
    Computation of cosmic rays masks using several equivalent exposures.

    This function computes cosmic ray masks from a list of 2D numpy arrays.
    Two different methods are implemented:
    1. Cosmic ray detection using the Laplacian edge detection algorithm
       (van Dokkum 2001), as implemented in ccdproc.cosmicray_lacosmic.
    2. Cosmic ray detection using a numerically derived boundary in the
       median combined image. The cosmic ray detection is based on a boundary
       that is derived numerically making use of the provided gain and readout
       noise values. The function also supports generating diagnostic plots to
       visualize the cosmic ray detection process.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined. The arrays are assumed to be
        provided in ADU.
    gain : 2D array, float or None
        The gain value (in e/ADU) of the detector.
        If None, it is assumed to be 1.0.
    rnoise : 2D array, float or None
        The readout noise (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    bias : 2D array, float or None
        The bias value (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    crmethod : str
        The method to use for cosmic ray detection. Valid options are:
        - 'lacosmic': use the cosmic-ray rejection by Laplacian edge
        detection (van Dokkum 2001), as implemented in ccdproc.
        - 'mmcosmic': use the numerically derived boundary to
        detect cosmic rays in the median combined image.
        - 'mm_lacosmic': use both methods: 'lacosmic' and 'mmcosmic'.
        Pixels detected by either method are included in the final mask.
    use_lamedian: bool, optional
        If True, use the corrected values from the lacosmic algorithm
        when replacing the cosmic-ray affected pixels in the median
        combined image. This parameter is only used when
        crmethod is 'lacosmic', or 'mm_lacosmic'.
    flux_factor : str, list, float or None, optional
        The flux scaling factor for each exposure (default is None).
        If 'auto', the flux factor is determined automatically.
        If None or 'none', it is set to 1.0 for all images.
        If a float is provided, it is used as the flux factor for all images.
        If a list is provided, it should contain a value
        for each single image in `list_arrays`.
    flux_factor_regions : list of lists of 4 integers or None, optional
        The regions to use for computing the flux scaling factors
        when flux_factor='auto'. If None, the entire image is used.
        The format of the each region must be a list of 4 integers, following
        the FITS convention [xmin, xmax, ymin, ymax], where the indices
        start from 1 to NAXIS[12].
    apply_flux_factor_to : str
        Specifies to which images the flux factor should be applied.
        Valid options are:
        - 'original': apply the flux factor to the original images.
        - 'simulated': apply the flux factor to the simulated images
          used to derive the boundary.
    interactive : bool, optional
        If True, enable interactive mode for plots.
    dilation : int, optional
        The dilation factor for the coincident cosmic-ray pixel mask.
    regions_to_be_skipped : list of lists of 4 integers or None, optional
        The regions to be skipped during cosmic ray detection.
        If None, no regions are skipped. The format of each region must
        be a list of 4 integers, following the FITS convention
        [xmin, xmax, ymin, ymax], where the indices start from 1 to NAXIS[12].
    pixels_to_be_flagged_as_cr : list of (x, y) tuples, or None, optional
        List of pixel coordinates to be included in the masks
        (Assuming FITS criterium; first pixel is (1, 1)).
    pixels_to_be_ignored_as_cr : list of (x, y) tuples, or None, optional
        List of pixel coordinates to be excluded from the masks
        (Assuming FITS criterium; first pixel is (1, 1)).
    pixels_to_be_replaced_by_local_median : list of (x, y, x_with, y_width) tuples, or None, optional
        List of pixel coordinates to be replaced by the median value
        when removing the cosmic rays. This information is stored as a binary
        table in one of the extensions of the output HDUList with the masks.
        (Assuming FITS criterium; first pixel is (1, 1)).
    dtype : data-type, optional
        The desired data type to build the 3D stack (default is np.float32).
    verify_cr : bool, optional
        If True, verify the cosmic ray detection by comparing the
        detected positions with the original images (default is True). It
        is only used if interactive=True.
    semiwindow : int, optional
        The semiwindow size to plot the coincident cosmic-ray pixels (default is 15).
    color_scale : str, optional
        The color scale to use for the plots (default is 'minmax').
        Valid options are 'minmax' and 'zscale'.
    maxplots : int, optional
        The maximum number of coincident cosmic-ray pixels to plot (default is -1).
        If negative, all detected cosmic-ray pixels will be plotted.
    debug : bool, optional
        If True, enable debug mode (default is False).
    _logger : logging.Logger or None, optional
        The logger to use for logging. If None, a new logger is created.
    la_gain_apply: bool, optional
        If True, apply the gain when computing the cosmic ray mask
        with the lacosmic algorithm. Default is True.
    la_sigclip : float
        The sigma clipping threshold. Employed when crmethod='lacosmic'.
    la_sigfrac : float
        The fractional detection limit for neighboring pixels.
        Employed when crmethod='lacosmic'.
    la_objlim : float
        Minimum contrast between Laplacian image and fine structure image.
        Employed when crmethod='lacosmic'.
    la_satlevel : float
        The saturation level (in ADU) of the detector. Employed when crmethod='lacosmic'.
    la_niter : int
        The number of iterations to perform. Employed when crmethod='lacosmic'.
    la_sepmed : bool
        If True, use separable median filter instead of the full median filter.
        Employed when crmethod='lacosmic'.
    la_fsmode : str
        The mode to use for the fine structure image. Valid options are:
        'median' or 'convolve'. Employed when crmethod='lacosmic'.
    la_psfmodel : str
        The model to use for the PSF if la_fsmode='convolve'.
        Valid options are:
        - circular kernels: 'gauss' or 'moffat'
        - Gaussian in the x and y directions: 'gaussx' and 'gaussy'
        - elliptical Gaussian: 'gaussxy' (this kernel is not available
          in ccdproc.cosmicray_lacosmic, so it is implemented here)
        Employed when crmethod='lacosmic'.
    la_psffwhm_x : float
        The full width at half maximum (FWHM, in pixels) of the PSF in
        the x direction. Employed when crmethod='lacosmic'.
    la_psffwhm_y : float
        The full width at half maximum (FWHM, in pixels) of the PSF
        in the y direction. Employed when crmethod='lacosmic'.
    la_psfsize : int
        The kernel size to use for the PSF. It must be an odd integer >= 3.
        Employed when crmethod='lacosmic'.
    la_psfbeta : float
        The beta parameter of the Moffat PSF. It is only used if
        la_psfmodel='moffat'. Employed when crmethod='lacosmic'.
    la_verbose : bool
        If True, print additional information during the
        execution. Employed when crmethod='lacosmic'.
    mm_xy_offsets: list of [x_offset, y_offset] or None
        The offsets to apply to each simulated individual exposure
        when computing the diagnostic diagram for the mmcosmic method.
        If None, no offsets are applied.
        This option is not compatible with 'mm_crosscorr_region'.
    mm_crosscorr_region : list of 4 integers or None
        The region to use for the 2D cross-correlation to determine
        the offsets between the individual images and the median image.
        If None, no offsets are computed and it is assumed that
        the images are already aligned. The format of the region
        must follow the FITS convention [xmin, xmax, ymin, ymax],
        where the indices start from 1 to NAXIS[12].
        This option is not compatible with 'mm_xy_offsets'.
    mm_boundary_fit : str, or None
        The method to use for the boundary fitting. Valid options are:
        - 'spline': use a spline fit to the boundary.
        - 'piecewise': use a piecewise linear fit to the boundary.
    mm_knots_splfit : int, optional
        The number of knots for the spline fit to the boundary.
    mm_fixed_points_in_boundary : str, or list or None
        The fixed points to use for the boundary fitting.
    mm_nsimulations : int, optional
        The number of simulations of each set of input images to compute
        the detection boundary.
    mm_niter_boundary_extension : int, optional
        The number of iterations for the boundary extension.
    mm_weight_boundary_extension : float, optional
        The weight for the boundary extension.
        In each iteration, the boundary is extended by applying an
        extra weight to the points above the previous boundary. This
        extra weight is computed as `mm_weight_boundary_extension**iter`,
        where `iter` is the current iteration number (starting from 1).
    mm_threshold: float, optional
        Minimum threshold for median2d - min2d to consider a pixel as a
        cosmic ray (default is None). If None, the threshold is computed
        automatically from the minimum boundary value in the numerical
        simulations.
    mm_minimum_max2d_rnoise : float, optional
        Minimum value for max2d in readout noise units to flag a pixel
        as a coincident cosmic-ray pixel.
    mm_seed : int or None, optional
        The random seed for reproducibility.

    Returns
    -------
    hdul_masks : hdulist
        The HDUList containing the mask arrays for cosmic ray
        removal using different methods. The primary HDU only contains
        information about the parameters used to determine the
        suspected pixels. The extensions are:
        - 'MEDIANCR': Mask for coincident cosmic-ray pixels detected using the
        median combination.
        - 'MEANCRT': Mask for cosmic-ray pixels detected when adding all the
        individual arrays. That summed image contains all the cosmic-ray pixels.
        of all the images.
        - 'CRMASK1', 'CRMASK2', ...: Masks for cosmic-ray pixels detected
        in each individual array.
    """

    if _logger is None:
        _logger = logging.getLogger(__name__)
        rich_configured = False
    else:
        # use the provided logger
        root_logger = logging.getLogger()
        # check if RichHandler is configured in the parent logger's handler
        rich_configured = any(isinstance(handler, RichHandler) for handler in root_logger.handlers)
    # Set up rich labels for logging
    if rich_configured:
        rlabel_crmethod = f"[bold green]{crmethod}[/bold green]"
        rlabel_lacosmic = "[bold red]lacosmic[/bold red]"
        rlabel_mmcosmic = "[bold blue]mmcosmic[/bold blue]"
    else:
        rlabel_crmethod = f"{crmethod}"
        rlabel_lacosmic = "lacosmic"
        rlabel_mmcosmic = "mmcosmic"

    # Check that the input is a list
    if not isinstance(list_arrays, list):
        raise TypeError("Input must be a list of arrays.")

    # Check that the list contains numpy 2D arrays
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in list_arrays):
        raise ValueError("All elements in the list must be 2D numpy arrays.")

    # Check that the list contains at least 3 arrays
    num_images = len(list_arrays)
    if num_images < 3:
        raise ValueError("At least 3 images are required for useful mediancr combination.")

    # Check that all arrays have the same shape
    for i, array in enumerate(list_arrays):
        if array.shape != list_arrays[0].shape:
            raise ValueError(f"Array {i} has a different shape than the first array.")
    naxis2, naxis1 = list_arrays[0].shape

    # Log the number of input arrays and their shapes
    _logger.info("number of input arrays: %d", len(list_arrays))
    for i, array in enumerate(list_arrays):
        _logger.debug("array %d shape: %s, dtype: %s", i, array.shape, array.dtype)

    # Check that interactive is True if verify_cr is True
    if verify_cr and not interactive:
        raise ValueError("interactive must be True if verify_cr is True.")

    # Define the gain
    gain_scalar = None  # store gain as a scalar value (if applicable)
    if gain is None:
        gain = np.ones((naxis2, naxis1), dtype=float)
        _logger.warning("gain not defined, assuming gain=1.0 for all pixels.")
        gain_scalar = 1.0
    elif isinstance(gain, (float, int)):
        gain = np.full((naxis2, naxis1), gain, dtype=float)
        _logger.info("gain defined as a constant value: %f", gain[0, 0])
        gain_scalar = float(gain[0, 0])
    elif isinstance(gain, np.ndarray):
        if gain.shape != (naxis2, naxis1):
            raise ValueError(f"gain must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("gain defined as a 2D array with shape: %s", gain.shape)
        if np.all(gain == gain[0, 0]):
            gain_scalar = float(gain[0, 0])
    else:
        raise TypeError(f"Invalid type for gain: {type(gain)}. Must be float, int, or numpy array.")

    # Define the readout noise
    rnoise_scalar = None  # store rnoise as a scalar value (if applicable)
    if rnoise is None:
        rnoise = np.zeros((naxis2, naxis1), dtype=float)
        _logger.warning("readout noise not defined, assuming readout noise=0.0 for all pixels.")
        rnoise_scalar = 0.0
    elif isinstance(rnoise, (float, int)):
        rnoise = np.full((naxis2, naxis1), rnoise, dtype=float)
        _logger.info("readout noise defined as a constant value: %f", rnoise[0, 0])
        rnoise_scalar = float(rnoise[0, 0])
    elif isinstance(rnoise, np.ndarray):
        if rnoise.shape != (naxis2, naxis1):
            raise ValueError(f"rnoise must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("readout noise defined as a 2D array with shape: %s", rnoise.shape)
        if np.all(rnoise == rnoise[0, 0]):
            rnoise_scalar = float(rnoise[0, 0])
    else:
        raise TypeError(f"Invalid type for rnoise: {type(rnoise)}. Must be float, int, or numpy array.")

    # Define the bias
    if bias is None:
        bias = np.zeros((naxis2, naxis1), dtype=float)
        _logger.warning("bias not defined, assuming bias=0.0 for all pixels.")
    elif isinstance(bias, (float, int)):
        bias = np.full((naxis2, naxis1), bias, dtype=float)
        _logger.info("bias defined as a constant value: %f", bias[0, 0])
    elif isinstance(bias, np.ndarray):
        if bias.shape != (naxis2, naxis1):
            raise ValueError(f"bias must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("bias defined as a 2D array with shape: %s", bias.shape)
    else:
        raise TypeError(f"Invalid type for bias: {type(bias)}. Must be float, int, or numpy array.")

    # Convert the list of arrays to a 3D numpy array
    image3d = np.zeros((num_images, naxis2, naxis1), dtype=dtype)
    for i, array in enumerate(list_arrays):
        image3d[i] = array.astype(dtype)

    # Subtract the bias from the input arrays
    _logger.info("subtracting bias from the input arrays")
    image3d -= bias

    # Check crmethod
    if crmethod not in VALID_CRMETHODS:
        raise ValueError(f"Invalid crmethod: {crmethod}. Valid options are {VALID_CRMETHODS}.")
    _logger.info("computing crmasks using crmethod: %s", rlabel_crmethod)
    # Check use_lamedian
    if use_lamedian and crmethod not in ['lacosmic', 'mm_lacosmic']:
        raise ValueError("use_lamedian can only be True when crmethod is 'lacosmic' or 'mm_lacosmic'.")
    _logger.info("use_lamedian: %s", str(use_lamedian))

    # Check flux_factor
    if isinstance(flux_factor_regions, str):
        if flux_factor_regions.lower() == 'none':
            flux_factor_regions = None
    if flux_factor is None:
        flux_factor = np.ones(num_images, dtype=float)
    elif isinstance(flux_factor, str):
        if flux_factor.lower() == 'auto':
            _logger.info("flux_factor set to 'auto', computing values...")
            list_flux_factor_regions = []
            if isinstance(flux_factor_regions, str):
                flux_factor_regions = ast.literal_eval(flux_factor_regions)
            if isinstance(flux_factor_regions, list):
                for flux_factor_region in flux_factor_regions:
                    if isinstance(flux_factor_region, list):
                        all_integers = all(isinstance(val, int) for val in flux_factor_region)
                        if not all_integers:
                            raise TypeError(f"Invalid flux_factor_region: {flux_factor_region}. "
                                            "All elements must be integers.")
                        if len(flux_factor_region) != 4:
                            raise ValueError(f"Invalid length for flux_factor_region: {flux_factor_region}. "
                                             "Must be a list of 4 integers [xmin, xmax, ymin, ymax].")
                        dumreg = f"[{flux_factor_region[0]}:{flux_factor_region[1]}, " + \
                                 f"{flux_factor_region[2]}:{flux_factor_region[3]}]"
                        _logger.debug("defined flux factor region: %s", dumreg)
                        ff_region = tea.SliceRegion2D(dumreg, mode='fits', naxis1=naxis1, naxis2=naxis2)
                    else:
                        raise TypeError(f"Invalid type for flux_factor_region in the list: {type(flux_factor_region)}. "
                                        "Must be a list of 4 integers")
                    list_flux_factor_regions.append(ff_region)
            elif flux_factor_regions is None:
                ff_region = tea.SliceRegion2D(f'[1:{naxis1}, 1:{naxis2}]', mode='fits')
                list_flux_factor_regions = [ff_region]
            else:
                raise TypeError(f"Invalid type for flux_factor_regions: {type(flux_factor_regions)}. "
                                "Must be list of 4 integers or None.")
            median2d = np.median(image3d, axis=0)
            flux_factor = compute_flux_factor(image3d, median2d, list_flux_factor_regions, _logger, interactive, debug)
            _logger.info("computed flux_factor set to %s", str(flux_factor))
        elif flux_factor.lower() == 'none':
            flux_factor = np.ones(num_images, dtype=float)
            if flux_factor_regions is not None:
                raise ValueError("Using flux_factor='none', but flux_factor_regions is provided. "
                                 "You must use flux_factor='auto' to use flux_factor_regions.")
        elif isinstance(ast.literal_eval(flux_factor), list):
            flux_factor = ast.literal_eval(flux_factor)
            if len(flux_factor) != num_images:
                raise ValueError(f"flux_factor must have the same length as the number of images ({num_images}).")
            if not all_valid_numbers(flux_factor):
                raise ValueError(f"All elements in flux_factor={flux_factor} must be valid numbers.")
            flux_factor = np.array(flux_factor, dtype=float)
        elif isinstance(ast.literal_eval(flux_factor), (float, int)):
            flux_factor = np.full(num_images, ast.literal_eval(flux_factor), dtype=float)
        else:
            raise ValueError(f"Invalid flux_factor string: {flux_factor}. Use 'auto' or 'none'.")
    elif isinstance(flux_factor, list):
        if len(flux_factor) != num_images:
            raise ValueError(f"flux_factor must have the same length as the number of images ({num_images}).")
        if not all_valid_numbers(flux_factor):
            raise ValueError(f"All elements in flux_factor={flux_factor} must be valid numbers.")
        flux_factor = np.array(flux_factor, dtype=float)
    else:
        raise ValueError(f"Invalid flux_factor value: {flux_factor}.")
    _logger.info("flux_factor: %s", str(flux_factor))
    if apply_flux_factor_to not in ['original', 'simulated']:
        raise ValueError(f"Invalid apply_flux_factor_to: {apply_flux_factor_to}. "
                         "Valid options are 'original' and 'simulated'.")
    _logger.info("apply_flux_factor_to: %s", apply_flux_factor_to)

    # Apply the flux factor to the input arrays if requested
    if apply_flux_factor_to == 'original':
        for i in range(num_images):
            image3d[i] /= flux_factor[i]

    # Define regions to be cleaned by computing a boolean mask
    # that is True for pixels not included in the regions to be skipped
    bool_to_be_cleaned = np.ones((naxis2, naxis1), dtype=bool)  # default is to clean all pixels
    if isinstance(regions_to_be_skipped, str):
        if regions_to_be_skipped.lower() == 'none':
            regions_to_be_skipped = None
        else:
            regions_to_be_skipped = ast.literal_eval(regions_to_be_skipped)
    if isinstance(regions_to_be_skipped, list):
        for region in regions_to_be_skipped:
            if isinstance(region, list):
                all_integers = all(isinstance(val, int) for val in region)
                if not all_integers:
                    raise TypeError(f"Invalid region_to_be_skipped: {region}. All elements must be integers.")
                if len(region) != 4:
                    raise ValueError(f"Invalid length for region_to_be_skipped: {region}. "
                                     "Must be a list of 4 integers [xmin, xmax, ymin, ymax].")
                dumreg = f"[{region[0]}:{region[1]}, {region[2]}:{region[3]}]"
                _logger.debug("defined region to be skipped: %s", dumreg)
                skip_region = tea.SliceRegion2D(dumreg, mode='fits', naxis1=naxis1, naxis2=naxis2)
                bool_to_be_cleaned[skip_region.python[0].start:skip_region.python[0].stop,
                                   skip_region.python[1].start:skip_region.python[1].stop] = False
            else:
                raise TypeError(f"Invalid type for region_to_be_skipped in the list: {type(region)}. "
                                "Must be a list of 4 integers.")

    # Compute minimum, maximum, median and mean along the first axis
    min2d = np.min(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    mean2d = np.mean(image3d, axis=0)

    # Compute points for diagnostic diagram of mmcosmic method
    xplot = min2d.flatten()  # bias was already subtracted above
    yplot = median2d.flatten() - min2d.flatten()

    # Check that color_scale is valid
    if color_scale not in ['minmax', 'zscale']:
        raise ValueError(f"Invalid color_scale: {color_scale}. Valid options are 'minmax' and 'zscale'.")

    # Define the pixels to be flagged as CR
    if isinstance(pixels_to_be_flagged_as_cr, str):
        if pixels_to_be_flagged_as_cr.lower() == 'none':
            pixels_to_be_flagged_as_cr = None
        else:
            pixels_to_be_flagged_as_cr = ast.literal_eval(pixels_to_be_flagged_as_cr)
    if isinstance(pixels_to_be_flagged_as_cr, (list, tuple)):
        for p in pixels_to_be_flagged_as_cr:
            if (not isinstance(p, (list, tuple)) or len(p) != 2 or
                    not all(isinstance(item, int) for item in p)):
                raise ValueError(f"Invalid numbers in pixels_to_be_flagged_as_cr: {p}. "
                                 "Each pixel must be a tuple or list of two integers (X, Y).")
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_flagged_as_cr are out of bounds.")
            # ensure these pixels are cleaned, independently of being included
            # in regions_to_be_skipped
            bool_to_be_cleaned[p[1]-1, p[0]-1] = True
    elif pixels_to_be_flagged_as_cr is not None:
        raise TypeError(f"Invalid type for pixels_to_be_flagged_as_cr: {type(pixels_to_be_flagged_as_cr)}. "
                        "Must be a list of (x, y) tuples or None.")
    _logger.info("individual pixels to be initially flagged as CR: %s",
                 "None" if pixels_to_be_flagged_as_cr is None else str(pixels_to_be_flagged_as_cr))

    # Define the pixels to be ignored as CR
    # check that the provided pixels are valid
    if isinstance(pixels_to_be_ignored_as_cr, str):
        if pixels_to_be_ignored_as_cr.lower() == 'none':
            pixels_to_be_ignored_as_cr = None
        else:
            pixels_to_be_ignored_as_cr = ast.literal_eval(pixels_to_be_ignored_as_cr)
    if isinstance(pixels_to_be_ignored_as_cr, (list, tuple)):
        for p in pixels_to_be_ignored_as_cr:
            if (not isinstance(p, (list, tuple)) or len(p) != 2 or
                    not all(isinstance(item, int) for item in p)):
                raise ValueError(f"Invalid numbers in pixels_to_be_ignored_as_cr: {p}. "
                                 "Each pixel must be a tuple or list of two integers (X, Y).")
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_ignored_as_cr are out of bounds.")
            # ensure these pixels are not cleaned
            bool_to_be_cleaned[p[1]-1, p[0]-1] = False
    elif pixels_to_be_ignored_as_cr is not None:
        raise TypeError(f"Invalid type for pixels_to_be_ignored_as_cr: {type(pixels_to_be_ignored_as_cr)}. "
                        "Must be a list of (x, y) tuples or None.")
    _logger.info("individual pixels to be initially ignored as CR: %s",
                 "None" if pixels_to_be_ignored_as_cr is None else str(pixels_to_be_ignored_as_cr))

    # Define the pixels to be replaced by the median value when removing the CRs
    # check that the provided pixels are valid
    if isinstance(pixels_to_be_replaced_by_local_median, str):
        if pixels_to_be_replaced_by_local_median.lower() == 'none':
            pixels_to_be_replaced_by_local_median = None
    if isinstance(pixels_to_be_replaced_by_local_median, (list, tuple)):
        for p in pixels_to_be_replaced_by_local_median:
            if (not isinstance(p, (list, tuple)) or len(p) != 4 or
                    not all(isinstance(item, int) for item in p)):
                raise ValueError(f"Invalid numbers in pixels_to_be_replaced_by_local_median: {p}. "
                                 "Each pixel must be a tuple or list of four integers (X, Y, X_width, Y_width).")
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_replaced_by_local_median are out of bounds.")
            if p[2] % 2 == 0 or p[3] % 2 == 0 or p[2] < 1 or p[3] < 1:
                raise ValueError(f"Pixel {p}: X_width and Y_width in pixels_to_be_replaced_by_local_median "
                                 "must be odd integers >= 1.")
            if p[2] * p[3] < 3:
                raise ValueError(f"Pixel {p}: The area defined by X_width and Y_width in "
                                 "pixels_to_be_replaced_by_local_median must be >= 3.")
    elif pixels_to_be_replaced_by_local_median is not None:
        raise TypeError(f"Invalid type for pixels_to_be_replaced_by_local_median: "
                        f"{type(pixels_to_be_replaced_by_local_median)}. "
                        "Must be a list of (x, y, x_width, y_width) tuples or None.")
    _logger.info("pixels to be replaced by the median value when removing the CRs: %s",
                 "None" if pixels_to_be_replaced_by_local_median is None
                 else str(pixels_to_be_replaced_by_local_median))

    # These pixels to be replaced by the local median should not be
    # flagged as CR if they will be replaced by the local median anyway
    if pixels_to_be_replaced_by_local_median is not None:
        for p in pixels_to_be_replaced_by_local_median:
            if bool_to_be_cleaned[p[1]-1, p[0]-1]:
                _logger.warning("Pixel %s is set to be replaced by the local median "
                                "but it is also set to be cleaned as CR. "
                                "It will not be cleaned as CR but will be replaced by the local median.", str(p))
                bool_to_be_cleaned[p[1]-1, p[0]-1] = False
        _logger.info("updated pixels to be ignored as CR: %s",
                     "None" if pixels_to_be_ignored_as_cr is None else str(pixels_to_be_ignored_as_cr))

    # Log the input parameters
    if crmethod in ['mmcosmic', 'mm_lacosmic']:
        _logger.debug("mm_xy_offsets: %s", str(mm_xy_offsets) if mm_xy_offsets is not None else "None")
        _logger.debug("mm_crosscorr_region: %s", mm_crosscorr_region if mm_crosscorr_region is not None else "None")
        _logger.debug("mm_boundary_fit: %s", mm_boundary_fit if mm_boundary_fit is not None else "None")
        _logger.debug("mm_knots_splfit: %d", mm_knots_splfit)
        _logger.debug("mm_fixed points_in_boundary: %s",
                      str(mm_fixed_points_in_boundary) if mm_fixed_points_in_boundary is not None else "None")
        _logger.debug("mm_nsimulations: %d", mm_nsimulations)
        _logger.debug("mm_niter_boundary_extension: %d", mm_niter_boundary_extension)
        _logger.debug("mm_weight_boundary_extension: %f", mm_weight_boundary_extension)
        _logger.debug("mm_threshold: %s", mm_threshold if mm_threshold is not None else "None")
        _logger.debug("mm_minimum_max2d_rnoise: %f", mm_minimum_max2d_rnoise)
        _logger.debug("mm_seed: %s", str(mm_seed))

    if crmethod in ['lacosmic', 'mm_lacosmic']:
        # Check la_gain_apply
        if la_gain_apply is None:
            la_gain_apply = True
            _logger.warning("la_gain_apply for lacosmic not defined, assuming la_gain_apply=True")
        else:
            _logger.debug("la_gain_apply for lacosmic: %s", str(la_gain_apply))
        # Check la_sigclip
        if la_sigclip is None:
            _logger.warning("la_sigclip for lacosmic not defined, assuming la_sigclip=5.0")
            la_sigclip = 5.0
        else:
            _logger.debug("la_sigclip for lacosmic: %f", la_sigclip)
        # Check la_sigfrac
        if la_sigfrac is None:
            _logger.warning("la_sigfrac for lacosmic not defined, assuming la_sigfrac=0.3")
            la_sigfrac = 0.3
        else:
            _logger.debug("la_sigfrac for lacosmic: %f", la_sigfrac)
        # Check la_objlim
        if la_objlim is None:
            _logger.warning("la_objlim for lacosmic not defined, assuming la_objlim=5.0")
            la_objlim = 5.0
        else:
            _logger.debug("la_objlim for lacosmic: %f", la_objlim)
        # Check la_satlevel
        if la_satlevel is None:
            _logger.warning("la_satlevel for lacosmic not defined, assuming la_satlevel=None")
        else:
            _logger.debug("la_satlevel for lacosmic: %f", la_satlevel)
        # Check niter
        if la_niter is None:
            _logger.warning("la_niter for lacosmic not defined, assuming la_niter=4")
            la_niter = 4
        else:
            _logger.debug("la_niter for lacosmic: %d", la_niter)
        # Check la_sepmed
        if la_sepmed is None:
            _logger.warning("la_sepmed for lacosmic not defined, assuming la_sepmed=True")
            la_sepmed = True
        else:
            _logger.debug("la_sepmed for lacosmic: %s", str(la_sepmed))
        # Check la_cleantype
        if la_cleantype is None:
            raise ValueError("la_cleantype for lacosmic must be provided.")
        elif isinstance(la_cleantype, str):
            if la_cleantype not in VALID_LACOSMIC_CLEANTYPE:
                raise ValueError(f"la_cleantype must be one of {VALID_LACOSMIC_CLEANTYPE} or 'none'.")
        else:
            raise TypeError("la_cleantype must be a string.")
        # Check la_fsmode
        if la_fsmode not in ['median', 'convolve']:
            raise ValueError("la_fsmode must be 'median' or 'convolve'.")
        else:
            _logger.debug("la_fsmode for lacosmic: %s", la_fsmode)
        # Check la_psfmodel
        if la_psfmodel not in ['gauss', 'moffat', 'gaussx', 'gaussy', 'gaussxy']:
            raise ValueError("la_psfmodel must be 'gauss', 'moffat', 'gaussx', 'gaussy', or 'gaussxy'.")
        else:
            _logger.debug("la_psfmodel for lacosmic: %s", la_psfmodel)
        # Check la_psffwhm_x, la_psffwhm_y, la_psfsize
        if la_fsmode == 'convolve':
            if la_psffwhm_x is None or la_psffwhm_y is None or la_psfsize is None:
                raise ValueError("For la_fsmode='convolve', "
                                 "la_psffwhm_x, la_psffwhm_y, and la_psfsize must be provided.")
            else:
                _logger.debug("la_psffwhm_x for lacosmic: %f", la_psffwhm_x)
                _logger.debug("la_psffwhm_y for lacosmic: %f", la_psffwhm_y)
                _logger.debug("la_psfsize for lacosmic: %d", la_psfsize)
            if la_psfsize % 2 == 0 or la_psfsize < 3:
                raise ValueError("la_psfsize must be an odd integer >= 3.")
        # Check la_psfbeta
        if la_psfmodel == 'moffat':
            if la_psfbeta is None:
                raise ValueError("For la_psfmodel='moffat', la_psfbeta must be provided.")
            else:
                _logger.debug("la_psfbeta for lacosmic: %f", la_psfbeta)
        # Set la_verbose
        current_logging_level = logging.getLogger().getEffectiveLevel()
        if current_logging_level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            la_verbose = False
        _logger.debug("la_verbose for lacosmic: %s", str(la_verbose))
        # Define dictionary with the parameters for cosmicray_lacosmic() function
        # Note: the "pssl" parameter is not used here because it was deprecated
        # in version 2.3.0 and will be removed in a future version.
        # The "pssl" keyword will be removed in ccdproc 3.0.
        # Use "inbkg" instead to have astroscrappy temporarily remove the background during processing.
        dict_la_params = {
            'gain': gain_scalar,
            'readnoise': rnoise_scalar,
            'sigclip': la_sigclip,
            'sigfrac': la_sigfrac,
            'objlim': la_objlim,
            'satlevel': la_satlevel * gain_scalar if la_satlevel is not None else None,  # in electrons!
            'niter': la_niter,
            'sepmed': la_sepmed,
            'cleantype': la_cleantype,
            'fsmode': la_fsmode,
            'psfmodel': la_psfmodel,
            'psffwhm': None,
            'psfsize': la_psfsize,
            'psfbeta': la_psfbeta,
            'verbose': la_verbose,
            'psfk': None,
            'inbkg': None,
            'invar': None
        }
        if la_psfmodel in ['gauss', 'moffat']:
            if la_psffwhm_x is None or la_psfsize is None:
                raise ValueError("For la_psfmodel='gauss' or 'moffat', "
                                 "la_psffwhm_x and la_psfsize must be provided.")
            dict_la_params['psffwhm'] = (la_psffwhm_x + la_psffwhm_y) / 2.0  # average FWHM
        elif la_psfmodel == 'gaussx':
            if la_psffwhm_x is None or la_psfsize is None:
                raise ValueError("For la_psfmodel='gaussx', "
                                 "la_psffwhm_x and la_psfsize must be provided.")
            dict_la_params['psffwhm'] = la_psffwhm_x
        elif la_psfmodel == 'gaussy':
            if la_psffwhm_y is None or la_psfsize is None:
                raise ValueError("For la_psfmodel='gaussy', "
                                 "la_psffwhm_y and la_psfsize must be provided.")
            dict_la_params['psffwhm'] = la_psffwhm_y
        elif la_psfmodel == 'gaussxy':
            dict_la_params['psffwhm'] = None  # not used in this case
            dict_la_params['psfk'] = gausskernel2d_elliptical(
                fwhm_x=la_psffwhm_x,
                fwhm_y=la_psffwhm_y,
                kernsize=la_psfsize
            )
        else:
            raise ValueError("la_psfmodel must be 'gauss', 'moffat', 'gaussx', 'gaussy', or 'gaussxy'.")

    _logger.info("dtype for output arrays: %s", dtype)
    _logger.info("dilation factor: %d", dilation)
    _logger.info("verify cosmic-ray detection: %s", verify_cr)
    _logger.info("semiwindow size for plotting coincident cosmic-ray pixels: %d", semiwindow)
    _logger.info("maximum number of coincident cosmic-ray pixels to plot: %d", maxplots)
    _logger.info("color scale for plots: %s", color_scale)

    if la_verbose:
        for key in dict_la_params.keys():
            if key == 'psfk':
                if dict_la_params[key] is None:
                    _logger.info("%s for lacosmic: None", key)
                else:
                    _logger.info("%s for lacosmic: array with shape %s", key, str(dict_la_params[key].shape))
            else:
                _logger.info("%s for lacosmic: %s", key, str(dict_la_params[key]))

    if rich_configured:
        _logger.info("[green]" + "-" * 79 + "[/green]")
        _logger.info("starting cosmic ray detection in [magenta]median2d[/magenta] image...")
    else:
        _logger.info("-" * 73)
        _logger.info("starting cosmic ray detection in median2d image...")

    if crmethod in ['lacosmic', 'mm_lacosmic']:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using the
        # Laplacian edge detection method from ccdproc. This only works if gain and
        # rnoise are constant values (scalars).
        # ---------------------------------------------------------------------
        _logger.info(f"detecting cosmic rays in median2d image using {rlabel_lacosmic}...")
        if gain_scalar is None or rnoise_scalar is None:
            raise ValueError("gain and rnoise must be constant values (scalars) when using crmethod='lacosmic'.")
        median2d_lacosmic, flag_la = decorated_cosmicray_lacosmic(
            ccd=median2d,
            **{key: value for key, value in dict_la_params.items() if value is not None}
        )
        _logger.info("pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
                     rlabel_lacosmic, np.sum(flag_la), np.sum(flag_la) / flag_la.size * 100)
        flag_la = np.logical_and(flag_la, bool_to_be_cleaned)
        flag_la = flag_la.flatten()
        if crmethod == 'lacosmic':
            xplot_boundary = None
            yplot_boundary = None
            mm_threshold = None
            flag_sb = np.zeros_like(flag_la, dtype=bool)
    else:
        median2d_lacosmic = None

    if crmethod in ['mmcosmic', 'mm_lacosmic']:
        # ---------------------------------------------------------------------
        # Detect cosmic rays in the median2d image using the numerically
        # derived boundary.
        # ---------------------------------------------------------------------
        # Define mm_fixed_points_in_boundary
        _logger.info("detecting cosmic rays in median2d image using %s...", rlabel_mmcosmic)
        if isinstance(mm_fixed_points_in_boundary, str):
            if mm_fixed_points_in_boundary.lower() == 'none':
                mm_fixed_points_in_boundary = None
        if mm_fixed_points_in_boundary is None:
            if mm_boundary_fit == 'piecewise':
                raise ValueError("For mm_boundary_fit='piecewise', "
                                 "mm_fixed_points_in_boundary must be provided.")
        else:
            mm_fixed_points_in_boundary = list(eval(str(mm_fixed_points_in_boundary)))
            x_mm_fixed_points_in_boundary = []
            y_mm_fixed_points_in_boundary = []
            w_mm_fixed_points_in_boundary = []
            for item in mm_fixed_points_in_boundary:
                if not (isinstance(item, (list, tuple)) and len(item) in [2, 3]):
                    raise ValueError("Each item in mm_fixed_points_in_boundary must be a list or tuple of "
                                     "2 or 3 elements: (x, y) or (x, y, weight).")
                if not all_valid_numbers(item):
                    raise ValueError(f"All elements in mm_fixed_points_in_boundary={mm_fixed_points_in_boundary} "
                                     "must be valid numbers.")
                if len(item) == 2:
                    x_mm_fixed_points_in_boundary.append(float(item[0]))
                    y_mm_fixed_points_in_boundary.append(float(item[1]))
                    w_mm_fixed_points_in_boundary.append(10000)
                else:
                    x_mm_fixed_points_in_boundary.append(float(item[0]))
                    y_mm_fixed_points_in_boundary.append(float(item[1]))
                    w_mm_fixed_points_in_boundary.append(float(item[2]))
            x_mm_fixed_points_in_boundary = np.array(x_mm_fixed_points_in_boundary, dtype=float)
            y_mm_fixed_points_in_boundary = np.array(y_mm_fixed_points_in_boundary, dtype=float)
            w_mm_fixed_points_in_boundary = np.array(w_mm_fixed_points_in_boundary, dtype=float)

        if mm_boundary_fit is None:
            raise ValueError(f"mm_boundary_fit is None and must be one of {VALID_BOUNDARY_FITS}.")
        elif mm_boundary_fit not in VALID_BOUNDARY_FITS:
            raise ValueError(f"Invalid mm_boundary_fit: {mm_boundary_fit}. Valid options are {VALID_BOUNDARY_FITS}.")
        if mm_boundary_fit == 'piecewise':
            if mm_fixed_points_in_boundary is None:
                raise ValueError("For mm_boundary_fit='piecewise', "
                                 "mm_fixed_points_in_boundary must be provided.")
            elif len(x_mm_fixed_points_in_boundary) < 2:
                raise ValueError("For mm_boundary_fit='piecewise', "
                                 "at least two fixed points must be provided in mm_fixed_points_in_boundary.")

        # Compute offsets between each single exposure and the median image
        if isinstance(mm_xy_offsets, str):
            if mm_xy_offsets.lower() == 'none':
                mm_xy_offsets = None
        if mm_xy_offsets is not None and mm_crosscorr_region is not None:
            raise ValueError("You can only provide one of mm_xy_offsets or mm_crosscorr_region, not both.")
        shift_images = False
        if mm_xy_offsets is not None:
            if isinstance(mm_xy_offsets, str):
                mm_xy_offsets = ast.literal_eval(mm_xy_offsets)
            if isinstance(mm_xy_offsets, list):
                if len(mm_xy_offsets) != num_images:
                    raise ValueError(f"mm_xy_offsets must have the same length as the number of images ({num_images}).")
                for offset in mm_xy_offsets:
                    if (not isinstance(offset, (list, tuple)) or len(offset) != 2 or
                            not all_valid_numbers(offset)):
                        raise ValueError(f"Invalid offset in mm_xy_offsets: {offset}. "
                                         "Each offset must be a tuple or list of two numbers (x_offset, y_offset).")
                list_yx_offsets = [(float(offset[1]), float(offset[0])) for offset in mm_xy_offsets]
                for i, yx_offsets in enumerate(list_yx_offsets):
                    _logger.info("provided offsets for image %d: y=%+f, x=%+f", i+1, yx_offsets[0], yx_offsets[1])
            else:
                raise TypeError(f"Invalid type for mm_xy_offsets: {type(mm_xy_offsets)}. "
                                "Must be list of [x_offset, y_offset)].")
            shift_images = True
        else:
            if isinstance(mm_crosscorr_region, str):
                if mm_crosscorr_region.lower() == 'none':
                    mm_crosscorr_region = None
                else:
                    mm_crosscorr_region = ast.literal_eval(mm_crosscorr_region)
            if isinstance(mm_crosscorr_region, list):
                all_integers = all(isinstance(val, int) for val in mm_crosscorr_region)
                if not all_integers:
                    raise TypeError(f"Invalid mm_crosscorr_region: {mm_crosscorr_region}. "
                                    "All elements must be integers.")
                if len(mm_crosscorr_region) != 4:
                    raise ValueError(f"Invalid length for mm_crosscorr_region: {mm_crosscorr_region}. "
                                     "Must be a list of 4 integers [xmin, xmax, ymin, ymax].")
                dumreg = f"[{mm_crosscorr_region[0]}:{mm_crosscorr_region[1]}, " + \
                         f"{mm_crosscorr_region[2]}:{mm_crosscorr_region[3]}]"
                _logger.debug("defined mm_crosscorr_region: %s", dumreg)
                crossregion = tea.SliceRegion2D(dumreg, mode='fits', naxis1=naxis1, naxis2=naxis2)
                if crossregion.area() < 100:
                    raise ValueError("The area of mm_crosscorr_region must be at least 100 pixels.")
                shift_images = True
            elif mm_crosscorr_region is None:
                crossregion = None
            else:
                raise TypeError(f"Invalid type for mm_crosscorr_region: {type(mm_crosscorr_region)}. "
                                "Must be list of 4 integers or None.")
            list_yx_offsets = []
            for i in range(num_images):
                if crossregion is None:
                    list_yx_offsets.append((0.0, 0.0))
                else:
                    reference_image = median2d[crossregion.python]
                    moving_image = image3d[i][crossregion.python]
                    _logger.info("computing offsets for image %d using cross-correlation...", i+1)
                    yx_offsets, _, _ = phase_cross_correlation(
                        reference_image=reference_image,
                        moving_image=moving_image,
                        upsample_factor=100,
                        normalization=None  # use None to avoid artifacts with images with many cosmic rays
                    )
                    yx_offsets[0] *= -1  # invert sign
                    yx_offsets[1] *= -1  # invert sign
                    _logger.info("offsets for image %d: y=%+f, x=%+f", i+1, yx_offsets[0], yx_offsets[1])

                    def on_key_cross(event):
                        if event.key == 'x':
                            _logger.info("Exiting program as per user request ('x' key pressed).")
                            plt.close(fig)
                            sys.exit(0)

                    fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6.4*1.5, 4.8*1.5))
                    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_cross(event))
                    axarr = axarr.flatten()
                    vmin = np.min(reference_image)
                    vmax = np.max(reference_image)
                    extent = [crossregion.fits[0].start - 0.5, crossregion.fits[0].stop + 0.5,
                              crossregion.fits[1].start - 0.5, crossregion.fits[1].stop + 0.5]
                    tea.imshow(fig, axarr[0], reference_image, vmin=vmin, vmax=vmax,
                               extent=extent, aspect='auto', title='Median')
                    tea.imshow(fig, axarr[1], moving_image, vmin=vmin, vmax=vmax,
                               extent=extent, aspect='auto', title=f'Image {i+1}')
                    shifted_image2d = shift_image2d(
                        moving_image,
                        xoffset=-yx_offsets[1],
                        yoffset=-yx_offsets[0],
                        resampling=2
                    )
                    dumdiff1 = reference_image - moving_image
                    dumdiff2 = reference_image - shifted_image2d
                    vmin = np.percentile(dumdiff1, 5)
                    vmax = np.percentile(dumdiff2, 95)
                    tea.imshow(fig, axarr[2], dumdiff1, vmin=vmin, vmax=vmax,
                               extent=extent, aspect='auto', title=f'Median - Image {i+1}')
                    tea.imshow(fig, axarr[3], dumdiff2, vmin=vmin, vmax=vmax,
                               extent=extent, aspect='auto', title=f'Median - Shifted Image {i+1}')
                    plt.tight_layout()
                    png_filename = f'xyoffset_crosscorr_{i+1}.png'
                    _logger.info(f"saving {png_filename}")
                    plt.savefig(png_filename, dpi=150)
                    if interactive:
                        _logger.info("Entering interactive mode (press 'q' to close figure, 'x' to quit program)")
                        plt.show()
                    plt.close(fig)
                    list_yx_offsets.append(yx_offsets)

        # Estimate limits for the diagnostic plot
        rng = np.random.default_rng(mm_seed)  # Random number generator for reproducibility
        xdiag_min, xdiag_max, ydiag_min, ydiag_max = estimate_diagnostic_limits(
            rng=rng,
            gain=np.median(gain),  # Use median value to simplify the computation
            rnoise=np.median(rnoise),  # Use median value to simplify the computation
            maxvalue=np.max(min2d),
            num_images=num_images,
            npixels=10000
        )
        if np.min(xplot) < xdiag_min:
            xdiag_min = np.min(xplot)
        if np.max(xplot) > xdiag_max:
            xdiag_max = np.max(xplot)
        if mm_fixed_points_in_boundary is not None:
            if np.max(y_mm_fixed_points_in_boundary) > ydiag_max:
                ydiag_max = np.max(y_mm_fixed_points_in_boundary)
        if shift_images:
            ydiag_max *= 4.0  # Add 300% margin to the maximum y limit
        else:
            ydiag_max *= 2.0  # Add 100% margin to the maximum y limit
        _logger.debug("xdiag_min=%f", xdiag_min)
        _logger.debug("ydiag_min=%f", ydiag_min)
        _logger.debug("xdiag_max=%f", xdiag_max)
        _logger.debug("ydiag_max=%f", ydiag_max)

        # Define binning for the diagnostic plot
        nbins_xdiag = 100
        nbins_ydiag = 100
        bins_xdiag = np.linspace(xdiag_min, xdiag_max, nbins_xdiag + 1)
        bins_ydiag = np.linspace(0, ydiag_max, nbins_ydiag + 1)
        xcbins = (bins_xdiag[:-1] + bins_xdiag[1:]) / 2
        ycbins = (bins_ydiag[:-1] + bins_ydiag[1:]) / 2

        # Create a 2D histogram for the diagnostic plot, using
        # integers to avoid rounding errors
        hist2d_accummulated = np.zeros((nbins_ydiag, nbins_xdiag), dtype=int)
        lam = median2d.copy()
        lam[lam < 0] = 0  # Avoid negative values
        lam3d = np.zeros((num_images, naxis2, naxis1))
        if apply_flux_factor_to == 'original':
            flux_factor_for_simulated = np.ones(num_images, dtype=float)
        elif apply_flux_factor_to == 'simulated':
            flux_factor_for_simulated = flux_factor
        else:
            raise ValueError(f"Invalid apply_flux_factor_to: {apply_flux_factor_to}. "
                             "Valid options are 'original' and 'simulated'.")
        _logger.info("flux factor for simulated images: %s", str(flux_factor_for_simulated))
        if not shift_images:
            _logger.info("assuming no offsets between images")
            for i in range(num_images):
                lam3d[i] = lam / flux_factor_for_simulated[i]
        else:
            _logger.info("xy-shifting median2d to speed up simulations...")
            for i in range(num_images):
                _logger.info("shifted image %d/%d -> delta_y=%+f, delta_x=%+f",
                             i + 1, num_images, list_yx_offsets[i][0], list_yx_offsets[i][1])
                # apply offsets to the median image to simulate the expected individual exposures
                lam3d[i] = shift_image2d(lam,
                                         xoffset=list_yx_offsets[i][1],
                                         yoffset=list_yx_offsets[i][0],
                                         resampling=2) / flux_factor_for_simulated[i]
                # replace any NaN values introduced by the shift with zeros
                lam3d[i] = np.nan_to_num(lam3d[i], nan=0.0)
                # replace any negative values with zeros
                lam3d[i][lam3d[i] < 0] = 0.0
        _logger.info("computing simulated 2D histogram...")
        for k in range(mm_nsimulations):
            time_ini = datetime.now()
            image3d_simul = np.zeros((num_images, naxis2, naxis1))
            for i in range(num_images):
                # convert from ADU to electrons to apply Poisson noise, and then back to ADU
                image3d_simul[i] = rng.poisson(lam=lam3d[i] * gain).astype(float) / gain
                # add readout noise in ADU
                image3d_simul[i] += rng.normal(loc=0, scale=rnoise)
            min2d_simul = np.min(image3d_simul, axis=0)
            median2d_simul = np.median(image3d_simul, axis=0)
            xplot_simul = min2d_simul.flatten()
            yplot_simul = median2d_simul.flatten() - min2d_simul.flatten()
            hist2d, edges = np.histogramdd(
                sample=(yplot_simul, xplot_simul),
                bins=(bins_ydiag, bins_xdiag)
            )
            hist2d_accummulated += hist2d.astype(int)
            time_end = datetime.now()
            _logger.info("simulation %d/%d, time elapsed: %s", k + 1, mm_nsimulations, time_end - time_ini)
        # Average the histogram over the number of simulations
        hist2d_accummulated = hist2d_accummulated.astype(float) / mm_nsimulations
        vmin = np.min(hist2d_accummulated[hist2d_accummulated > 0])
        if vmin == 0:
            vmin = 1
        vmax = np.max(hist2d_accummulated)
        cmap1 = plt.get_cmap('cividis_r')
        cmap2 = plt.get_cmap('viridis')
        n_colors = 256
        n_colors2 = int((np.log10(vmax) - np.log10(1.0)) / (np.log10(vmax) - np.log10(vmin)) * n_colors)
        n_colors2 += 1
        if n_colors2 > n_colors:
            n_colors2 = n_colors
        if n_colors2 < n_colors:
            n_colors1 = n_colors - n_colors2
        else:
            n_colors1 = 0
        colors1 = cmap1(np.linspace(0, 1, n_colors1))
        colors2 = cmap2(np.linspace(0, 1, n_colors2))
        combined_colors = np.vstack((colors1, colors2))
        combined_cmap = LinearSegmentedColormap.from_list('combined_cmap', combined_colors)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        def on_key_2dhist(event):
            if event.key == 'x':
                _logger.info("Exiting program as per user request ('x' key pressed).")
                plt.close(fig)
                sys.exit(0)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.1, 5.5))
        fig.canvas.mpl_connect('key_press_event', lambda event: on_key_2dhist(event))
        # Display 2D histogram of the simulated data
        extent = [bins_xdiag[0], bins_xdiag[-1], bins_ydiag[0], bins_ydiag[-1]]
        tea.imshow(fig, ax1, hist2d_accummulated, norm=norm, extent=extent,
                   aspect='auto', cblabel='Number of pixels', cmap=combined_cmap)
        # Display 2D histogram of the original data
        hist2d_original, edges = np.histogramdd(
            sample=(yplot, xplot),
            bins=(bins_ydiag, bins_xdiag)
        )
        tea.imshow(fig, ax2, hist2d_original, norm=norm, extent=extent,
                   aspect='auto', cblabel='Number of pixels', cmap=combined_cmap)

        # Determine the detection boundary for coincident cosmic-ray detection
        _logger.info("computing numerical boundary for coincident cosmic-ray detection...")
        xboundary = []
        yboundary = []
        for i in range(nbins_xdiag):
            fsum = np.sum(hist2d_accummulated[:, i])
            if fsum > 0:
                pdensity = hist2d_accummulated[:, i] / fsum
                perc = (1 - (1 / mm_nsimulations) / fsum)
                p = np.interp(perc, np.cumsum(pdensity), np.arange(nbins_ydiag))
                xboundary.append(xcbins[i])
                yboundary.append(ycbins[int(p + 0.5)])
        xboundary = np.array(xboundary)
        yboundary = np.array(yboundary)
        ax1.plot(xboundary, yboundary, 'r+')
        boundaryfit = None  # avoid flake8 warning
        if mm_boundary_fit == 'spline':
            for iterboundary in range(mm_niter_boundary_extension + 1):
                wboundary = np.ones_like(xboundary, dtype=float)
                if iterboundary == 0:
                    label = 'initial spline fit'
                else:
                    wboundary[yboundary > boundaryfit(xboundary)] = mm_weight_boundary_extension**iterboundary
                    label = f'Iteration {iterboundary}'
                if mm_fixed_points_in_boundary is None:
                    xboundary_fit = xboundary
                    yboundary_fit = yboundary
                    wboundary_fit = wboundary
                else:
                    wboundary_max = np.max(wboundary)
                    xboundary_fit = np.concatenate((xboundary, x_mm_fixed_points_in_boundary))
                    yboundary_fit = np.concatenate((yboundary, y_mm_fixed_points_in_boundary))
                    wboundary_fit = np.concatenate((wboundary, w_mm_fixed_points_in_boundary * wboundary_max))
                isort = np.argsort(xboundary_fit)
                boundaryfit, knots = spline_positive_derivative(
                    x=xboundary_fit[isort],
                    y=yboundary_fit[isort],
                    w=wboundary_fit[isort],
                    n_total_knots=mm_knots_splfit,
                )
                ydum = boundaryfit(xcbins)
                ydum[xcbins < knots[0]] = boundaryfit(knots[0])
                ydum[xcbins > knots[-1]] = boundaryfit(knots[-1])
                ax1.plot(xcbins, ydum, '-', color=f'C{iterboundary}', label=label)
                ax1.plot(knots, boundaryfit(knots), 'o', color=f'C{iterboundary}', markersize=4)
        elif mm_boundary_fit == 'piecewise':
            boundaryfit = define_piecewise_linear_function(
                xarray=x_mm_fixed_points_in_boundary,
                yarray=y_mm_fixed_points_in_boundary
            )
            ax1.plot(xcbins, boundaryfit(xcbins), 'r-', label='Piecewise linear fit')
        else:
            raise ValueError(f"Invalid mm_boundary_fit: {mm_boundary_fit}. Valid options are {VALID_BOUNDARY_FITS}.")

        if mm_threshold is None:
            # Use the minimum value of the boundary as the mm_threshold
            mm_threshold = np.min(yplot_boundary)
            _logger.info("updated mm_threshold for cosmic-ray detection: %f", mm_threshold)

        # Apply the criterium to detect coincident cosmic-ray pixels
        flag1 = yplot > boundaryfit(xplot)
        flag2 = yplot > mm_threshold
        flag_sb = np.logical_and(flag1, flag2)
        flag3 = max2d.flatten() > mm_minimum_max2d_rnoise * rnoise.flatten()
        flag_sb = np.logical_and(flag_sb, flag3)
        flag_sb = np.logical_and(flag_sb, bool_to_be_cleaned.flatten())
        _logger.info("pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
                     rlabel_mmcosmic, np.sum(flag_sb), np.sum(flag_sb) / flag_sb.size * 100)
        if crmethod == 'mmcosmic':
            flag_la = np.zeros_like(flag_sb, dtype=bool)

        # Plot the results
        if mm_fixed_points_in_boundary is not None:
            ax1.plot(x_mm_fixed_points_in_boundary, y_mm_fixed_points_in_boundary, 'ms', markersize=6, alpha=0.5,
                     label='Fixed points')
        ax1.set_xlabel(r'min2d $-$ bias')
        ax1.set_ylabel(r'median2d $-$ min2d')
        ax1.set_title(f'Simulated data (mm_nsimulations = {mm_nsimulations})')
        if mm_niter_boundary_extension > 1:
            ax1.legend(loc=4)
        xplot_boundary = np.linspace(xdiag_min, xdiag_max, 100)
        yplot_boundary = boundaryfit(xplot_boundary)
        if mm_boundary_fit == 'spline':
            # For spline fit, force the boundary to be constant outside the knots
            yplot_boundary[xplot_boundary < knots[0]] = boundaryfit(knots[0])
            yplot_boundary[xplot_boundary > knots[-1]] = boundaryfit(knots[-1])
        ax2.plot(xplot_boundary, yplot_boundary, 'r-', label='Detection boundary')
        if mm_fixed_points_in_boundary is not None:
            ax2.plot(x_mm_fixed_points_in_boundary, y_mm_fixed_points_in_boundary, 'ms', markersize=6, alpha=0.5,
                     label='Fixed points')
        ax2.set_xlim(xdiag_min, xdiag_max)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel(ax1.get_xlabel())
        ax2.set_ylabel(ax1.get_ylabel())
        ax2.set_title('Original data')
        ax2.legend(loc=4)
        plt.tight_layout()
        png_filename = 'diagnostic_histogram2d.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close figure, 'x' to quit program)")
            plt.show()
        plt.close(fig)

    # Define the final cosmic ray flag
    if flag_la is None and flag_sb is None:
        raise RuntimeError("Both flag_la and flag_sb are None. This should never happen.")
    elif flag_la is None:
        flag = flag_sb
        flag_integer = 2 * flag_sb.astype(np.uint8)
    elif flag_sb is None:
        flag = flag_la
        flag_integer = 3 * flag_la.astype(np.uint8)
    else:
        # Combine the flags from lacosmic and mmcosmic
        flag = np.logical_or(flag_la, flag_sb)
        flag_integer = 2 * flag_sb.astype(np.uint8) + 3 * flag_la.astype(np.uint8)
        sdum = str(np.sum(flag))
        cdum = f"{np.sum(flag):{len(sdum)}d}"
        _logger.info("pixels flagged as cosmic rays by "
                     "%s or  %s: %s (%08.4f%%)", rlabel_lacosmic, rlabel_mmcosmic, cdum,
                     np.sum(flag) / flag.size * 100)
        cdum = f"{np.sum(flag_integer == 3):{len(sdum)}d}"
        _logger.info("pixels flagged as cosmic rays by "
                     "%s only........: %s (%08.4f%%)", rlabel_lacosmic, cdum,
                     np.sum(flag_integer == 3) / flag.size * 100)
        cdum = f"{np.sum((flag_integer == 2)):{len(sdum)}d}"
        _logger.info("pixels flagged as cosmic rays by "
                     "%s only........: %s (%08.4f%%)", rlabel_mmcosmic, cdum,
                     np.sum(flag_integer == 2) / flag.size * 100)
        cdum = f"{np.sum((flag_integer == 5)):{len(sdum)}d}"
        _logger.info("pixels flagged as cosmic rays by "
                     "%s and %s: %s (%08.4f%%)", rlabel_lacosmic, rlabel_mmcosmic, cdum,
                     np.sum((flag_integer == 5)) / flag.size * 100)
    flag = flag.reshape((naxis2, naxis1))
    flag_integer = flag_integer.reshape((naxis2, naxis1))
    flag_integer[flag_integer == 5] = 4  # pixels flagged by both methods are set to 4

    # Show diagnostic plot for the cosmic ray detection
    _logger.info("generating diagnostic plot for MEDIANCR...")
    ylabel = r'median2d $-$ min2d'
    diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag_la, flag_sb,
                    mm_threshold, ylabel, interactive,
                    target2d=median2d, target2d_name='median2d',
                    min2d=min2d, mean2d=mean2d, image3d=image3d,
                    _logger=_logger, png_filename='diagnostic_mediancr.png')

    # Check if any cosmic ray was detected
    if not np.any(flag):
        _logger.info("no coincident cosmic-ray pixels detected.")
        mask_mediancr = np.zeros_like(median2d, dtype=bool)
    else:
        _logger.info("coincident cosmic-ray pixels detected...")
        # Use the integer version of the flag for dilation
        if dilation > 0:
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            sdum = str(np.sum(flag_integer_dilated > 0))
            cdum = f"{np.sum(flag_integer > 0):{len(sdum)}d}"
            _logger.info("before dilation: %s pixels flagged as coincident cosmic-ray pixels", cdum)
            cdum = f"{np.sum(flag_integer_dilated > 0):{len(sdum)}d}"
            _logger.info("after dilation : %s pixels flagged as coincident cosmic-ray pixels", cdum)
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as coincident cosmic-ray pixels",
                         np.sum(flag_integer > 0))
        # Set the pixels that were originally flagged as cosmic rays
        # to the integer value before dilation (this is to distinguish them
        # from the pixels that were dilated,which will be set to 1)
        flag_integer_dilated[flag] = flag_integer[flag]
        # Compute mask
        mask_mediancr = flag_integer_dilated > 0
        # Fix the median2d array by replacing the flagged pixels with the minimum value
        # of the corresponding pixel in the input arrays
        median2d_corrected = median2d.copy()
        if use_lamedian:
            median2d_corrected[mask_mediancr] = median2d_lacosmic[mask_mediancr]
        else:
            median2d_corrected[mask_mediancr] = min2d[mask_mediancr]
        # Label the connected pixels as individual cosmic rays
        labels_cr, number_cr = ndimage.label(flag_integer_dilated > 0)
        _logger.info("number of grouped cosmic-ray pixels detected: %d", number_cr)
        display_detected_cr(
            num_images=num_images,
            image3d=image3d,
            median2d=median2d,
            median2d_corrected=median2d_corrected,
            flag_integer_dilated=flag_integer_dilated,
            labels_cr=labels_cr,
            number_cr=number_cr,
            mask_mediancr=mask_mediancr,
            list_mask_single_exposures=None,
            mask_all_singles=None,
            semiwindow=semiwindow,
            maxplots=maxplots,
            verify_cr=verify_cr,
            color_scale=color_scale,
            _logger=_logger
        )

    # Generate list of HDUs with masks
    hdu_mediancr = fits.ImageHDU(mask_mediancr.astype(np.uint8), name='MEDIANCR')
    list_hdu_masks = [hdu_mediancr]

    # Apply the same algorithm but now with mean2d and with each individual array
    for i, target2d in enumerate([mean2d] + list_arrays):
        if i == 0:
            target2d_name = 'mean2d'
        else:
            target2d_name = f'single exposure #{i}'
        if rich_configured:
            _logger.info("[green]" + "-" * 79 + "[/green]")
            _logger.info(f"starting cosmic ray detection in [magenta]{target2d_name}[/magenta] image...")
        else:
            _logger.info("-" * 73)
            _logger.info(f"starting cosmic ray detection in {target2d_name} image...")
        if crmethod in ['lacosmic', 'mm_lacosmic']:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_lacosmic}...")
            array_lacosmic, flag_la = decorated_cosmicray_lacosmic(
                ccd=target2d,
                **{key: value for key, value in dict_la_params.items() if value is not None}
            )
            flag_la = np.logical_and(flag_la, bool_to_be_cleaned)
            flag_la = flag_la.flatten()
            if crmethod == 'lacosmic':
                xplot_boundary = None
                yplot_boundary = None
                mm_threshold = None
                flag_sb = np.zeros_like(flag_la, dtype=bool)
        if crmethod in ['mmcosmic', 'mm_lacosmic']:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_mmcosmic}...")
            xplot = min2d.flatten()
            yplot = target2d.flatten() - min2d.flatten()
            flag1 = yplot > boundaryfit(xplot)
            flag2 = yplot > mm_threshold
            flag_sb = np.logical_and(flag1, flag2)
            flag3 = max2d.flatten() > mm_minimum_max2d_rnoise * rnoise.flatten()
            flag_sb = np.logical_and(flag_sb, flag3)
            flag_sb = np.logical_and(flag_sb, bool_to_be_cleaned.flatten())
            if crmethod == 'mmcosmic':
                flag_la = np.zeros_like(flag_sb, dtype=bool)
        # For the mean2d mask, force the flag to be True if the pixel
        # was flagged as a coincident cosmic-ray pixel when using the median2d array
        # (this is to ensure that all pixels flagged in MEDIANCR are also
        # flagged in MEANCRT)
        if i == 0:
            flag_la = np.logical_or(flag_la, list_hdu_masks[0].data.astype(bool).flatten())
            flag_sb = np.logical_or(flag_sb, list_hdu_masks[0].data.astype(bool).flatten())
        # For the individual array masks, force the flag to be True if the pixel
        # is flagged both in the individual exposure and in the mean2d array
        if i > 0:
            flag_la = np.logical_and(flag_la, list_hdu_masks[1].data.astype(bool).flatten())
            flag_sb = np.logical_and(flag_sb, list_hdu_masks[1].data.astype(bool).flatten())
        sflag_la = str(np.sum(flag_la))
        sflag_sb = str(np.sum(flag_sb))
        smax = max(len(sflag_la), len(sflag_sb))
        _logger.info("pixels flagged as cosmic rays by "
                     "%s: %s (%08.4f%%)", rlabel_lacosmic, f"{np.sum(flag_la):{smax}d}",
                     np.sum(flag_la) / flag_la.size * 100)
        _logger.info("pixels flagged as cosmic rays by "
                     "%s: %s (%08.4f%%)", rlabel_mmcosmic, f"{np.sum(flag_sb):{smax}d}",
                     np.sum(flag_sb) / flag_sb.size * 100)
        if i == 0:
            _logger.info("generating diagnostic plot for MEANCRT...")
            png_filename = 'diagnostic_meancr.png'
            ylabel = r'mean2d $-$ min2d'
        else:
            _logger.info(f"generating diagnostic plot for CRMASK{i}...")
            png_filename = f'diagnostic_crmask{i}.png'
            ylabel = f'array{i}' + r' $-$ min2d'
        interactive_eff = interactive and debug
        diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag_la, flag_sb,
                        mm_threshold, ylabel, interactive_eff,
                        target2d=target2d, target2d_name=target2d_name,
                        min2d=min2d, mean2d=mean2d, image3d=image3d,
                        _logger=_logger, png_filename=png_filename)
        flag = np.logical_or(flag_la, flag_sb)
        flag = flag.reshape((naxis2, naxis1))
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            sdum = str(np.sum(flag_integer_dilated))
            cdum = f"{np.sum(flag_integer):{len(sdum)}d}"
            _logger.info("before dilation: %s pixels flagged as cosmic rays", cdum)
            cdum = f"{np.sum(flag_integer_dilated):{len(sdum)}d}"
            _logger.info("after dilation : %s pixels flagged as cosmic rays", cdum)
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as cosmic rays", np.sum(flag_integer))
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask = flag_integer_dilated > 0
        if i == 0:
            name = 'MEANCRT'
        else:
            name = f'CRMASK{i}'
        hdu_mask = fits.ImageHDU(mask.astype(np.uint8), name=name)
        list_hdu_masks.append(hdu_mask)

    if median2d_lacosmic is not None:
        hdu_median2d_lacosmic = fits.ImageHDU(median2d_lacosmic.astype(np.float32), name='LAMEDIAN')
        list_hdu_masks.append(hdu_median2d_lacosmic)

    # Find problematic cosmic-ray pixels (those masked in all individual CRMASKi)
    mask_all_singles = np.ones((naxis2, naxis1), dtype=bool)
    for hdu in list_hdu_masks[2:]:
        mask_all_singles = np.logical_and(mask_all_singles, hdu.data.astype(bool))
    problematic_pixels = np.argwhere(mask_all_singles)
    if rich_configured:
        _logger.info("[green]" + "-" * 79 + "[/green]")
    else:
        _logger.info("-" * 73)
    _logger.info("number of problematic cosmic-ray pixels masked in all individual CRMASKi: %d",
                 len(problematic_pixels))
    if len(problematic_pixels) > 0:
        # Label the connected problematic pixels as individual problematic cosmic rays
        labels_cr, number_cr = ndimage.label(mask_all_singles)
        _logger.info("number of grouped problematic cosmic-ray pixels: %d", number_cr)
        display_detected_cr(
            num_images=num_images,
            image3d=image3d,
            median2d=median2d,
            median2d_corrected=median2d_corrected,
            flag_integer_dilated=flag_integer_dilated,
            labels_cr=labels_cr,
            number_cr=number_cr,
            mask_mediancr=mask_mediancr,
            list_mask_single_exposures=[hdu.data for hdu in list_hdu_masks[2:2+num_images]],
            mask_all_singles=mask_all_singles,
            semiwindow=semiwindow,
            maxplots=maxplots,
            verify_cr=False,
            color_scale=color_scale,
            _logger=_logger
        )

    # Generate output HDUList with masks
    args = inspect.signature(compute_crmasks).parameters
    if crmethod == 'lacosmic':
        prefix_of_excluded_args = 'mm_'
    elif crmethod == 'mmcosmic':
        prefix_of_excluded_args = 'la_'
    else:
        prefix_of_excluded_args = 'xxx'
    filtered_args = {k: v for k, v in locals().items() if
                     k in args and
                     k not in ['list_arrays'] and
                     k[:3] != prefix_of_excluded_args}
    hdu_primary = fits.PrimaryHDU()
    hdu_primary.header['UUID'] = str(uuid.uuid4())
    for i, fluxf in enumerate(flux_factor):
        hdu_primary.header[f'FLUXF{i+1}'] = fluxf
    hdu_primary.header.add_history(f"CRMasks generated by {__name__}")
    hdu_primary.header.add_history(f"at {datetime.now().isoformat()}")
    for key, value in filtered_args.items():
        if isinstance(value, np.ndarray):
            if np.unique(value).size == 1:
                value = value.flatten()[0]
            elif value.ndim == 1 and len(value) == num_images:
                value = str(value.tolist())
            else:
                value = f'array_shape: {value.shape}'
        elif isinstance(value, list):
            value = str(value)
        hdu_primary.header.add_history(f"- {key} = {value}")
    # Include extension with pixels to be replaced by the median value around them
    # (stored as a binary table with four columns: 'X_pixel', 'Y_pixel', 'X_width', 'Y_width')
    if pixels_to_be_replaced_by_local_median is not None:
        col1 = fits.Column(name='X_pixel', format='K',
                           array=[p[0] for p in pixels_to_be_replaced_by_local_median])
        col2 = fits.Column(name='Y_pixel', format='K',
                           array=[p[1] for p in pixels_to_be_replaced_by_local_median])
        col3 = fits.Column(name='X_width', format='K',
                           array=[p[2] for p in pixels_to_be_replaced_by_local_median])
        col4 = fits.Column(name='Y_width', format='K',
                           array=[p[3] for p in pixels_to_be_replaced_by_local_median])
        hdu_table = fits.BinTableHDU.from_columns([col1, col2, col3, col4], name='RPMEDIAN')
        list_hdu_masks.append(hdu_table)

    hdul_masks = fits.HDUList([hdu_primary] + list_hdu_masks)
    return hdul_masks
