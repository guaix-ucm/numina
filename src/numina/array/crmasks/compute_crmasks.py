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
import logging
import sys
import uuid

from astropy.io import fits
from astropy.table import Table
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from rich.logging import RichHandler
from scipy import ndimage
from skimage.registration import phase_cross_correlation

try:
    import deepCR

    DEEPCR_AVAILABLE = True
except ModuleNotFoundError as e:
    DEEPCR_AVAILABLE = False

try:
    import PyCosmic

    PYCOSMIC_AVAILABLE = True
except ModuleNotFoundError as e:
    PYCOSMIC_AVAILABLE = False

try:
    import cosmic_conn

    CONN_AVAILABLE = True
except ModuleNotFoundError as e:
    CONN_AVAILABLE = False

from numina.array.distortion import shift_image2d
import teareduce as tea

from .all_valid_numbers import all_valid_numbers
from .apply_crmasks import apply_crmasks
from .apply_threshold_cr import apply_threshold_cr
from .compute_flux_factor import compute_flux_factor
from .diagnostic_plot import diagnostic_plot
from .display_detected_cr import display_detected_cr
from .display_hist2d import display_hist2d
from .estimate_diagnostic_limits import estimate_diagnostic_limits
from .execute_conn import execute_conn
from .execute_deepcr import execute_deepcr
from .execute_lacosmic import execute_lacosmic
from .execute_pycosmic import execute_pycosmic
from .gausskernel2d_elliptical import gausskernel2d_elliptical
from .valid_parameters import VALID_CRMETHODS
from .valid_parameters import VALID_LACOSMIC_CLEANTYPE
from .valid_parameters import VALID_BOUNDARY_FITS
from .valid_parameters import DEFAULT_WEIGHT_FIXED_POINTS_IN_BOUNDARY


def compute_crmasks(
    list_arrays,
    gain=None,
    rnoise=None,
    bias=None,
    crmethod="mm_lacosmic",
    use_auxmedian=False,
    flux_factor=None,
    flux_factor_regions=None,
    apply_flux_factor_to=None,
    interactive=True,
    dilation=0,
    regions_to_be_skipped=None,
    pixels_to_be_flagged_as_cr=None,
    pixels_to_be_ignored_as_cr=None,
    pixels_to_be_replaced_by_local_median=None,
    dtype=np.float32,
    verify_cr=False,
    semiwindow=15,
    color_scale="minmax",
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
    la_padwidth=None,
    pc_sigma_det=None,
    pc_rlim=None,
    pc_iterations=None,
    pc_fwhm_gauss_x=None,
    pc_fwhm_gauss_y=None,
    pc_replace_box_x=None,
    pc_replace_box_y=None,
    pc_replace_error=None,
    pc_increase_radius=0,
    pc_verbose=False,
    dc_mask=None,
    dc_threshold=None,
    dc_verbose=False,
    nn_model=None,
    nn_threshold=None,
    nn_verbose=False,
    mm_synthetic=None,
    mm_hist2d_min_neighbors=0,
    mm_ydiag_max=0,
    mm_dilation=0,
    mm_xy_offsets=None,
    mm_crosscorr_region=None,
    mm_boundary_fit=None,
    mm_knots_splfit=3,
    mm_fixed_points_in_boundary=None,
    mm_nsimulations=10,
    mm_niter_boundary_extension=3,
    mm_weight_boundary_extension=10.0,
    mm_threshold_rnoise=0.0,
    mm_minimum_max2d_rnoise=5.0,
    mm_seed=None,
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
        - 'pycosmic': use the PyCosmic algorithm (Husemann et al. 2012).
        - 'deepcr': use the DeepCR algorithm (Zhang & Bloom 2021).
        - 'conn': use a convolutional neural network for cosmic ray detection
        - 'mm_lacosmic': use both methods: 'lacosmic' and the detection
           boundary derived from numerical simulations.
        - 'mm_pycosmic': use both methods: 'pycosmic' and the detection
           boundary derived from numerical simulations.
        - 'mm_deepcr': use both methods: 'deepcr' and the detection
           boundary derived from numerical simulations.
        - 'mm_conn': use both methods: 'conn' and the detection
           boundary derived from numerical simulations.
    use_auxmedian: bool, optional
        If True, use the corrected values from the auxiliary algorithm
        when replacing the cosmic-ray affected pixels in the median
        combined image. This parameter is only used when
        crmethod is 'lacosmic', 'mm_lacosmic', 'pycosmic', 'mm_pycosmic',
        'deepcr', or 'mm_deepcr'.
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
        The sigma clipping threshold. Employed when crmethod='lacosmic'
        or 'mm_lacosmic'.
    la_sigfrac : float
        The fractional detection limit for neighboring pixels.
        Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_objlim : float
        Minimum contrast between Laplacian image and fine structure image.
        Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_satlevel : float
        The saturation level (in ADU) of the detector. Employed when
        crmethod='lacosmic' or 'mm_lacosmic'.
    la_niter : int
        The number of iterations to perform. Employed when
        crmethod='lacosmic' or 'mm_lacosmic'.
    la_sepmed : bool
        If True, use separable median filter instead of the full median filter.
        Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_fsmode : str
        The mode to use for the fine structure image. Valid options are:
        'median' or 'convolve'. Employed when crmethod='lacosmic'
        or 'mm_lacosmic'.
    la_psfmodel : str
        The model to use for the PSF if la_fsmode='convolve'.
        Valid options are:
        - circular kernels: 'gauss' or 'moffat'
        - Gaussian in the x and y directions: 'gaussx' and 'gaussy'
        - elliptical Gaussian: 'gaussxy' (this kernel is not available
          in ccdproc.cosmicray_lacosmic, so it is implemented here)
        Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_psffwhm_x : float
        The full width at half maximum (FWHM, in pixels) of the PSF in
        the x direction. Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_psffwhm_y : float
        The full width at half maximum (FWHM, in pixels) of the PSF
        in the y direction. Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_psfsize : int
        The kernel size to use for the PSF. It must be an odd integer >= 3.
        Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_psfbeta : float
        The beta parameter of the Moffat PSF. It is only used if
        la_psfmodel='moffat'. Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_verbose : bool
        If True, print additional information during the
        execution. Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    la_padwidth : int or None
        The width of the padding to apply to the input image
        when using the lacosmic algorithm. If None, no padding is applied.
        The padding helps to mitigate edge effects during the
        cosmic ray detection. Employed when crmethod='lacosmic' or 'mm_lacosmic'.
    pc_sigma_det : float or None
        The detection limit above the noise.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_rlim : float or None
        The detection threshold.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_iterations : int or None
        The number of iterations to perform.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_fwhm_gauss_x : float or None
        FWHM of the Gaussian smoothing kernel (X direction).
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_fwhm_gauss_y : float or None
        FWHM of the Gaussian smoothing kernel (Y direction).
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_replace_box_x : int or None
        Median box size (X axis) to estimate replacement
        values for cosmic ray affected pixels.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_replace_box_y : int or None
        Median box size (Y axis) to estimate replacement
        values for cosmic ray affected pixels.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_replace_error : float or None
        Error value for bad pixels.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_increase_radius : int, optional
        The radius increase for the neighboring pixels
        to be considered as affected by cosmic rays.
        Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    pc_verbose : bool
        If True, print additional information during the
        execution. Employed when crmethod='pycosmic' or 'mm_pycosmic'.
    dc_mask : str or None
        The instrument/detector mask to use for DeepCR.
        Valid options are 'ACS-WFC' and 'WFC3-UVIS'.
        Employed when crmethod='deepcr' or 'mm_deepcr'.
    dc_threshold : float or None
        The detection threshold for DeepCR.
        Employed when crmethod='deepcr' or 'mm_deepcr'.
    dc_verbose : bool
        If True, print additional information during the
        execution. Employed when crmethod='deepcr' or 'mm_deepcr'.
    nn_model : str or None
        The neural network model to use for cosmic ray detection.
        Employed when crmethod='conn' or 'mm_conn'.
    nn_threshold : float or None
        The threshold value for the neural network model.
        Employed when crmethod='conn' or 'mm_conn'.
    nn_verbose : bool
        If True, print additional information during the
        execution. Employed when crmethod='conn' or 'mm_conn'.
    mm_synthetic : str or None
        The type of synthetic images to use for the numerical simulations.
        Valid options are:
        - 'median': use the pre-cleaned median combined image.
        - 'single': use the pre-cleaned single images.
    mm_hist2d_min_neighbors : int, optional
        The minimum number of neighboring pixels required to consider
        a pixel as part of a cosmic ray in the 2D histogram used to
        derive the detection boundary.
    mm_ydiag_max : float or None, optional
        The maximum y value to use in the diagnostic diagram.
        If zero or None, it is computed automatically.
    mm_dilation : int, optional
        The dilation factor for the cosmic-ray pixel mask.
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
    mm_threshold_rnoise: float, optional
        Minimum threshold for median2d - min2d in readout noise units
        to consider a pixel as a cosmic ray (default is None).
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
    rlabel_crmethod_plain = f"{crmethod}"
    rlabel_lacosmic_plain = "lacosmic"
    rlabel_pycosmic_plain = "pycosmic"
    rlabel_deepcr_plain = "deepcr  "
    rlabel_conn_plain = "conn    "
    rlabel_mmcosmic_plain = "mmcosmic"
    if rich_configured:
        rlabel_crmethod = f"[bold green]{crmethod}[/bold green]"
        rlabel_lacosmic = "[bold red]lacosmic[/bold red]"
        rlabel_pycosmic = "[bold red]pycosmic[/bold red]"
        rlabel_deepcr = "[bold red]deepcr  [/bold red]"
        rlabel_conn = "[bold red]conn    [/bold red]"
        rlabel_mmcosmic = "[bold blue]mmcosmic[/bold blue]"
    else:
        rlabel_crmethod = rlabel_crmethod_plain
        rlabel_lacosmic = rlabel_lacosmic_plain
        rlabel_pycosmic = rlabel_pycosmic_plain
        rlabel_deepcr = rlabel_deepcr_plain
        rlabel_conn = rlabel_conn_plain
        rlabel_mmcosmic = rlabel_mmcosmic_plain

    # Check crmethod
    if crmethod not in VALID_CRMETHODS:
        raise ValueError(f"Invalid crmethod: {crmethod}. Valid options are {VALID_CRMETHODS}.")
    if crmethod in ["pycosmic", "mm_pycosmic"] and not PYCOSMIC_AVAILABLE:
        raise ImportError(
            "PyCosmic is not installed. Please install PyCosmic to use\n"
            "the 'pycosmic' or 'mm_pycosmic' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install git+https://github.com/nicocardiel/PyCosmic.git@test\n"
        )
    if crmethod in ["deepcr", "mm_deepcr"] and not DEEPCR_AVAILABLE:
        raise ImportError(
            "DeepCR is not installed. Please install DeepCR to use\n"
            "the 'deepcr' or 'mm_deepcr' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install deepCR\n"
        )
    if crmethod in ["conn", "mm_conn"] and not CONN_AVAILABLE:
        raise ImportError(
            "cosmic-conn is not installed. Please install cosmic-conn to use\n"
            "the 'conn' or 'mm_conn' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install cosmic-conn\n"
        )
    # Display crmethod
    _logger.info("computing crmasks using crmethod: %s", rlabel_crmethod)
    # Define prefixes of excluded arguments for each crmethod. They are used
    # later to filter the local variables when generating the output HDUList with masks
    # (to avoid including irrelevant parameters)
    # mm = Median-Mean diagnostic method
    # la = L.A.Cosmic method
    # pc = PyCosmic method
    # dc = DeepCR method
    if crmethod == "lacosmic":
        prefix_of_excluded_args = ["mm_", "pc_", "dc_", "nn_"]
    elif crmethod == "mm_lacosmic":
        prefix_of_excluded_args = ["pc_", "dc_", "nn_"]
    elif crmethod == "pycosmic":
        prefix_of_excluded_args = ["mm_", "la_", "dc_", "nn_"]
    elif crmethod == "mm_pycosmic":
        prefix_of_excluded_args = ["la_", "dc_", "nn_"]
    elif crmethod == "deepcr":
        prefix_of_excluded_args = ["mm_", "la_", "pc_", "nn_"]
    elif crmethod == "mm_deepcr":
        prefix_of_excluded_args = ["la_", "pc_", "nn_"]
    elif crmethod == "conn":
        prefix_of_excluded_args = ["mm_", "la_", "pc_", "dc_"]
    elif crmethod == "mm_conn":
        prefix_of_excluded_args = ["la_", "pc_", "dc_"]
    else:
        prefix_of_excluded_args = ["xxx"]  # should never happen
    # Define acronyms for problematic pixels
    if crmethod in ["lacosmic", "mm_lacosmic"]:
        acronym_aux = "la"
    elif crmethod in ["pycosmic", "mm_pycosmic"]:
        acronym_aux = "pc"
    elif crmethod in ["deepcr", "mm_deepcr"]:
        acronym_aux = "dc"
    elif crmethod in ["conn", "mm_conn"]:
        acronym_aux = "nn"
    elif crmethod in ["mmcosmic"]:
        acronym_aux = None
    else:
        raise ValueError(f"Invalid crmethod: {crmethod}. This should never happen.")

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

    # Define mm_threshold
    if rnoise_scalar is not None:
        mm_threshold = rnoise_scalar * mm_threshold_rnoise
    else:
        mm_threshold = 0.0

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

    # Check use_auxmedian
    if use_auxmedian and crmethod not in ["lacosmic", "mm_lacosmic", "pycosmic", "mm_pycosmic", "deepcr", "mm_deepcr"]:
        raise ValueError(
            "use_auxmedian can only be True when crmethod is 'lacosmic', 'mm_lacosmic', 'pycosmic', 'mm_pycosmic', 'deepcr', or 'mm_deepcr'."
        )
    _logger.info("use_auxmedian: %s", str(use_auxmedian))

    # Check flux_factor
    if isinstance(flux_factor_regions, str):
        if flux_factor_regions.lower() == "none":
            flux_factor_regions = None
    if flux_factor is None:
        flux_factor = np.ones(num_images, dtype=float)
    elif isinstance(flux_factor, str):
        if flux_factor.lower() == "auto":
            _logger.info("flux_factor set to 'auto', computing values...")
            list_flux_factor_regions = []
            if isinstance(flux_factor_regions, str):
                flux_factor_regions = ast.literal_eval(flux_factor_regions)
            if isinstance(flux_factor_regions, list):
                for flux_factor_region in flux_factor_regions:
                    if isinstance(flux_factor_region, list):
                        all_integers = all(isinstance(val, int) for val in flux_factor_region)
                        if not all_integers:
                            raise TypeError(
                                f"Invalid flux_factor_region: {flux_factor_region}. " "All elements must be integers."
                            )
                        if len(flux_factor_region) != 4:
                            raise ValueError(
                                f"Invalid length for flux_factor_region: {flux_factor_region}. "
                                "Must be a list of 4 integers [xmin, xmax, ymin, ymax]."
                            )
                        dumreg = (
                            f"[{flux_factor_region[0]}:{flux_factor_region[1]}, "
                            + f"{flux_factor_region[2]}:{flux_factor_region[3]}]"
                        )
                        _logger.debug("defined flux factor region: %s", dumreg)
                        ff_region = tea.SliceRegion2D(dumreg, mode="fits", naxis1=naxis1, naxis2=naxis2)
                    else:
                        raise TypeError(
                            f"Invalid type for flux_factor_region in the list: {type(flux_factor_region)}. "
                            "Must be a list of 4 integers"
                        )
                    list_flux_factor_regions.append(ff_region)
            elif flux_factor_regions is None:
                ff_region = tea.SliceRegion2D(f"[1:{naxis1}, 1:{naxis2}]", mode="fits")
                list_flux_factor_regions = [ff_region]
            else:
                raise TypeError(
                    f"Invalid type for flux_factor_regions: {type(flux_factor_regions)}. "
                    "Must be list of 4 integers or None."
                )
            median2d = np.median(image3d, axis=0)
            flux_factor = compute_flux_factor(image3d, median2d, list_flux_factor_regions, _logger, interactive, debug)
            _logger.info("computed flux_factor set to %s", str(flux_factor))
        elif flux_factor.lower() == "none":
            flux_factor = np.ones(num_images, dtype=float)
            if flux_factor_regions is not None:
                raise ValueError(
                    "Using flux_factor='none', but flux_factor_regions is provided. "
                    "You must use flux_factor='auto' to use flux_factor_regions."
                )
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
    if mm_synthetic == "single" and not np.array_equal(flux_factor, np.ones(num_images)):
        raise NotImplementedError("mm_synthetic='single' is not compatible with flux_factor values different from 1.0.")
    if apply_flux_factor_to not in ["original", "simulated"]:
        raise ValueError(
            f"Invalid apply_flux_factor_to: {apply_flux_factor_to}. " "Valid options are 'original' and 'simulated'."
        )
    _logger.info("apply_flux_factor_to: %s", apply_flux_factor_to)

    # Apply the flux factor to the input arrays if requested
    if apply_flux_factor_to == "original":
        for i in range(num_images):
            image3d[i] /= flux_factor[i]

    # Define regions to be cleaned by computing a boolean mask
    # that is True for pixels not included in the regions to be skipped
    bool_to_be_cleaned = np.ones((naxis2, naxis1), dtype=bool)  # default is to clean all pixels
    if isinstance(regions_to_be_skipped, str):
        if regions_to_be_skipped.lower() == "none":
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
                    raise ValueError(
                        f"Invalid length for region_to_be_skipped: {region}. "
                        "Must be a list of 4 integers [xmin, xmax, ymin, ymax]."
                    )
                dumreg = f"[{region[0]}:{region[1]}, {region[2]}:{region[3]}]"
                _logger.debug("defined region to be skipped: %s", dumreg)
                skip_region = tea.SliceRegion2D(dumreg, mode="fits", naxis1=naxis1, naxis2=naxis2)
                bool_to_be_cleaned[
                    skip_region.python[0].start : skip_region.python[0].stop,
                    skip_region.python[1].start : skip_region.python[1].stop,
                ] = False
            else:
                raise TypeError(
                    f"Invalid type for region_to_be_skipped in the list: {type(region)}. "
                    "Must be a list of 4 integers."
                )

    # Compute minimum, maximum, median and mean along the first axis
    min2d = np.min(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    mean2d = np.mean(image3d, axis=0)

    # Compute points for diagnostic diagram of mmcosmic method
    xplot = min2d.flatten()  # bias was already subtracted above
    yplot = median2d.flatten() - min2d.flatten()

    # Check that color_scale is valid
    if color_scale not in ["minmax", "zscale"]:
        raise ValueError(f"Invalid color_scale: {color_scale}. Valid options are 'minmax' and 'zscale'.")

    _logger.info("dtype for output arrays: %s", dtype)
    _logger.info("dilation factor: %d", dilation)
    _logger.info("verify cosmic-ray detection: %s", verify_cr)
    _logger.info("semiwindow size for plotting coincident cosmic-ray pixels: %d", semiwindow)
    _logger.info("maximum number of coincident cosmic-ray pixels to plot: %d", maxplots)
    _logger.info("color scale for plots: %s", color_scale)

    # Define the pixels to be flagged as CR
    if isinstance(pixels_to_be_flagged_as_cr, str):
        if pixels_to_be_flagged_as_cr.lower() == "none":
            pixels_to_be_flagged_as_cr = None
        else:
            pixels_to_be_flagged_as_cr = ast.literal_eval(pixels_to_be_flagged_as_cr)
    if isinstance(pixels_to_be_flagged_as_cr, (list, tuple)):
        for p in pixels_to_be_flagged_as_cr:
            if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(item, int) for item in p):
                raise ValueError(
                    f"Invalid numbers in pixels_to_be_flagged_as_cr: {p}. "
                    "Each pixel must be a tuple or list of two integers (X, Y)."
                )
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_flagged_as_cr are out of bounds.")
            # ensure these pixels are cleaned, independently of being included
            # in regions_to_be_skipped
            bool_to_be_cleaned[p[1] - 1, p[0] - 1] = True
    elif pixels_to_be_flagged_as_cr is not None:
        raise TypeError(
            f"Invalid type for pixels_to_be_flagged_as_cr: {type(pixels_to_be_flagged_as_cr)}. "
            "Must be a list of (x, y) tuples or None."
        )
    _logger.info(
        "individual pixels to be initially flagged as CR: %s",
        "None" if pixels_to_be_flagged_as_cr is None else str(pixels_to_be_flagged_as_cr),
    )

    # Define the pixels to be ignored as CR
    # check that the provided pixels are valid
    if isinstance(pixels_to_be_ignored_as_cr, str):
        if pixels_to_be_ignored_as_cr.lower() == "none":
            pixels_to_be_ignored_as_cr = None
        else:
            pixels_to_be_ignored_as_cr = ast.literal_eval(pixels_to_be_ignored_as_cr)
    if isinstance(pixels_to_be_ignored_as_cr, (list, tuple)):
        for p in pixels_to_be_ignored_as_cr:
            if not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(item, int) for item in p):
                raise ValueError(
                    f"Invalid numbers in pixels_to_be_ignored_as_cr: {p}. "
                    "Each pixel must be a tuple or list of two integers (X, Y)."
                )
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_ignored_as_cr are out of bounds.")
            # ensure these pixels are not cleaned
            bool_to_be_cleaned[p[1] - 1, p[0] - 1] = False
    elif pixels_to_be_ignored_as_cr is not None:
        raise TypeError(
            f"Invalid type for pixels_to_be_ignored_as_cr: {type(pixels_to_be_ignored_as_cr)}. "
            "Must be a list of (x, y) tuples or None."
        )
    _logger.info(
        "individual pixels to be initially ignored as CR: %s",
        "None" if pixels_to_be_ignored_as_cr is None else str(pixels_to_be_ignored_as_cr),
    )

    # Define the pixels to be replaced by the median value when removing the CRs
    # check that the provided pixels are valid
    if isinstance(pixels_to_be_replaced_by_local_median, str):
        if pixels_to_be_replaced_by_local_median.lower() == "none":
            pixels_to_be_replaced_by_local_median = None
        else:
            pixels_to_be_replaced_by_local_median = ast.literal_eval(pixels_to_be_replaced_by_local_median)
    if isinstance(pixels_to_be_replaced_by_local_median, (list, tuple)):
        for p in pixels_to_be_replaced_by_local_median:
            if not isinstance(p, (list, tuple)) or len(p) != 4 or not all(isinstance(item, int) for item in p):
                raise ValueError(
                    f"Invalid numbers in pixels_to_be_replaced_by_local_median: {p}. "
                    "Each pixel must be a tuple or list of four integers (X, Y, X_width, Y_width)."
                )
            if p[0] < 1 or p[0] > naxis1 or p[1] < 1 or p[1] > naxis2:
                raise ValueError(f"Pixel coordinates {p} in pixels_to_be_replaced_by_local_median are out of bounds.")
            if p[2] % 2 == 0 or p[3] % 2 == 0 or p[2] < 1 or p[3] < 1:
                raise ValueError(
                    f"Pixel {p}: X_width and Y_width in pixels_to_be_replaced_by_local_median "
                    "must be odd integers >= 1."
                )
            if p[2] * p[3] < 3:
                raise ValueError(
                    f"Pixel {p}: The area defined by X_width and Y_width in "
                    "pixels_to_be_replaced_by_local_median must be >= 3."
                )
    elif pixels_to_be_replaced_by_local_median is not None:
        raise TypeError(
            f"Invalid type for pixels_to_be_replaced_by_local_median: "
            f"{type(pixels_to_be_replaced_by_local_median)}. "
            "Must be a list of (x, y, x_width, y_width) tuples or None."
        )
    _logger.info(
        "pixels to be replaced by the median value when removing the CRs: %s",
        "None" if pixels_to_be_replaced_by_local_median is None else str(pixels_to_be_replaced_by_local_median),
    )

    # These pixels to be replaced by the local median should not be
    # flagged as CR if they will be replaced by the local median anyway
    if pixels_to_be_replaced_by_local_median is not None:
        for p in pixels_to_be_replaced_by_local_median:
            if bool_to_be_cleaned[p[1] - 1, p[0] - 1]:
                _logger.warning(
                    "Pixel %s is set to be replaced by the local median "
                    "but it is also set to be cleaned as CR. "
                    "It will not be cleaned as CR but will be replaced by the local median.",
                    str(p),
                )
                bool_to_be_cleaned[p[1] - 1, p[0] - 1] = False
        _logger.info(
            "updated pixels to be ignored as CR: %s",
            "None" if pixels_to_be_ignored_as_cr is None else str(pixels_to_be_ignored_as_cr),
        )

    # Log the input parameters
    if crmethod in ["mm_lacosmic", "mm_pycosmic", "mm_deepcr", "mm_conn"]:
        _logger.info("mm_synthetic: %s", str(mm_synthetic))
        _logger.info("mm_hist2d_min_neighbors: %d", mm_hist2d_min_neighbors)
        if mm_hist2d_min_neighbors < 0:
            raise ValueError(f"{mm_hist2d_min_neighbors=} must be >= 0.")
        if mm_hist2d_min_neighbors > 8:
            raise ValueError(f"{mm_hist2d_min_neighbors=} must be <= 8.")
        _logger.info("mm_dilation: %d", mm_dilation)
        _logger.info("mm_xy_offsets: %s", str(mm_xy_offsets) if mm_xy_offsets is not None else "None")
        _logger.info("mm_crosscorr_region: %s", mm_crosscorr_region if mm_crosscorr_region is not None else "None")
        _logger.info("mm_boundary_fit: %s", mm_boundary_fit if mm_boundary_fit is not None else "None")
        _logger.info("mm_knots_splfit: %d", mm_knots_splfit)
        _logger.info(
            "mm_fixed points_in_boundary: %s",
            str(mm_fixed_points_in_boundary) if mm_fixed_points_in_boundary is not None else "None",
        )
        _logger.info("mm_nsimulations: %d", mm_nsimulations)
        _logger.info("mm_niter_boundary_extension: %d", mm_niter_boundary_extension)
        _logger.info("mm_weight_boundary_extension: %f", mm_weight_boundary_extension)
        _logger.info("mm_threshold_rnoise: %s", mm_threshold_rnoise)
        _logger.info("mm_threshold: %f (derived parameter)", mm_threshold)
        _logger.info("mm_minimum_max2d_rnoise: %f", mm_minimum_max2d_rnoise)
        _logger.info("mm_seed: %s", str(mm_seed))

    # Check and update L.A.Cosmic parameters
    if crmethod in ["lacosmic", "mm_lacosmic"]:
        # Check la_gain_apply
        if la_gain_apply is None:
            la_gain_apply = True
            _logger.warning("la_gain_apply for lacosmic not defined, assuming la_gain_apply=True")
        else:
            _logger.debug("la_gain_apply for lacosmic: %s", str(la_gain_apply))
        # Check la_sigclip
        if la_sigclip is None:
            la_sigclip = 5.0
            _logger.warning(f"la_sigclip for lacosmic not defined, assuming la_sigclip={la_sigclip}")
        else:
            _logger.debug("la_sigclip for lacosmic: %s", str(la_sigclip))
        # Check la_sigfrac
        if la_sigfrac is None:
            la_sigfrac = 0.3
            _logger.warning(f"la_sigfrac for lacosmic not defined, assuming la_sigfrac={la_sigfrac}")
        else:
            _logger.debug("la_sigfrac for lacosmic: %s", str(la_sigfrac))
        # Check la_objlim
        if la_objlim is None:
            la_objlim = 5.0
            _logger.warning(f"la_objlim for lacosmic not defined, assuming la_objlim={la_objlim}")
        else:
            _logger.debug("la_objlim for lacosmic: %s", str(la_objlim))
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
        if la_fsmode not in ["median", "convolve"]:
            raise ValueError("la_fsmode must be 'median' or 'convolve'.")
        else:
            _logger.debug("la_fsmode for lacosmic: %s", la_fsmode)
        # Check la_psfmodel
        if la_psfmodel not in ["gauss", "moffat", "gaussx", "gaussy", "gaussxy"]:
            raise ValueError("la_psfmodel must be 'gauss', 'moffat', 'gaussx', 'gaussy', or 'gaussxy'.")
        else:
            _logger.debug("la_psfmodel for lacosmic: %s", la_psfmodel)
        # Check la_psffwhm_x, la_psffwhm_y, la_psfsize
        if la_fsmode == "convolve":
            if la_psffwhm_x is None or la_psffwhm_y is None or la_psfsize is None:
                raise ValueError(
                    "For la_fsmode='convolve', " "la_psffwhm_x, la_psffwhm_y, and la_psfsize must be provided."
                )
            else:
                _logger.debug("la_psffwhm_x for lacosmic: %f", la_psffwhm_x)
                _logger.debug("la_psffwhm_y for lacosmic: %f", la_psffwhm_y)
                _logger.debug("la_psfsize for lacosmic: %d", la_psfsize)
            if la_psfsize % 2 == 0 or la_psfsize < 3:
                raise ValueError("la_psfsize must be an odd integer >= 3.")
        # Check la_psfbeta
        if la_psfmodel == "moffat":
            if la_psfbeta is None:
                raise ValueError("For la_psfmodel='moffat', la_psfbeta must be provided.")
            else:
                _logger.debug("la_psfbeta for lacosmic: %f", la_psfbeta)
        # Set la_verbose
        current_logging_level = logging.getLogger().getEffectiveLevel()
        if current_logging_level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            la_verbose = False
        _logger.debug("la_verbose for lacosmic: %s", str(la_verbose))
        # Check la_padwidth
        if la_padwidth is None:
            la_padwidth = 0
            _logger.debug("la_padwidth for lacosmic not defined, assuming la_padwidth=0")
        else:
            if not isinstance(la_padwidth, int) or la_padwidth < 0:
                raise ValueError("la_padwidth must be a non-negative integer.")
            _logger.debug("la_padwidth for lacosmic: %d", la_padwidth)
        # Define dictionary with the parameters for cosmicray_lacosmic() function
        # Note: the "pssl" parameter is not used here because it was deprecated
        # in version 2.3.0 and will be removed in a future version.
        # The "pssl" keyword will be removed in ccdproc 3.0.
        # Use "inbkg" instead to have astroscrappy temporarily remove the background during processing.
        dict_la_params_run1 = {
            "gain": gain_scalar,
            "readnoise": rnoise_scalar,
            "sigclip": None,  # to be set below
            "sigfrac": None,  # to be set below
            "objlim": None,  # to be set below
            "satlevel": la_satlevel * gain_scalar if la_satlevel is not None else None,  # in electrons!
            "niter": la_niter,
            "sepmed": la_sepmed,
            "cleantype": la_cleantype,
            "fsmode": la_fsmode,
            "psfmodel": la_psfmodel,
            "psffwhm": None,
            "psfsize": la_psfsize,
            "psfbeta": la_psfbeta,
            "verbose": la_verbose,
            "psfk": None,
            "inbkg": None,
            "invar": None,
        }
        dict_la_params_run2 = dict_la_params_run1.copy()
        # update sigclip
        if isinstance(la_sigclip, (float, int)):
            dict_la_params_run1["sigclip"] = la_sigclip
            dict_la_params_run2["sigclip"] = la_sigclip
        elif all_valid_numbers(la_sigclip, fixed_length=2):
            dict_la_params_run1["sigclip"] = la_sigclip[0]
            dict_la_params_run2["sigclip"] = la_sigclip[1]
        else:
            raise TypeError("la_sigclip must be a number or a list of 2 numbers.")
        # update sigfrac
        if isinstance(la_sigfrac, (float, int)):
            dict_la_params_run1["sigfrac"] = la_sigfrac
            dict_la_params_run2["sigfrac"] = la_sigfrac
        elif all_valid_numbers(la_sigfrac, fixed_length=2):
            dict_la_params_run1["sigfrac"] = la_sigfrac[0]
            dict_la_params_run2["sigfrac"] = la_sigfrac[1]
        else:
            raise TypeError("la_sigfrac must be a number or a list of 2 numbers.")
        # update objlim
        if isinstance(la_objlim, (float, int)):
            dict_la_params_run1["objlim"] = la_objlim
            dict_la_params_run2["objlim"] = la_objlim
        elif all_valid_numbers(la_objlim, fixed_length=2):
            dict_la_params_run1["objlim"] = la_objlim[0]
            dict_la_params_run2["objlim"] = la_objlim[1]
        else:
            raise TypeError("la_objlim must be a number or a list of 2 numbers.")
        # Update psffwhm or psfk based on la_psfmodel
        if la_psfmodel in ["gauss", "moffat"]:
            if la_psffwhm_x is None or la_psfsize is None:
                raise ValueError(
                    "For la_psfmodel='gauss' or 'moffat', " "la_psffwhm_x and la_psfsize must be provided."
                )
            dict_la_params_run1["psffwhm"] = (la_psffwhm_x + la_psffwhm_y) / 2.0  # average FWHM
            dict_la_params_run2["psffwhm"] = (la_psffwhm_x + la_psffwhm_y) / 2.0  # average FWHM
        elif la_psfmodel == "gaussx":
            if la_psffwhm_x is None or la_psfsize is None:
                raise ValueError("For la_psfmodel='gaussx', " "la_psffwhm_x and la_psfsize must be provided.")
            dict_la_params_run1["psffwhm"] = la_psffwhm_x
            dict_la_params_run2["psffwhm"] = la_psffwhm_x
        elif la_psfmodel == "gaussy":
            if la_psffwhm_y is None or la_psfsize is None:
                raise ValueError("For la_psfmodel='gaussy', " "la_psffwhm_y and la_psfsize must be provided.")
            dict_la_params_run1["psffwhm"] = la_psffwhm_y
            dict_la_params_run2["psffwhm"] = la_psffwhm_y
        elif la_psfmodel == "gaussxy":
            dict_la_params_run1["psffwhm"] = None  # not used in this case
            dict_la_params_run2["psffwhm"] = None  # not used in this case
            dict_la_params_run1["psfk"] = gausskernel2d_elliptical(
                fwhm_x=la_psffwhm_x, fwhm_y=la_psffwhm_y, kernsize=la_psfsize
            )
            dict_la_params_run2["psfk"] = gausskernel2d_elliptical(
                fwhm_x=la_psffwhm_x, fwhm_y=la_psffwhm_y, kernsize=la_psfsize
            )
        else:
            raise ValueError("la_psfmodel must be 'gauss', 'moffat', 'gaussx', 'gaussy', or 'gaussxy'.")

    # Check and update PyCosmic parameters
    if crmethod in ["pycosmic", "mm_pycosmic"]:
        # Check pc_sigma_det
        if pc_sigma_det is None:
            pc_sigma_det = 5.0
            _logger.warning(f"pc_sigma_det for pycosmic not defined, assuming pc_sigma_det={pc_sigma_det}")
        else:
            _logger.debug("pc_sigma_det for pycosmic: %s", str(pc_sigma_det))
        # Check pc_rlim
        if pc_rlim is None:
            pc_rlim = 1.2
            _logger.warning(f"pc_rlim for pycosmic not defined, assuming pc_rlim={pc_rlim}")
        else:
            _logger.debug("pc_rlim for pycosmic: %s", str(pc_rlim))
        # Check pc_iterations
        if pc_iterations is None:
            pc_iterations = 5
            _logger.warning(f"pc_iterations for pycosmic not defined, assuming pc_iterations={pc_iterations}")
        else:
            _logger.debug("pc_iterations for pycosmic: %d", pc_iterations)
        # Check pc_fwhm_gauss_x
        if pc_fwhm_gauss_x is None:
            pc_fwhm_gauss_x = 2.5
            _logger.warning(f"pc_fwhm_gauss_x for pycosmic not defined, assuming pc_fwhm_gauss_x={pc_fwhm_gauss_x}")
        else:
            _logger.debug("pc_fwhm_gauss_x for pycosmic: %f", pc_fwhm_gauss_x)
        # Check pc_fwhm_gauss_y
        if pc_fwhm_gauss_y is None:
            pc_fwhm_gauss_y = 2.5
            _logger.warning(f"pc_fwhm_gauss_y for pycosmic not defined, assuming pc_fwhm_gauss_y={pc_fwhm_gauss_y}")
        else:
            _logger.debug("pc_fwhm_gauss_y for pycosmic: %f", pc_fwhm_gauss_y)
        # Check pc_replace_box_x
        if pc_replace_box_x is None:
            pc_replace_box_x = 5
            _logger.warning(f"pc_replace_box_x for pycosmic not defined, assuming pc_replace_box_x={pc_replace_box_x}")
        else:
            _logger.debug("pc_replace_box_x for pycosmic: %d", pc_replace_box_x)
        # Check pc_replace_box_y
        if pc_replace_box_y is None:
            pc_replace_box_y = 5
            _logger.warning(f"pc_replace_box_y for pycosmic not defined, assuming pc_replace_box_y={pc_replace_box_y}")
        else:
            _logger.debug("pc_replace_box_y for pycosmic: %d", pc_replace_box_y)
        # Check pc_replace_error
        if pc_replace_error is None:
            pc_replace_error = 1.0e6
            _logger.warning(f"pc_replace_error for pycosmic not defined, assuming pc_replace_error={pc_replace_error}")
        else:
            try:
                pc_replace_error = float(pc_replace_error)
            except ValueError:
                raise TypeError("pc_replace_error must be a valid number.")
            _logger.debug("pc_replace_error for pycosmic: %s", pc_replace_error)
        # Check pc_increase_radius
        if pc_increase_radius is None:
            pc_increase_radius = 0
            _logger.warning(
                f"pc_increase_radius for pycosmic not defined, assuming pc_increase_radius={pc_increase_radius}"
            )
        else:
            _logger.debug("pc_increase_radius for pycosmic: %d", pc_increase_radius)
        # Set pc_verbose
        current_logging_level = logging.getLogger().getEffectiveLevel()
        if current_logging_level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            pc_verbose = False
        _logger.debug("pc_verbose for pycosmic: %s", str(pc_verbose))
        # Define dictionary with the parameters for cosmicray_pycosmic() function
        dict_pc_params_run1 = {
            "sigma_det": None,  # to be set below
            "rlim": None,  # to be set below
            "iterations": pc_iterations,
            "fwhm_gauss": [pc_fwhm_gauss_x, pc_fwhm_gauss_y],
            "replace_box": [pc_replace_box_x, pc_replace_box_y],
            "replace_error": pc_replace_error,
            "increase_radius": pc_increase_radius,
            "gain": gain_scalar,
            "rdnoise": rnoise_scalar,
            "bias": 0.0,
            "verbose": pc_verbose,
        }
        dict_pc_params_run2 = dict_pc_params_run1.copy()
        # update sigma_det
        if isinstance(pc_sigma_det, (float, int)):
            dict_pc_params_run1["sigma_det"] = pc_sigma_det
            dict_pc_params_run2["sigma_det"] = pc_sigma_det
        elif all_valid_numbers(pc_sigma_det, fixed_length=2):
            dict_pc_params_run1["sigma_det"] = pc_sigma_det[0]
            dict_pc_params_run2["sigma_det"] = pc_sigma_det[1]
        else:
            raise TypeError("pc_sigma_det must be a number or a list of 2 numbers.")
        # update rlim
        if isinstance(pc_rlim, (float, int)):
            dict_pc_params_run1["rlim"] = pc_rlim
            dict_pc_params_run2["rlim"] = pc_rlim
        elif all_valid_numbers(pc_rlim, fixed_length=2):
            dict_pc_params_run1["rlim"] = pc_rlim[0]
            dict_pc_params_run2["rlim"] = pc_rlim[1]
        else:
            raise TypeError("pc_rlim must be a number or a list of 2 numbers.")

    # Check and update DeepCR parameters
    if crmethod in ["deepcr", "mm_deepcr"]:
        # Check dc_mask
        if dc_mask is None:
            dc_mask = "ACS-WFC"
            _logger.warning(f"dc_mask for deepcr not defined, assuming dc_mask='{dc_mask}'")
        else:
            _logger.debug("dc_mask for deepcr: %s", str(dc_mask))
            if dc_mask not in ["ACS-WFC", "WFC3-UVIS"]:
                raise ValueError("dc_mask must be 'ACS-WFC' or 'WFC3-UVIS'.")
        # Check dc_threshold
        if dc_threshold is None:
            dc_threshold = 0.5
            _logger.warning(f"dc_threshold for deepcr not defined, assuming dc_threshold={dc_threshold}")
        else:
            _logger.debug("dc_threshold for deepcr: %s", str(dc_threshold))
            if (dc_threshold <= 0.0) or (dc_threshold >= 1.0):
                raise ValueError("dc_threshold must be a number between 0 and 1.")
        # Set dc_verbose
        current_logging_level = logging.getLogger().getEffectiveLevel()
        if current_logging_level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            dc_verbose = False
        _logger.debug("dc_verbose for deepcr: %s", str(dc_verbose))
        # Define dictionary with the parameters for deepCR
        dict_dc_params = {
            "mask": dc_mask,
            "threshold": dc_threshold,
            "verbose": dc_verbose,
        }

    # Check and update Cosmic-CoNN parameters
    if crmethod in ["conn", "mm_conn"]:
        # Check nn_model
        if nn_model is None:
            nn_model = "ground_imaging"
            _logger.warning(f"nn_model for Cosmic-CoNN not defined, assuming nn_model='{nn_model}'")
        else:
            _logger.debug("nn_model for Cosmic-CoNN: %s", str(nn_model))
            if nn_model not in ["ground_imaging", "NRES", "HST_ACS_WFC"]:
                raise ValueError("nn_model must be 'ground_imaging', 'NRES', or 'HST_ACS_WFC'.")
        # Check nn_threshold
        if nn_threshold is None:
            nn_threshold = 0.5
            _logger.warning(f"nn_threshold for Cosmic-CoNN not defined, assuming nn_threshold={nn_threshold}")
        else:
            _logger.debug("nn_threshold for Cosmic-CoNN: %s", str(nn_threshold))
            if (nn_threshold <= 0.0) or (nn_threshold >= 1.0):
                raise ValueError("nn_threshold must be a number between 0 and 1.")
        # Set nn_verbose
        current_logging_level = logging.getLogger().getEffectiveLevel()
        if current_logging_level in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            nn_verbose = False
        _logger.debug("nn_verbose for Cosmic-CoNN: %s", str(nn_verbose))
        # Define dictionary with the parameters for Cosmic-CoNN
        dict_nn_params = {
            "model": nn_model,
            "threshold": nn_threshold,
            "verbose": nn_verbose,
        }

    # #########################################################################
    # Start cosmic ray detection in median2d image
    # #########################################################################
    median2d_aux = None
    flag_aux = None
    rlabel_aux = None
    rlabel_aux_plain = None

    if rich_configured:
        _logger.info("[green]" + "-" * 79 + "[/green]")
        _logger.info("starting cosmic ray detection in [magenta]median2d[/magenta]...")
    else:
        _logger.info("-" * 73)
        _logger.info("starting cosmic ray detection in median2d...")

    if crmethod in ["lacosmic", "mm_lacosmic"]:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using L.A.Cosmic,
        # the Laplacian edge detection method from ccdproc. This only works
        # if gain and rnoise are constant values (scalars).
        # ---------------------------------------------------------------------
        rlabel_aux = rlabel_lacosmic
        rlabel_aux_plain = rlabel_lacosmic_plain
        if gain_scalar is None or rnoise_scalar is None:
            raise ValueError("gain and rnoise must be constant values (scalars) when using crmethod='lacosmic'.")
        median2d_aux, flag_aux = execute_lacosmic(
            image2d=median2d,
            bool_to_be_cleaned=bool_to_be_cleaned,
            rlabel_lacosmic=rlabel_lacosmic,
            dict_la_params_run1=dict_la_params_run1,
            dict_la_params_run2=dict_la_params_run2,
            la_padwidth=la_padwidth,
            displaypar=dict_la_params_run1["verbose"] or dict_la_params_run2["verbose"],
            _logger=_logger,
        )
    elif crmethod in ["pycosmic", "mm_pycosmic"]:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using PyCosmic.
        # This only works if gain and rnoise are constant values (scalars).
        # ---------------------------------------------------------------------
        rlabel_aux = rlabel_pycosmic
        rlabel_aux_plain = rlabel_pycosmic_plain
        if gain_scalar is None or rnoise_scalar is None:
            raise ValueError("gain and rnoise must be constant values (scalars) when using crmethod='lacosmic'.")
        median2d_aux, flag_aux = execute_pycosmic(
            image2d=median2d,
            bool_to_be_cleaned=bool_to_be_cleaned,
            rlabel_pycosmic=rlabel_pycosmic,
            dict_pc_params_run1=dict_pc_params_run1,
            dict_pc_params_run2=dict_pc_params_run2,
            displaypar=dict_pc_params_run1["verbose"] or dict_pc_params_run2["verbose"],
            _logger=_logger,
        )
    elif crmethod in ["deepcr", "mm_deepcr"]:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using DeepCR.
        # ---------------------------------------------------------------------
        rlabel_aux = rlabel_deepcr
        rlabel_aux_plain = rlabel_deepcr_plain
        median2d_aux, flag_aux = execute_deepcr(
            image2d=median2d,
            bool_to_be_cleaned=bool_to_be_cleaned,
            rlabel_deepcr=rlabel_deepcr,
            dict_dc_params=dict_dc_params,
            displaypar=dict_dc_params["verbose"],
            _logger=_logger,
        )
    elif crmethod in ["conn", "mm_conn"]:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using Cosmic-CoNN.
        # ---------------------------------------------------------------------
        rlabel_aux = rlabel_conn
        rlabel_aux_plain = rlabel_conn_plain
        flag_aux = execute_conn(
            image2d=median2d,
            bool_to_be_cleaned=bool_to_be_cleaned,
            rlabel_conn=rlabel_conn,
            dict_nn_params=dict_nn_params,
            displaypar=dict_nn_params["verbose"],
            _logger=_logger,
        )
        median2d_aux = None
        if use_auxmedian:
            raise NotImplementedError("use_auxmedian=True is not implemented for crmethod='conn' or 'mm_conn'.")
    else:
        raise ValueError(f"Invalid crmethod: {crmethod}. " f"Valid options are: {VALID_CRMETHODS}.")

    if crmethod in ["mm_lacosmic", "mm_pycosmic", "mm_deepcr", "mm_conn"]:
        # Apply thresholding in M.M. diagram to help removing false positives in auxiliary method
        _logger.info(f"applying mm_threshold={mm_threshold} to auxiliary method {rlabel_aux}...")
        npixels_found_before = np.sum(flag_aux)
        flag_aux = apply_threshold_cr(
            bool_crmask2d=flag_aux.reshape((naxis2, naxis1)),
            bool_threshold2d=(yplot > mm_threshold).reshape((naxis2, naxis1)),
        ).flatten()
        npixels_found_after = np.sum(flag_aux)
        ldum = len(str(npixels_found_before))
        _logger.info(f"number of CR pixels before applying mm_threshold: {npixels_found_before:>{ldum}d}")
        _logger.info(f"number of CR pixels after  applying mm_threshold: {npixels_found_after:>{ldum}d}")

    # Define a pre-cleaned median2d image for creating the fake individual images
    median2d_precleaned = median2d.copy()
    # Ensure no negative values in median2d_precleaned
    median2d_precleaned[median2d_precleaned < 0.0] = 0.0
    # Pre-clean the median2d image by replacing the detected CR pixels with the minimum value
    flag_aux = flag_aux.reshape((naxis2, naxis1))
    median2d_precleaned[flag_aux > 0] = min2d[flag_aux > 0]
    flag_aux = flag_aux.flatten()
    image3d_cleaned_single = np.zeros((num_images, naxis2, naxis1), dtype=float)
    flag3d_cleaned_single = np.zeros((num_images, naxis2, naxis1), dtype=int)
    # ---------------------------------------------------------------------
    # Clean each individual image. The resulting images can be employed
    # to compute the detection boundary.
    # ---------------------------------------------------------------------
    for i in range(num_images):
        dumimage2d = image3d[i, :, :]
        # Clean the dumimage2d image
        if rich_configured:
            _logger.info("[green]" + "-" * 79 + "[/green]")
            _logger.info(f"starting cosmic ray detection in [magenta]image#{i+1}[/magenta]...")
        else:
            _logger.info("-" * 73)
            _logger.info(f"starting cosmic ray detection in image#{i+1}...")
        if crmethod in ["lacosmic", "mm_lacosmic"]:
            dumimage2d_cleaned, flagdum = execute_lacosmic(
                image2d=dumimage2d,
                bool_to_be_cleaned=bool_to_be_cleaned,
                rlabel_lacosmic=rlabel_lacosmic,
                dict_la_params_run1=dict_la_params_run1,
                dict_la_params_run2=dict_la_params_run2,
                la_padwidth=la_padwidth,
                displaypar=False,
                _logger=_logger,
            )
        elif crmethod in ["pycosmic", "mm_pycosmic"]:
            dumimage2d_cleaned, flagdum = execute_pycosmic(
                image2d=dumimage2d,
                bool_to_be_cleaned=bool_to_be_cleaned,
                rlabel_pycosmic=rlabel_pycosmic,
                dict_pc_params_run1=dict_pc_params_run1,
                dict_pc_params_run2=dict_pc_params_run2,
                displaypar=False,
                _logger=_logger,
            )
        elif crmethod in ["deepcr", "mm_deepcr"]:
            dumimage2d_cleaned, flagdum = execute_deepcr(
                image2d=dumimage2d,
                bool_to_be_cleaned=bool_to_be_cleaned,
                rlabel_deepcr=rlabel_deepcr,
                dict_dc_params=dict_dc_params,
                displaypar=False,
                _logger=_logger,
            )
        elif crmethod in ["conn", "mm_conn"]:
            flagdum = execute_conn(
                image2d=dumimage2d,
                bool_to_be_cleaned=bool_to_be_cleaned,
                rlabel_conn=rlabel_conn,
                dict_nn_params=dict_nn_params,
                displaypar=False,
                _logger=_logger,
            )
            dumimage2d_cleaned = dumimage2d  # not used
        else:
            raise ValueError(f"Invalid crmethod: {crmethod}.")
        # Store the cleaned image
        image3d_cleaned_single[i] = dumimage2d_cleaned
        # Ensure no negative values
        image3d_cleaned_single[i][image3d_cleaned_single[i] < 0.0] = 0.0
        # Create 3D flag array
        flag3d_cleaned_single[i] = flagdum.reshape((naxis2, naxis1)).astype(int)
    # Create 2D flag array by summing over the 3D flag array
    flag2d_cleaned_single = np.sum(flag3d_cleaned_single, axis=0)
    # Flatten the 2D flag array to 1D
    flag2d_cleaned_single = flag2d_cleaned_single.flatten()
    # Add pixels flagged in median2d
    flag2d_cleaned_single += flag_aux

    if crmethod in ["mm_lacosmic", "mm_pycosmic", "mm_deepcr", "mm_conn"]:
        # ---------------------------------------------------------------------
        # Compute detection boundary in M.M. diagram to detect cosmic rays
        # in median2d image
        # ---------------------------------------------------------------------
        # Define mm_fixed_points_in_boundary
        if rich_configured:
            _logger.info("[green]" + "-" * 79 + "[/green]")
            _logger.info(f"detecting cosmic rays in [magenta]median2d[/magenta] using {rlabel_mmcosmic}...")
        else:
            _logger.info("-" * 73)
            _logger.info(f"detecting cosmic rays in median2d using {rlabel_mmcosmic}...")
        if isinstance(mm_fixed_points_in_boundary, str):
            if mm_fixed_points_in_boundary.lower() == "none":
                mm_fixed_points_in_boundary = None
        if mm_fixed_points_in_boundary is None:
            if mm_boundary_fit == "piecewise":
                raise ValueError("For mm_boundary_fit='piecewise', " "mm_fixed_points_in_boundary must be provided.")
            x_mm_fixed_points_in_boundary = None
            y_mm_fixed_points_in_boundary = None
            w_mm_fixed_points_in_boundary = None
        else:
            mm_fixed_points_in_boundary = list(eval(str(mm_fixed_points_in_boundary)))
            x_mm_fixed_points_in_boundary = []
            y_mm_fixed_points_in_boundary = []
            w_mm_fixed_points_in_boundary = []
            for item in mm_fixed_points_in_boundary:
                if not (isinstance(item, (list, tuple)) and len(item) in [2, 3]):
                    raise ValueError(
                        "Each item in mm_fixed_points_in_boundary must be a list or tuple of "
                        "2 or 3 elements: (x, y) or (x, y, weight)."
                    )
                if not all_valid_numbers(item):
                    raise ValueError(
                        f"All elements in mm_fixed_points_in_boundary={mm_fixed_points_in_boundary} "
                        "must be valid numbers."
                    )
                if len(item) == 2:
                    x_mm_fixed_points_in_boundary.append(float(item[0]))
                    y_mm_fixed_points_in_boundary.append(float(item[1]))
                    w_mm_fixed_points_in_boundary.append(DEFAULT_WEIGHT_FIXED_POINTS_IN_BOUNDARY)
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
        if mm_boundary_fit == "piecewise":
            if mm_fixed_points_in_boundary is None:
                raise ValueError("For mm_boundary_fit='piecewise', " "mm_fixed_points_in_boundary must be provided.")
            elif len(x_mm_fixed_points_in_boundary) < 2:
                raise ValueError(
                    "For mm_boundary_fit='piecewise', "
                    "at least two fixed points must be provided in mm_fixed_points_in_boundary."
                )

        # Compute offsets between each single exposure and the median image
        if isinstance(mm_xy_offsets, str):
            if mm_xy_offsets.lower() == "none":
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
                    if not isinstance(offset, (list, tuple)) or len(offset) != 2 or not all_valid_numbers(offset):
                        raise ValueError(
                            f"Invalid offset in mm_xy_offsets: {offset}. "
                            "Each offset must be a tuple or list of two numbers (x_offset, y_offset)."
                        )
                list_yx_offsets = [(float(offset[1]), float(offset[0])) for offset in mm_xy_offsets]
                for i, yx_offsets in enumerate(list_yx_offsets):
                    _logger.info("provided offsets for image %d: y=%+f, x=%+f", i + 1, yx_offsets[0], yx_offsets[1])
            else:
                raise TypeError(
                    f"Invalid type for mm_xy_offsets: {type(mm_xy_offsets)}. " "Must be list of [x_offset, y_offset)]."
                )
            shift_images = True
            if mm_synthetic == "single":
                raise NotImplementedError("xy-shifting with mm_synthetic='single' is not implemented.")
        else:
            if isinstance(mm_crosscorr_region, str):
                if mm_crosscorr_region.lower() == "none":
                    mm_crosscorr_region = None
                else:
                    mm_crosscorr_region = ast.literal_eval(mm_crosscorr_region)
            if isinstance(mm_crosscorr_region, list):
                all_integers = all(isinstance(val, int) for val in mm_crosscorr_region)
                if not all_integers:
                    raise TypeError(
                        f"Invalid mm_crosscorr_region: {mm_crosscorr_region}. " "All elements must be integers."
                    )
                if len(mm_crosscorr_region) != 4:
                    raise ValueError(
                        f"Invalid length for mm_crosscorr_region: {mm_crosscorr_region}. "
                        "Must be a list of 4 integers [xmin, xmax, ymin, ymax]."
                    )
                dumreg = (
                    f"[{mm_crosscorr_region[0]}:{mm_crosscorr_region[1]}, "
                    + f"{mm_crosscorr_region[2]}:{mm_crosscorr_region[3]}]"
                )
                _logger.debug("defined mm_crosscorr_region: %s", dumreg)
                crossregion = tea.SliceRegion2D(dumreg, mode="fits", naxis1=naxis1, naxis2=naxis2)
                if crossregion.area() < 100:
                    raise ValueError("The area of mm_crosscorr_region must be at least 100 pixels.")
                shift_images = True
                if mm_synthetic == "single":
                    raise NotImplementedError("xy-shifting with mm_synthetic='single' is not implemented.")
            elif mm_crosscorr_region is None:
                crossregion = None
            else:
                raise TypeError(
                    f"Invalid type for mm_crosscorr_region: {type(mm_crosscorr_region)}. "
                    "Must be list of 4 integers or None."
                )
            list_yx_offsets = []
            for i in range(num_images):
                if crossregion is None:
                    list_yx_offsets.append((0.0, 0.0))
                else:
                    reference_image = median2d[crossregion.python]
                    moving_image = image3d[i][crossregion.python]
                    _logger.info("computing offsets for image %d using cross-correlation...", i + 1)
                    yx_offsets, _, _ = phase_cross_correlation(
                        reference_image=reference_image,
                        moving_image=moving_image,
                        upsample_factor=100,
                        normalization=None,  # use None to avoid artifacts with images with many cosmic rays
                    )
                    yx_offsets[0] *= -1  # invert sign
                    yx_offsets[1] *= -1  # invert sign
                    _logger.info("offsets for image %d: y=%+f, x=%+f", i + 1, yx_offsets[0], yx_offsets[1])

                    def on_key_cross(event):
                        if event.key == "x":
                            _logger.info("Exiting program as per user request ('x' key pressed).")
                            plt.close(fig)
                            sys.exit(0)
                        elif event.key == "c":
                            _logger.info("Closing figure as per user request ('c' key pressed).")
                            plt.close(fig)

                    fig, axarr = plt.subplots(
                        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6.4 * 1.5, 4.8 * 1.5)
                    )
                    fig.canvas.mpl_connect("key_press_event", lambda event: on_key_cross(event))
                    axarr = axarr.flatten()
                    vmin = np.min(reference_image)
                    vmax = np.max(reference_image)
                    extent = [
                        crossregion.fits[0].start - 0.5,
                        crossregion.fits[0].stop + 0.5,
                        crossregion.fits[1].start - 0.5,
                        crossregion.fits[1].stop + 0.5,
                    ]
                    tea.imshow(
                        fig,
                        axarr[0],
                        reference_image,
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        aspect="auto",
                        title="Median",
                    )
                    tea.imshow(
                        fig,
                        axarr[1],
                        moving_image,
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        aspect="auto",
                        title=f"Image {i+1}",
                    )
                    shifted_image2d = shift_image2d(
                        moving_image, xoffset=-yx_offsets[1], yoffset=-yx_offsets[0], resampling=2
                    )
                    dumdiff1 = reference_image - moving_image
                    dumdiff2 = reference_image - shifted_image2d
                    vmin = np.percentile(dumdiff1, 5)
                    vmax = np.percentile(dumdiff2, 95)
                    tea.imshow(
                        fig,
                        axarr[2],
                        dumdiff1,
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        aspect="auto",
                        title=f"Median - Image {i+1}",
                    )
                    tea.imshow(
                        fig,
                        axarr[3],
                        dumdiff2,
                        vmin=vmin,
                        vmax=vmax,
                        extent=extent,
                        aspect="auto",
                        title=f"Median - Shifted Image {i+1}",
                    )
                    plt.tight_layout()
                    png_filename = f"xyoffset_crosscorr_{i+1}.png"
                    _logger.info(f"saving {png_filename}")
                    plt.savefig(png_filename, dpi=150)
                    if interactive:
                        _logger.info("Entering interactive mode (press 'c' to continue, 'x' to quit program)")
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
            npixels=10000,
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
        if mm_ydiag_max is not None:
            if mm_ydiag_max > 0:
                ydiag_max = mm_ydiag_max
                _logger.info("using user-defined mm_ydiag_max=%f", mm_ydiag_max)
        xylim_table = Table(
            names=["xdiag_min", "xdiag_max", "ydiag_min", "ydiag_max"], dtype=[float, float, float, float]
        )
        for colname in xylim_table.colnames:
            xylim_table[colname].format = ".3f"
        xylim_table.add_row([xdiag_min, xdiag_max, ydiag_min, ydiag_max])
        _logger.info("diagnostic diagram limits:\n%s", str(xylim_table))

        # Define binning for the diagnostic plot
        nbins_xdiag = 100
        nbins_ydiag = 100
        bins_xdiag = np.linspace(xdiag_min, xdiag_max, nbins_xdiag + 1)
        bins_ydiag = np.linspace(0, ydiag_max, nbins_ydiag + 1)

        # Create a 2D histogram for the diagnostic plot, using
        # integers to avoid rounding errors
        hist2d_accummulated = np.zeros((nbins_ydiag, nbins_xdiag), dtype=int)
        if apply_flux_factor_to == "original":
            flux_factor_for_simulated = np.ones(num_images, dtype=float)
        elif apply_flux_factor_to == "simulated":
            flux_factor_for_simulated = flux_factor
        else:
            raise ValueError(
                f"Invalid apply_flux_factor_to: {apply_flux_factor_to}. "
                "Valid options are 'original' and 'simulated'."
            )
        _logger.info("flux factor for simulated images: %s", str(flux_factor_for_simulated))
        lam3d = np.zeros((num_images, naxis2, naxis1))
        if not shift_images:
            _logger.info("assuming no offsets between images")
            for i in range(num_images):
                if mm_synthetic == "single":
                    lam3d[i] = image3d_cleaned_single[i] / flux_factor_for_simulated[i]
                elif mm_synthetic == "median":
                    lam3d[i] = median2d_precleaned / flux_factor_for_simulated[i]
                else:
                    raise ValueError(f"Invalid mm_synthetic: {mm_synthetic}.")
        else:
            _logger.info("xy-shifting median2d to speed up simulations...")
            for i in range(num_images):
                _logger.info(
                    "shifted image %d/%d -> delta_y=%+f, delta_x=%+f",
                    i + 1,
                    num_images,
                    list_yx_offsets[i][0],
                    list_yx_offsets[i][1],
                )
                # apply offsets to the median image to simulate the expected individual exposures
                lam3d[i] = (
                    shift_image2d(
                        median2d_precleaned,
                        xoffset=list_yx_offsets[i][1],
                        yoffset=list_yx_offsets[i][0],
                        resampling=2,
                    )
                    / flux_factor_for_simulated[i]
                )
                # replace any NaN values introduced by the shift with zeros
                lam3d[i] = np.nan_to_num(lam3d[i], nan=0.0)
        # replace any negative values with zeros
        for i in range(num_images):
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
            # Avoid pixels that were flagged as cosmic rays in the median and single images
            xplot_simul = xplot_simul[flag2d_cleaned_single == 0]
            yplot_simul = yplot_simul[flag2d_cleaned_single == 0]
            hist2d, edges = np.histogramdd(sample=(yplot_simul, xplot_simul), bins=(bins_ydiag, bins_xdiag))
            hist2d_accummulated += hist2d.astype(int)
            time_end = datetime.now()
            nchars = len(str(mm_nsimulations))
            _logger.info(f"simulation {k + 1:0{nchars}d}/{mm_nsimulations}, time elapsed: {time_end - time_ini}")
        # Display hist2d
        result_hist2d = display_hist2d(
            _logger=_logger,
            rlabel_mmcosmic=rlabel_mmcosmic,
            mm_hist2d_min_neighbors=mm_hist2d_min_neighbors,
            hist2d_accummulated=hist2d_accummulated,
            mm_nsimulations=mm_nsimulations,
            mm_synthetic=mm_synthetic,
            bins_xdiag=bins_xdiag,
            bins_ydiag=bins_ydiag,
            xplot=xplot,
            yplot=yplot,
            xdiag_min=xdiag_min,
            xdiag_max=xdiag_max,
            max2d=max2d,
            bool_to_be_cleaned=bool_to_be_cleaned,
            rnoise=rnoise,
            mm_threshold=mm_threshold,
            mm_boundary_fit=mm_boundary_fit,
            mm_knots_splfit=mm_knots_splfit,
            mm_minimum_max2d_rnoise=mm_minimum_max2d_rnoise,
            mm_dilation=mm_dilation,
            mm_niter_boundary_extension=mm_niter_boundary_extension,
            mm_weight_boundary_extension=mm_weight_boundary_extension,
            mm_fixed_points_in_boundary=mm_fixed_points_in_boundary,
            x_mm_fixed_points_in_boundary=x_mm_fixed_points_in_boundary,
            y_mm_fixed_points_in_boundary=y_mm_fixed_points_in_boundary,
            w_mm_fixed_points_in_boundary=w_mm_fixed_points_in_boundary,
            interactive=interactive,
        )
        # retrieve main results
        xplot_boundary = result_hist2d["xplot_boundary"]
        yplot_boundary = result_hist2d["yplot_boundary"]
        boundaryfit = result_hist2d["boundaryfit"]
        flag_mm = result_hist2d["flag_mm"]
        # update additional parameters that might have been modified
        mm_fixed_points_in_boundary = result_hist2d["mm_fixed_points_in_boundary"]
        mm_hist2d_min_neighbors = result_hist2d["mm_hist2d_min_neighbors"]
        mm_boundary_fit = result_hist2d["mm_boundary_fit"]
        mm_knots_splfit = result_hist2d["mm_knots_splfit"]
        mm_niter_boundary_extension = result_hist2d["mm_niter_boundary_extension"]
        mm_weight_boundary_extension = result_hist2d["mm_weight_boundary_extension"]
    elif crmethod in ["lacosmic", "pycosmic", "deepcr", "conn"]:
        xplot_boundary = None
        yplot_boundary = None
        flag_mm = np.zeros_like(median2d, dtype=bool).flatten()
    else:
        raise ValueError(f"Invalid crmethod: {crmethod}.\nValid options are: {VALID_CRMETHODS}.")

    # #######################################################################################
    # Combine the flags from the auxiliary method (lacosmic | pycosmic | deepcr | conn)
    # and mmcosmic
    # #######################################################################################
    if flag_aux is None and flag_mm is None:
        raise RuntimeError("Both flag_aux and flag_mm are None. This should never happen.")
    elif flag_aux is None:
        flag = flag_mm
        flag_integer = 2 * flag_mm.astype(np.uint8)
    elif flag_mm is None:
        flag = flag_aux
        flag_integer = 3 * flag_aux.astype(np.uint8)
    else:
        # Combine the flags from [lacosmic|pycosmic] and mmcosmic
        flag = np.logical_or(flag_aux, flag_mm)
        flag_integer = 2 * flag_mm.astype(np.uint8) + 3 * flag_aux.astype(np.uint8)
        sdum = str(np.sum(flag))
        cdum = f"{np.sum(flag):{len(sdum)}d}"
        _logger.info(
            "pixels flagged as cosmic rays by %s or %s : %s (%08.4f%%)",
            rlabel_aux,
            rlabel_mmcosmic,
            cdum,
            np.sum(flag) / flag.size * 100,
        )
        cdum = f"{np.sum(flag_integer == 3):{len(sdum)}d}"
        _logger.info(
            "pixels flagged as cosmic rays by %s only........: %s (%08.4f%%)",
            rlabel_aux,
            cdum,
            np.sum(flag_integer == 3) / flag.size * 100,
        )
        cdum = f"{np.sum((flag_integer == 2)):{len(sdum)}d}"
        _logger.info(
            "pixels flagged as cosmic rays by %s only........: %s (%08.4f%%)",
            rlabel_mmcosmic,
            cdum,
            np.sum(flag_integer == 2) / flag.size * 100,
        )
        cdum = f"{np.sum((flag_integer == 5)):{len(sdum)}d}"
        _logger.info(
            "pixels flagged as cosmic rays by %s and %s: %s (%08.4f%%)",
            rlabel_aux,
            rlabel_mmcosmic,
            cdum,
            np.sum((flag_integer == 5)) / flag.size * 100,
        )
    flag = flag.reshape((naxis2, naxis1))
    flag_integer = flag_integer.reshape((naxis2, naxis1))
    flag_integer[flag_integer == 5] = 4  # pixels flagged by both methods are set to 4

    # Show diagnostic plot for the cosmic ray detection
    _logger.info("generating diagnostic plot for MEDIANCR...")
    ylabel = r"median2d $-$ min2d"
    diagnostic_plot(
        xplot=xplot,
        yplot=yplot,
        xplot_boundary=xplot_boundary,
        yplot_boundary=yplot_boundary,
        rlabel_aux_plain=rlabel_aux_plain,
        flag_aux=flag_aux,
        flag_mm=flag_mm,
        mm_threshold=mm_threshold,
        ylabel=ylabel,
        interactive=interactive,
        target2d=median2d,
        target2d_name="median2d",
        min2d=min2d,
        mean2d=mean2d,
        image3d=image3d,
        _logger=_logger,
        png_filename="diagnostic_mediancr.png",
    )

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
                flag_integer, structure=structure, iterations=dilation
            ).astype(np.uint8)
            sdum = str(np.sum(flag_integer_dilated > 0))
            cdum = f"{np.sum(flag_integer > 0):{len(sdum)}d}"
            _logger.info("before global dilation: %s pixels flagged as coincident CR pixels", cdum)
            cdum = f"{np.sum(flag_integer_dilated > 0):{len(sdum)}d}"
            _logger.info("after global dilation : %s pixels flagged as coincident CR pixels", cdum)
        else:
            flag_integer_dilated = flag_integer
            _logger.info(
                "no global dilation applied: %d pixels flagged as coincident CR pixels",
                np.sum(flag_integer > 0),
            )
        # Set the pixels that were originally flagged as cosmic rays
        # to the integer value before dilation (this is to distinguish them
        # from the pixels that were dilated,which will be set to 1)
        flag_integer_dilated[flag] = flag_integer[flag]
        # Compute mask
        mask_mediancr = flag_integer_dilated > 0
        # Fix the median2d array by replacing the flagged pixels with the minimum value
        # of the corresponding pixel in the input arrays
        median2d_corrected = median2d.copy()
        if use_auxmedian:
            median2d_corrected[mask_mediancr] = median2d_aux[mask_mediancr]
        else:
            median2d_corrected[mask_mediancr] = min2d[mask_mediancr]
        # Label the connected pixels as individual cosmic rays
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        labels_cr, number_cr = ndimage.label(flag_integer_dilated > 0, structure=structure)
        _logger.info("number of grouped cosmic-ray pixels detected: %d", number_cr)
        display_detected_cr(
            num_images=num_images,
            image3d=image3d,
            median2d=median2d,
            median2d_corrected=median2d_corrected,
            flag_integer_dilated=flag_integer_dilated,
            acronym_aux=acronym_aux,
            labels_cr=labels_cr,
            number_cr=number_cr,
            mask_mediancr=mask_mediancr,
            list_mask_single_exposures=None,
            mask_all_singles=None,
            semiwindow=semiwindow,
            maxplots=maxplots,
            verify_cr=verify_cr,
            color_scale=color_scale,
            xplot=xplot,
            yplot=yplot,
            xplot_boundary=xplot_boundary,
            yplot_boundary=yplot_boundary,
            mm_threshold=mm_threshold,
            _logger=_logger,
        )

    # Generate list of HDUs with masks
    hdu_mediancr = fits.ImageHDU(mask_mediancr.astype(np.uint8), name="MEDIANCR")
    list_hdu_masks = [hdu_mediancr]

    # Apply the same algorithm but now with mean2d and with each individual array
    for i, target2d in enumerate([mean2d] + list_arrays + [None]):
        if i == 0:
            target2d_name = "mean2d"
        elif i <= num_images:
            target2d_name = f"single exposure #{i}"
        elif i == num_images + 1:
            if rich_configured:
                _logger.info("[green]" + "-" * 79 + "[/green]")
            else:
                _logger.info("-" * 73)
            _logger.info("applying CRMASKi masks to generate MEANCR combination...")
            target2d_name = "MEANCR"
            hdu_primary = fits.PrimaryHDU()
            hdu_primary.header["UUID"] = str(uuid.uuid4())
            for ifluxf, fluxf in enumerate(flux_factor):
                hdu_primary.header[f"FLUXF{ifluxf+1}"] = fluxf
            hdul_masks = fits.HDUList([hdu_primary] + list_hdu_masks)
            target2d, _, _ = apply_crmasks(
                list_arrays=list_arrays,
                hdul_masks=hdul_masks,
                combination="meancr",
                use_auxmedian=False,
                dtype=dtype,
                apply_flux_factor=(apply_flux_factor_to == "original"),
                bias=bias,
            )
        else:
            raise RuntimeError("This should never happen.")
        if rich_configured:
            _logger.info("[green]" + "-" * 79 + "[/green]")
            _logger.info(f"starting cosmic ray detection in [magenta]{target2d_name}[/magenta]...")
        else:
            _logger.info("-" * 73)
            _logger.info(f"starting cosmic ray detection in {target2d_name}...")

        if crmethod in ["lacosmic", "mm_lacosmic"]:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_lacosmic}...")
            if i in [0, num_images + 1]:
                _, flag_aux = execute_lacosmic(
                    image2d=target2d,
                    bool_to_be_cleaned=bool_to_be_cleaned,
                    rlabel_lacosmic=rlabel_lacosmic,
                    dict_la_params_run1=dict_la_params_run1,
                    dict_la_params_run2=dict_la_params_run2,
                    la_padwidth=la_padwidth,
                    displaypar=False,
                    _logger=_logger,
                )
            else:
                _logger.info(f"image #{i} already cleaned, retrieving flags from previous step...")
                flag_aux = flag3d_cleaned_single[i - 1].flatten().astype(bool)
        elif crmethod in ["pycosmic", "mm_pycosmic"]:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_pycosmic}...")
            if i in [0, num_images + 1]:
                _, flag_aux = execute_pycosmic(
                    image2d=target2d,
                    bool_to_be_cleaned=bool_to_be_cleaned,
                    rlabel_pycosmic=rlabel_pycosmic,
                    dict_pc_params_run1=dict_pc_params_run1,
                    dict_pc_params_run2=dict_pc_params_run2,
                    displaypar=False,
                    _logger=_logger,
                )
            else:
                _logger.info(f"image #{i} already cleaned, retrieving flags from previous step...")
                flag_aux = flag3d_cleaned_single[i - 1].flatten().astype(bool)
        elif crmethod in ["deepcr", "mm_deepcr"]:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_deepcr}...")
            if i in [0, num_images + 1]:
                _, flag_aux = execute_deepcr(
                    image2d=target2d,
                    bool_to_be_cleaned=bool_to_be_cleaned,
                    rlabel_deepcr=rlabel_deepcr,
                    dict_dc_params=dict_dc_params,
                    displaypar=False,
                    _logger=_logger,
                )
            else:
                _logger.info(f"image #{i} already cleaned, retrieving flags from previous step...")
                flag_aux = flag3d_cleaned_single[i - 1].flatten().astype(bool)
        elif crmethod in ["conn", "mm_conn"]:
            _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_conn}...")
            if i in [0, num_images + 1]:
                flag_aux = execute_conn(
                    image2d=target2d,
                    bool_to_be_cleaned=bool_to_be_cleaned,
                    rlabel_conn=rlabel_conn,
                    dict_nn_params=dict_nn_params,
                    displaypar=False,
                    _logger=_logger,
                )
            else:
                _logger.info(f"image #{i} already cleaned, retrieving flags from previous step...")
                flag_aux = flag3d_cleaned_single[i - 1].flatten().astype(bool)
        else:
            raise ValueError(f"Invalid crmethod: {crmethod}.\n" f"Valid options are: {VALID_CRMETHODS}.")

        if i <= num_images:
            if crmethod in ["mm_lacosmic", "mm_pycosmic", "mm_deepcr", "mm_conn"]:
                _logger.info(f"detecting cosmic rays in {target2d_name} using {rlabel_mmcosmic}...")
                xplot = min2d.flatten()
                yplot = target2d.flatten() - min2d.flatten()
                flag1 = yplot > boundaryfit(xplot)
                flag2 = yplot > mm_threshold
                flag_mm = np.logical_and(flag1, flag2)
                flag3 = max2d.flatten() > mm_minimum_max2d_rnoise * rnoise.flatten()
                flag_mm = np.logical_and(flag_mm, flag3)
                flag_mm = np.logical_and(flag_mm, bool_to_be_cleaned.flatten())
                if mm_dilation > 0:
                    _logger.info("applying binary dilation with size=%d to cosmic-ray mask", mm_dilation)
                    num_pixels_before_dilation = np.sum(flag_mm)
                    structure = ndimage.generate_binary_structure(2, 2)
                    flag_mm = ndimage.binary_dilation(
                        flag_mm.reshape((naxis2, naxis1)), structure=structure, iterations=mm_dilation
                    ).flatten()
                    num_pixels_after_dilation = np.sum(flag_mm)
                    ldum = len(str(num_pixels_after_dilation))
                    _logger.info(f"number of pixels flagged before dilation : {num_pixels_before_dilation:{ldum}d}")
                    _logger.info(f"number of pixels flagged after dilation  : {num_pixels_after_dilation:{ldum}d}")
                    # note: there is no need to apply now thresholding again because all the
                    # dilated pixels are included in CRs that contain at least one pixel
                    # above the threshold
                # Apply thresholding in M.M. diagram to help removing false positives in auxiliary method
                _logger.info(f"applying mm_threshold={mm_threshold} to auxiliary method {rlabel_aux}...")
                npixels_found_before = np.sum(flag_aux)
                flag_aux = apply_threshold_cr(
                    bool_crmask2d=flag_aux.reshape((naxis2, naxis1)),
                    bool_threshold2d=(yplot > mm_threshold).reshape((naxis2, naxis1)),
                ).flatten()
                npixels_found_after = np.sum(flag_aux)
                ldum = len(str(npixels_found_before))
                _logger.info(f"number of CR pixels before applying mm_threshold: {npixels_found_before:>{ldum}d}")
                _logger.info(f"number of CR pixels after  applying mm_threshold: {npixels_found_after:>{ldum}d}")
            else:
                xplot_boundary = None
                yplot_boundary = None
                flag_mm = np.zeros_like(target2d.flatten(), dtype=bool)
        else:
            xplot_boundary = None
            yplot_boundary = None
            flag_mm = np.zeros_like(target2d.flatten(), dtype=bool)

        # For the mean2d mask, force the flag to be True if the pixel
        # was flagged as a coincident cosmic-ray pixel when using the median2d array
        # (this is to ensure that all pixels flagged in MEDIANCR are also
        # flagged in MEANCRT)
        if i == 0:
            _logger.info("including pixels flagged in MEDIANCR into MEANCRT (logical_or)...")
            if np.any(flag_aux):
                flag_aux = np.logical_or(flag_aux, list_hdu_masks[0].data.astype(bool).flatten())
            if np.any(flag_mm):
                flag_mm = np.logical_or(flag_mm, list_hdu_masks[0].data.astype(bool).flatten())
        # For the individual array masks, force the flag to be True if the pixel
        # is flagged both in the individual exposure and in the mean2d array
        if 1 <= i <= num_images:
            _logger.info("including pixels flagged in MEANCRT into CRMASK%d (logical_and)...", i)
            flag_aux = np.logical_and(flag_aux, list_hdu_masks[1].data.astype(bool).flatten())
            flag_mm = np.logical_and(flag_mm, list_hdu_masks[1].data.astype(bool).flatten())
        sflag_aux = str(np.sum(flag_aux))
        sflag_mm = str(np.sum(flag_mm))
        smax = max(len(sflag_aux), len(sflag_mm))
        _logger.info(
            "pixels flagged as cosmic rays by %s: %s (%08.4f%%)",
            rlabel_aux,
            f"{np.sum(flag_aux):{smax}d}",
            np.sum(flag_aux) / flag_aux.size * 100,
        )
        _logger.info(
            "pixels flagged as cosmic rays by %s: %s (%08.4f%%)",
            rlabel_mmcosmic,
            f"{np.sum(flag_mm):{smax}d}",
            np.sum(flag_mm) / flag_mm.size * 100,
        )
        if debug:
            if i == 0:
                _logger.info("generating diagnostic plot for MEANCRT...")
                png_filename = "diagnostic_meancrt.png"
                ylabel = r"mean2d $-$ min2d"
            elif 1 <= i <= num_images:
                _logger.info(f"generating diagnostic plot for CRMASK{i}...")
                png_filename = f"diagnostic_crmask{i}.png"
                ylabel = f"array{i}" + r" $-$ min2d"
            elif i == num_images + 1:
                _logger.info(f"generating diagnostic plot for MEANCR...")
                png_filename = f"diagnostic_meancr.png"
                ylabel = r"MEANCR $-$ min2d"
            else:
                raise RuntimeError("This should never happen.")
            diagnostic_plot(
                xplot=xplot,
                yplot=yplot,
                xplot_boundary=xplot_boundary,
                yplot_boundary=yplot_boundary,
                rlabel_aux_plain=rlabel_aux_plain,
                flag_aux=flag_aux,
                flag_mm=flag_mm,
                mm_threshold=mm_threshold,
                ylabel=ylabel,
                interactive=interactive,
                target2d=target2d,
                target2d_name=target2d_name,
                min2d=min2d,
                mean2d=mean2d,
                image3d=image3d,
                _logger=_logger,
                png_filename=png_filename,
            )
        flag = np.logical_or(flag_aux, flag_mm)
        flag = flag.reshape((naxis2, naxis1))
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            if i == num_images + 1:
                _logger.warning("ignoring dilation for MEANCR mask!")
            else:
                structure = ndimage.generate_binary_structure(2, 2)
                flag_integer_dilated = ndimage.binary_dilation(
                    flag_integer, structure=structure, iterations=dilation
                ).astype(np.uint8)
                sdum = str(np.sum(flag_integer_dilated))
                cdum = f"{np.sum(flag_integer):{len(sdum)}d}"
                _logger.info("before global dilation: %s pixels flagged as cosmic rays", cdum)
                cdum = f"{np.sum(flag_integer_dilated):{len(sdum)}d}"
                _logger.info("after global dilation : %s pixels flagged as cosmic rays", cdum)
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no global dilation applied: %d pixels flagged as cosmic rays", np.sum(flag_integer))
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask = flag_integer_dilated > 0
        if i == 0:
            name = "MEANCRT"
        elif i <= num_images:
            name = f"CRMASK{i}"
        elif i == num_images + 1:
            name = "MEANCR"
        else:
            raise RuntimeError("This should never happen.")
        hdu_mask = fits.ImageHDU(mask.astype(np.uint8), name=name)
        list_hdu_masks.append(hdu_mask)

    # Include auxiliary-corrected median2d image if applicable
    if median2d_aux is not None:
        hdu_median2d_aux = fits.ImageHDU(median2d_aux.astype(np.float32), name="AUXCLEAN")
        list_hdu_masks.append(hdu_median2d_aux)

    # Find problematic cosmic-ray pixels (those masked in all individual CRMASKi)
    mask_all_singles = np.ones((naxis2, naxis1), dtype=bool)
    for hdu in list_hdu_masks[2:]:
        mask_all_singles = np.logical_and(mask_all_singles, hdu.data.astype(bool))
    problematic_pixels = np.argwhere(mask_all_singles)
    if rich_configured:
        _logger.info("[green]" + "-" * 79 + "[/green]")
    else:
        _logger.info("-" * 73)
    _logger.info("number of problematic cosmic-ray pixels masked in all CRMASKi: %d", len(problematic_pixels))
    if len(problematic_pixels) > 0:
        # Label the connected problematic pixels as individual problematic cosmic rays
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        labels_cr, number_cr = ndimage.label(mask_all_singles, structure=structure)
        _logger.info("number of grouped problematic cosmic-ray pixels: %d", number_cr)
        display_detected_cr(
            num_images=num_images,
            image3d=image3d,
            median2d=median2d,
            median2d_corrected=median2d_corrected,
            flag_integer_dilated=flag_integer_dilated,
            acronym_aux=acronym_aux,
            labels_cr=labels_cr,
            number_cr=number_cr,
            mask_mediancr=mask_mediancr,
            list_mask_single_exposures=[hdu.data for hdu in list_hdu_masks[2 : 2 + num_images]],
            mask_all_singles=mask_all_singles,
            semiwindow=semiwindow,
            maxplots=maxplots,
            verify_cr=False,
            color_scale=color_scale,
            xplot=None,
            yplot=None,
            xplot_boundary=None,
            yplot_boundary=None,
            _logger=_logger,
        )

    # Generate output HDUList with masks
    args = inspect.signature(compute_crmasks).parameters
    filtered_args = {
        k: v
        for k, v in locals().items()
        if k in args and k not in ["list_arrays"] and k[:3] not in prefix_of_excluded_args
    }
    hdu_primary = fits.PrimaryHDU()
    hdu_primary.header["UUID"] = str(uuid.uuid4())
    for ifluxf, fluxf in enumerate(flux_factor):
        hdu_primary.header[f"FLUXF{ifluxf+1}"] = fluxf
    hdu_primary.header.add_history(f"CRMasks generated by {__name__}")
    hdu_primary.header.add_history(f"at {datetime.now().isoformat()}")
    for key, value in filtered_args.items():
        if isinstance(value, np.ndarray):
            if np.unique(value).size == 1:
                value = value.flatten()[0]
            elif value.ndim == 1 and len(value) == num_images:
                value = str(value.tolist())
            else:
                value = f"array_shape: {value.shape}"
        elif isinstance(value, list):
            value = str(value)
        hdu_primary.header.add_history(f"- {key} = {value}")
    # Include extension with pixels to be replaced by the median value around them
    # (stored as a binary table with four columns: 'X_pixel', 'Y_pixel', 'X_width', 'Y_width')
    if pixels_to_be_replaced_by_local_median is not None:
        col1 = fits.Column(name="X_pixel", format="K", array=[p[0] for p in pixels_to_be_replaced_by_local_median])
        col2 = fits.Column(name="Y_pixel", format="K", array=[p[1] for p in pixels_to_be_replaced_by_local_median])
        col3 = fits.Column(name="X_width", format="K", array=[p[2] for p in pixels_to_be_replaced_by_local_median])
        col4 = fits.Column(name="Y_width", format="K", array=[p[3] for p in pixels_to_be_replaced_by_local_median])
        hdu_table = fits.BinTableHDU.from_columns([col1, col2, col3, col4], name="RPMEDIAN")
        list_hdu_masks.append(hdu_table)

    hdul_masks = fits.HDUList([hdu_primary] + list_hdu_masks)
    return hdul_masks
