#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Execute L.A.Cosmic cosmic ray detection algorithm."""

from importlib.metadata import version

from datetime import datetime
from ccdproc import cosmicray_lacosmic
import numpy as np

from teareduce import cleanest

from .decorated_output import decorate_output


@decorate_output
def decorated_cosmicray_lacosmic(*args, **kwargs):
    """Wrapper for cosmicray_lacosmic with decorated output."""
    return cosmicray_lacosmic(*args, **kwargs)

@decorate_output
def decorated_merge_peak_tail_masks(*args, **kwargs):
    """Wrapper for merge_peak_tail_masks with decorated output."""
    return cleanest.merge_peak_tail_masks(*args, **kwargs)

def execute_lacosmic(
    image2d, bool_to_be_cleaned, rlabel_lacosmic, dict_la_params_run1, dict_la_params_run2, la_padwidth, _logger
):
    """Execute L.A.Cosmic cosmic ray detection algorithm."""
    la_verbose = dict_la_params_run1["verbose"] or dict_la_params_run2["verbose"]
    # Determine if 2 runs are needed
    la_sigclip_needs_2runs = False
    if dict_la_params_run1["sigclip"] != dict_la_params_run2["sigclip"]:
        la_sigclip_needs_2runs = True
    la_sigfrac_needs_2runs = False
    if dict_la_params_run1["sigfrac"] != dict_la_params_run2["sigfrac"]:
        la_sigfrac_needs_2runs = True
    la_objlim_needs_2runs = False
    if dict_la_params_run1["objlim"] != dict_la_params_run2["objlim"]:
        la_objlim_needs_2runs = True
    lacosmic_needs_2runs = la_sigclip_needs_2runs or la_sigfrac_needs_2runs or la_objlim_needs_2runs
    # Display parameters
    if la_verbose:
        _logger.info("[green][L.A.Cosmic parameters for run 1][/green]")
        for key in dict_la_params_run1.keys():
            if key == "psfk":
                if dict_la_params_run1[key] is None:
                    _logger.info("%s for lacosmic: None", key)
                else:
                    _logger.info("%s for lacosmic: array with shape %s", key, str(dict_la_params_run1[key].shape))
            else:
                _logger.info("%s for lacosmic: %s", key, str(dict_la_params_run1[key]))
        if lacosmic_needs_2runs:
            _logger.info("[green][L.A.Cosmic parameters modified for run 2][/green]")
            if la_sigclip_needs_2runs:
                _logger.info(
                    "la_sigclip for run 2 (run1): %f (%f)",
                    dict_la_params_run2["sigclip"],
                    dict_la_params_run1["sigclip"],
                )
            if la_sigfrac_needs_2runs:
                _logger.info(
                    "la_sigfrac for run 2 (run1): %f (%f)",
                    dict_la_params_run2["sigfrac"],
                    dict_la_params_run1["sigfrac"],
                )
            if la_objlim_needs_2runs:
                _logger.info(
                    "la_objlim for run 2 (run1): %f (%f)",
                    dict_la_params_run2["objlim"],
                    dict_la_params_run1["objlim"],
                )
    if lacosmic_needs_2runs:
        _logger.info("L.A.Cosmic will be run in 2 passes with modified parameters.")
    else:
        _logger.info("L.A.Cosmic will be run in a single pass.")
    _logger.info(f"detecting cosmic rays using {rlabel_lacosmic}...")
    # Detect ccdproc version
    try:
        version_ccdproc = version("ccdproc")
    except Exception:
        version_ccdproc = "unknown"
    _logger.info(f"using ccdproc version: {version_ccdproc}")
    # run 1
    datetime_ini = datetime.now()
    image2d_padded = np.pad(image2d, pad_width=la_padwidth, mode="reflect")
    image2d_lacosmic, flag_la = decorated_cosmicray_lacosmic(
        ccd=image2d_padded, **{key: value for key, value in dict_la_params_run1.items() if value is not None}
    )
    if la_padwidth > 0:
        image2d_lacosmic = image2d_lacosmic[la_padwidth:-la_padwidth, la_padwidth:-la_padwidth]
        flag_la = flag_la[la_padwidth:-la_padwidth, la_padwidth:-la_padwidth]
    # run 2 if needed
    if lacosmic_needs_2runs:
        image2d_lacosmic2, flag_la2 = decorated_cosmicray_lacosmic(
            ccd=image2d_padded,
            **{key: value for key, value in dict_la_params_run2.items() if value is not None},
        )
        if la_padwidth > 0:
            image2d_lacosmic2 = image2d_lacosmic2[la_padwidth:-la_padwidth, la_padwidth:-la_padwidth]
            flag_la2 = flag_la2[la_padwidth:-la_padwidth, la_padwidth:-la_padwidth]
        # combine results from both runs
        flag_la = decorated_merge_peak_tail_masks(flag_la, flag_la2, la_verbose)
        image2d_lacosmic = image2d_lacosmic2  # use the result from the 2nd run
    flag_la = np.logical_and(flag_la, bool_to_be_cleaned)
    flag_la = flag_la.flatten()
    datetime_end = datetime.now()
    delta_datetime = datetime_end - datetime_ini
    _logger.info("L.A.Cosmic execution time: %s", str(delta_datetime))
    _logger.info(
        "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
        rlabel_lacosmic,
        np.sum(flag_la),
        np.sum(flag_la) / flag_la.size * 100,
    )

    return image2d_lacosmic, flag_la
