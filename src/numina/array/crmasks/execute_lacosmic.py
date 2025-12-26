#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Execute L.A.Cosmic cosmic ray detection algorithm."""

import io
import logging

from ccdproc import cosmicray_lacosmic
from contextlib import redirect_stderr, redirect_stdout
import numpy as np

from teareduce import cleanest


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
        if output_lines and output_lines[-1] == "":
            output_lines = output_lines[:-1]
        if output_lines:
            _logger.info("\n" + "\n".join(output_lines))
        return result

    return wrapper


@decorate_output
def decorated_cosmicray_lacosmic(*args, **kwargs):
    """Wrapper for cosmicray_lacosmic with decorated output."""
    return cosmicray_lacosmic(*args, **kwargs)


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
    if la_verbose:
        _logger.info("[green][LACOSMIC parameters for run 1][/green]")
        for key in dict_la_params_run1.keys():
            if key == "psfk":
                if dict_la_params_run1[key] is None:
                    _logger.info("%s for lacosmic: None", key)
                else:
                    _logger.info("%s for lacosmic: array with shape %s", key, str(dict_la_params_run1[key].shape))
            else:
                _logger.info("%s for lacosmic: %s", key, str(dict_la_params_run1[key]))
        if lacosmic_needs_2runs:
            _logger.info("[green][LACOSMIC parameters modified for run 2][/green]")
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
        _logger.info("LACOSMIC will be run in 2 passes with modified parameters.")
    else:
        _logger.info("LACOSMIC will be run in a single pass.")
    _logger.info(f"detecting cosmic rays in image2d image using {rlabel_lacosmic}...")
    # run 1
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
            print(f"{image2d_lacosmic2.shape=}, {flag_la2.shape=}")
        # combine results from both runs
        flag_la = cleanest.merge_peak_tail_masks(flag_la, flag_la2, la_verbose)
        image2d_lacosmic = image2d_lacosmic2  # use the result from the 2nd run
    _logger.info(
        "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
        rlabel_lacosmic,
        np.sum(flag_la),
        np.sum(flag_la) / flag_la.size * 100,
    )
    flag_la = np.logical_and(flag_la, bool_to_be_cleaned)
    flag_la = flag_la.flatten()

    return image2d_lacosmic, flag_la
