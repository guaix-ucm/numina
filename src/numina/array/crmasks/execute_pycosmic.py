#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Execute PyCosmic cosmic ray detection algorithm."""

from importlib.metadata import version

from datetime import datetime
import numpy as np

try:
    import PyCosmic

    PYCOSMIC_AVAILABLE = True
except ModuleNotFoundError as e:
    PYCOSMIC_AVAILABLE = False

from teareduce.cleanest.mergemasks import merge_peak_tail_masks

from .decorated_output import decorate_output


@decorate_output(prompt="PyCosmic > ")
def decorated_pycosmic_det_cosmics(*args, **kwargs):
    """Wrapper for PyCosmic.det_cosmics with decorated output."""
    return PyCosmic.det_cosmics(*args, **kwargs)


@decorate_output(prompt="")
def decorated_merge_peak_tail_masks(*args, **kwargs):
    """Wrapper for merge_peak_tail_masks with decorated output."""
    return merge_peak_tail_masks(*args, **kwargs)


def execute_pycosmic(
    image2d, bool_to_be_cleaned, rlabel_pycosmic, dict_pc_params_run1, dict_pc_params_run2, displaypar, _logger
):
    """Execute PyCosmic cosmic ray detection algorithm."""
    if not PYCOSMIC_AVAILABLE:
        raise ImportError(
            "PyCosmic is not installed. Please install PyCosmic to use\n"
            "the 'pycosmic' or 'mm_pycosmic' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install git+https://github.com/nicocardiel/PyCosmic.git@test\n"
        )
    pc_verbose = dict_pc_params_run1["verbose"] or dict_pc_params_run2["verbose"]
    # Determine if 2 runs are needed
    pc_sigma_det_needs_2runs = False
    if dict_pc_params_run1["sigma_det"] != dict_pc_params_run2["sigma_det"]:
        pc_sigma_det_needs_2runs = True
    pc_rlim_needs_2runs = False
    if dict_pc_params_run1["rlim"] != dict_pc_params_run2["rlim"]:
        pc_rlim_needs_2runs = True
    pycosmic_needs_2runs = pc_sigma_det_needs_2runs or pc_rlim_needs_2runs
    if pycosmic_needs_2runs:
        _logger.info("PyCosmic will be run in 2 passes with modified parameters.")
    else:
        _logger.info("PyCosmic will be run in a single pass.")
    # Detect PyCosmic version
    try:
        version_pycosmic = version("PyCosmic")
    except Exception:
        version_pycosmic = "unknown"
    _logger.info(f"using PyCosmic version: {version_pycosmic}")
    # run 1
    if displaypar:
        _logger.info("[green][PyCosmic parameters for run 1][/green]")
        for key in dict_pc_params_run1.keys():
            _logger.info("%s for pycosmic: %s", key, str(dict_pc_params_run1[key]))
    datetime_ini = datetime.now()
    out = decorated_pycosmic_det_cosmics(
        data=image2d, **{key: value for key, value in dict_pc_params_run1.items() if value is not None}
    )
    median2d_pycosmic = out.data
    flag_pc = out.mask.astype(bool)
    # run 2 if needed
    if pycosmic_needs_2runs:
        if displaypar:
            _logger.info("[green][PyCosmic parameters modified for run 2][/green]")
            if pc_sigma_det_needs_2runs:
                _logger.info(
                    "pc_sigma_det for run 2 (run 1): %f (%f)",
                    dict_pc_params_run2["sigma_det"],
                    dict_pc_params_run1["sigma_det"],
                )
            if pc_rlim_needs_2runs:
                _logger.info(
                    "pc_rlim for run 2 (run 1): %f (%f)", dict_pc_params_run2["rlim"], dict_pc_params_run1["rlim"]
                )
        out2 = decorated_pycosmic_det_cosmics(
            data=image2d, **{key: value for key, value in dict_pc_params_run2.items() if value is not None}
        )
        median2d_pycosmic2 = out2.data
        flag_pc2 = out2.mask.astype(bool)
        # combine results from both runs
        flag_pc = decorated_merge_peak_tail_masks(flag_pc, flag_pc2, pc_verbose)
        median2d_pycosmic = median2d_pycosmic2  # use the result from the 2nd run
    flag_pc = np.logical_and(flag_pc, bool_to_be_cleaned)
    flag_pc = flag_pc.flatten()
    datetime_end = datetime.now()
    delta_datetime = datetime_end - datetime_ini
    _logger.info("PyCosmic execution time: %s", str(delta_datetime))
    _logger.info(
        "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
        rlabel_pycosmic,
        np.sum(flag_pc),
        np.sum(flag_pc) / flag_pc.size * 100,
    )

    return median2d_pycosmic, flag_pc
