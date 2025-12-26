#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Execute PyCosmic cosmic ray detection algorithm."""

from datetime import datetime
import numpy as np

try:
    import deepCR

    DEEPCR_AVAILABLE = True
except ModuleNotFoundError as e:
    DEEPCR_AVAILABLE = False


def execute_deepcr(image2d, bool_to_be_cleaned, rlabel_deepcr, dict_dc_params, _logger):
    """Execute DeepCR cosmic ray detection algorithm."""
    if not DEEPCR_AVAILABLE:
        raise ImportError(
            "DeepCR is not installed. Please install DeepCR to use\n"
            "the 'deepcr' or 'mm_deepcr' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install deepCR\n"
        )
    dc_verbose = dict_dc_params["verbose"]
    if dc_verbose:
        _logger.info("[green][DeepCR parameters][/green]")
        for key in dict_dc_params.keys():
            _logger.info("%s for deepCR: %s", key, str(dict_dc_params[key]))
    # Execute DeepCR
    _logger.info("using deepCR version: %s", deepCR.__version__)
    time_ini = datetime.now()
    mdl = deepCR.deepCR(mask=dict_dc_params["mask"])
    flag_dc, median2d_deepcr = mdl.clean(  # note the order of outputs!
        image2d,
        threshold=dict_dc_params["threshold"],
        inpaint=True,
    )
    flag_dc = np.logical_and(flag_dc, bool_to_be_cleaned)
    flag_dc = flag_dc.flatten()
    datetime_end = datetime.now()
    delta_datetime = datetime_end - time_ini
    _logger.info("deepCR execution time: %s", str(delta_datetime))
    _logger.info(
        "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
        rlabel_deepcr,
        np.sum(flag_dc),
        np.sum(flag_dc) / flag_dc.size * 100,
    )

    return median2d_deepcr, flag_dc
