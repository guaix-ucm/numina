#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Execute Cosmic-CoNN cosmic ray detection algorithm."""

from datetime import datetime
import numpy as np

try:
    import cosmic_conn

    CONN_AVAILABLE = True
except ModuleNotFoundError as e:
    CONN_AVAILABLE = False


def execute_conn(image2d, bool_to_be_cleaned, rlabel_conn, dict_nn_params, _logger):
    """Execute Cosmic-CoNN cosmic ray detection algorithm."""
    if not CONN_AVAILABLE:
        raise ImportError(
            "Cosmic-CoNN is not installed. Please install Cosmic-CoNN to use\n"
            "the 'conn' or 'mm_conn' crmethod options.\n"
            "You can try installing it via pip:\n"
            "pip install cosmic-conn\n"
        )
    nn_verbose = dict_nn_params["verbose"]
    if nn_verbose:
        _logger.info("[green][Cosmic-CoNN parameters][/green]")
        for key in dict_nn_params.keys():
            _logger.info("%s for Cosmic-CoNN: %s", key, str(dict_nn_params[key]))
    # Execute Cosmic-CoNN
    _logger.info("using Cosmic-CoNN version: %s", cosmic_conn.__version__)
    time_ini = datetime.now()
    cr_model = cosmic_conn.init_model(dict_nn_params["model"])
    cr_prob = cr_model.detect_cr(image2d.astype(np.float32))
    flag_nn = cr_prob > dict_nn_params["threshold"]
    flag_nn = np.logical_and(flag_nn, bool_to_be_cleaned)
    flag_nn = flag_nn.flatten()
    datetime_end = datetime.now()
    delta_datetime = datetime_end - time_ini
    _logger.info("Cosmic-CoNN execution time: %s", str(delta_datetime))
    _logger.info(
        "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
        rlabel_conn,
        np.sum(flag_nn),
        np.sum(flag_nn) / flag_nn.size * 100,
    )

    return flag_nn
