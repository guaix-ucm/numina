#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Decorator to capture stdout and stderr of a function and log it."""

import io
import logging

from contextlib import redirect_stderr, redirect_stdout


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

