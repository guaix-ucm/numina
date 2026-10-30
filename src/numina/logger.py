#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Extra logging handlers for the numina logging system."""

import logging

from astropy.io import fits


class FITSHistoryHandler(logging.Handler):
    """Logging handler using HISTORY FITS cards"""

    def __init__(self, header):
        """Add the messages as HISTORY cards to `header`."""
        logging.Handler.__init__(self)
        self.header = header

    def emit(self, record):
        """Add the formatted record to the header as a HISTORY card."""
        msg = self.format(record)
        self.header.add_history(msg)


def log_to_history(logger, name):
    """Decorate function, adding a logger handler stored in FITS."""

    def log_to_history_decorator(method):

        def l2h_method(self, ri):
            history_header = fits.Header()

            fh = FITSHistoryHandler(history_header)
            fh.setLevel(logging.INFO)
            logger.addHandler(fh)

            try:
                result = method(self, ri)
                field = getattr(result, name, None)
                if field:
                    with field.open() as hdulist:
                        hdr = hdulist[0].header
                        hdr.extend(history_header.cards)
                return result
            finally:
                logger.removeHandler(fh)

        return l2h_method

    return log_to_history_decorator


def has_rich_handler(logger):
    """True if logger, or the loggers its messages propagate to, have a RichHandler

    The handlers that show the messages of a logger are usually those of
    its parents, the root logger for a script, or 'numina' with numina run.
    """
    from rich.logging import RichHandler

    current = logger
    while current is not None:
        if any(isinstance(handler, RichHandler) for handler in current.handlers):
            return True
        if not current.propagate:
            break
        current = current.parent
    return False
