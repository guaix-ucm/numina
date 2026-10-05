#
# Copyright 2008-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""
Basic Data Products
"""

import warnings

from astropy.io import fits


class DataFrame:
    """A handle to a image in disk or in memory.

    An image opened from a file should be passed as `filename`. If it is
    passed as `frame`, numina can close it (after extracting metadata, for
    example) before its data are read, and a RuntimeWarning is emitted.
    """

    def __init__(self, frame=None, filename=None):
        if frame is None and filename is None:
            raise ValueError("only one in frame and filename can be None")
        if isinstance(frame, fits.HDUList) and len(frame) > 0 and frame.fileinfo(0) is not None:
            source = frame.fileinfo(0).get("filename")
            msg = (
                f"DataFrame created from an HDUList opened from '{source}', numina can close it "
                "before its data are read, use DataFrame(filename=...)"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        self.frame = frame
        self.filename = filename

    def open(self):
        if self.frame is None:
            return fits.open(self.filename, mode="readonly")
        else:
            return self.frame

    @property
    def label(self):
        return self.filename

    def __repr__(self):
        if self.frame is None:
            return f"DataFrame(filename={self.filename!r})"
        elif self.filename is None:
            return f"DataFrame(frame={self.frame!r})"
        else:
            fmt = "DataFrame(filename=%r, frame=%r)"
            return fmt % (self.filename, self.frame)

    def __numina_load__(self, obj):
        if obj is None:
            return None
        else:
            return DataFrame(filename=obj)
