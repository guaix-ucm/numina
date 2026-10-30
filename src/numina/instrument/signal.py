#
# Copyright 2016-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import traceback
import collections


class Signal:
    """Signal used for callbacks."""

    def __init__(self):
        self.callbacks = collections.OrderedDict()
        self._cid_counter = 0

    def connect(self, callback):
        """Register ``callback`` and return its connection id."""
        self._cid_counter += 1
        cid = self._cid_counter
        self.callbacks[cid] = callback
        return cid

    def delete(self, idx):
        """Remove the callback with the connection id ``idx``."""
        del self.callbacks[idx]

    def emit(self, *args, **kwds):
        """Call the callbacks with the given arguments.

        Returns
        -------
        list of tuple
            The connection id and the return value of each callback. The callbacks that
            raise ``TypeError`` are skipped, and the traceback is printed.
        """
        result = []
        for cid, cb in self.callbacks.items():
            try:
                res = cb(*args, **kwds)
                result.append((cid, res))
            except TypeError:
                traceback.print_exc()
        return result
