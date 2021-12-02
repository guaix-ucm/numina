#
# Copyright 2016-2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import traceback
import collections


class Signal(object):
    """Signal used for callbacks."""
    def __init__(self):
        self.callbacks = collections.OrderedDict()
        self._cid_counter = 0

    def connect(self, callback):
        self._cid_counter += 1
        cid = self._cid_counter
        self.callbacks[cid] = callback
        return cid

    def delete(self, idx):
        del self.callbacks[idx]

    def emit(self, *args, **kwds):
        result = []
        for cid, cb in self.callbacks.items():
            try:
                res = cb(*args, **kwds)
                result.append((cid, res))
            except TypeError:
                traceback.print_exc()
        return result
