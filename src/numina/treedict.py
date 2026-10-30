#
# Copyright 2008-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""An implementation of hierarchical dictionary."""

import collections.abc


class TreeDict(collections.abc.MutableMapping):
    """
    Hierarchical dictionary
    """

    def __init__(self, *args, **keywords):
        self._data = dict(*args, **keywords)

    def __getitem__(self, key):
        """Return the value of `key`, a dotted path of keys in the nested dictionaries."""
        keys = key.split(".")
        return self._rec_getitem(self._data, keys[0], keys[1:])

    def __iter__(self):
        """Iterate over the top-level keys."""
        return iter(self._data)

    def __setitem__(self, key, value):
        """Set the value of `key`, a dotted path of keys, creating the intermediate nodes."""
        keys = key.split(".")
        return self._rec_setitem(self._data, value, keys[0], keys[1:])

    def __delitem__(self, key):
        """Do nothing, deleting keys is not implemented."""
        pass

    def __len__(self):
        """Return 1, the length is not implemented."""
        return 1

    def _rec_setitem(self, node, value, key, rest):
        """Set `value` in `node`, following the path of keys `key` and `rest`."""
        if rest:
            if key not in node:
                node[key] = type(self)()
            self._rec_setitem(node[key], value, rest[0], rest[1:])
        else:
            node[key] = value
            return

    def _rec_getitem(self, data, key, rest):
        """Return the value in `data` following the path of keys `key` and `rest`."""
        if rest:
            return self._rec_getitem(data[key], rest[0], rest[1:])
        else:
            return data[key]
