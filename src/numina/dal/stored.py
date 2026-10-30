#
# Copyright 2014-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import uuid

from numina.types.qc import QC

from .unserial import unserial


class StoredProduct:
    """A product returned from the DAL"""

    def __init__(self, id, content, tags, **kwds):
        """Store the id, content and tags of the product, the other keywords are ignored."""
        self.id = id
        self.content = content
        self.tags = tags


class StoredParameter:
    """A parameter returned from the DAL"""

    def __init__(self, content):
        self.content = content


class StoredResult:
    """Recover the RecipeResult values stored in a file"""

    def __init__(self):
        """Create an empty result, with QC ``UNKNOWN`` and a nil uuid."""
        self.qc = QC.UNKNOWN
        self.uuid = uuid.UUID("00000000-0000-0000-0000-000000000000")

    @classmethod
    def load_data(cls, state):
        """Create a result from `state`, the contents of a result file."""
        obj = cls.__new__(cls)
        obj._from_dict(state)
        return obj

    def _from_dict(self, state):
        """Set the QC, the uuid and the values of `state`, loading the values that are names of files."""
        self.qc = QC[state.get("qc", "UNKNOWN")]
        if "uuid" in state:
            self.uuid = uuid.UUID(state["uuid"])

        values = state.get("values", {})
        if isinstance(values, list):
            values = {o["name"]: o["content"] for o in values}
        for key, val in values.items():
            loaded = unserial(val)
            setattr(self, key, loaded)


# A translation of the entries of oblocks
# Notice that this is different to ObservationResult
# that contains the results of the reductions
class ObservingBlock:
    """Observing block, as read from a table of observing blocks.

    Only one of `images` and `children` is not empty.

    Parameters
    ----------
    id
        Id of the observing block.
    instrument : str
        Name of the instrument.
    mode : str
        Name of the observing mode.
    images : list
        Images of the block, also available as ``frames``.
    children : list
        Ids of the child observing blocks.
    parent : optional
        Id of the parent observing block.
    facts : optional
        Additional information of the block.
    """

    def __init__(self, id, instrument, mode, images, children, parent=None, facts=None):
        self.id = id
        self.instrument = instrument
        self.mode = mode
        # only one of files and children can
        # be different from []
        self.images = images
        self.frames = self.images
        self.children = children
        self.parent = parent
        self.facts = facts
